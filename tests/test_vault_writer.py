import unittest

from ingest.vault_writer import write_managed_note, source_id_for, _parse_frontmatter, _hash_body


class FakeObsidian:
    """In-memory fake matching ObsidianClient's async interface (list_dir,
    read_note, write_note) -- no live MCP server required."""

    def __init__(self, files=None):
        self.files = dict(files or {})  # path -> content string
        self.write_calls = []

    async def list_dir(self, path=""):
        prefix = f"{path}/" if path else ""
        names = set()
        for p in self.files:
            if not p.startswith(prefix):
                continue
            rest = p[len(prefix):]
            if "/" in rest:
                names.add(rest.split("/")[0] + "/")
            else:
                names.add(rest)
        return sorted(names)

    async def read_note(self, path):
        if path not in self.files:
            raise RuntimeError(f"Obsidian MCP tool 'vault_read' failed: not found: {path}")
        return {"content": self.files[path], "path": path}

    async def write_note(self, path, content):
        self.write_calls.append(path)
        self.files[path] = content
        return "ok"


def default_metadata(source_path="/tmp/doc.txt", source_sha256="abc123"):
    return {"source_path": source_path, "source_sha256": source_sha256,
            "prompt_version": 1, "structure_model": "gemma4:12b"}


class WriteManagedNoteTests(unittest.IsolatedAsyncioTestCase):
    async def test_creates_new_managed_note_when_target_does_not_exist(self):
        ob = FakeObsidian()
        result = await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "Some body text.", default_metadata())
        self.assertEqual(result["status"], "created")
        content = ob.files["Knowledge Cortex/Managed/doc.md"]
        fm, body = _parse_frontmatter(content)
        self.assertEqual(fm["cortex_managed"], "true")
        self.assertIn("Some body text.", body)

    async def test_managed_note_contains_exactly_one_frontmatter_block(self):
        ob = FakeObsidian()
        body_with_ingestion_style_content = (
            "# A Document\n\nSome content here about order 4821 and https://example.com/warranty."
        )
        await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md",
                                  body_with_ingestion_style_content, default_metadata())
        content = ob.files["Knowledge Cortex/Managed/doc.md"]
        self.assertEqual(content.count("---\n"), 2)  # exactly one block: one opening, one closing delimiter
        fm, body = _parse_frontmatter(content)
        self.assertNotIn("---", body)  # no second, nested frontmatter-looking block in the body
        self.assertIn("# A Document", body)

    async def test_refuses_to_overwrite_existing_unmanaged_note(self):
        ob = FakeObsidian(files={"Knowledge Cortex/Managed/doc.md": "# A human wrote this\n\nDo not touch."})
        result = await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "New generated body.", default_metadata())
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(ob.files["Knowledge Cortex/Managed/doc.md"], "# A human wrote this\n\nDo not touch.")
        self.assertEqual(ob.write_calls, [])

    async def test_updates_unchanged_managed_note_safely(self):
        ob = FakeObsidian()
        await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "Version one.", default_metadata())
        result = await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "Version two.", default_metadata())
        self.assertEqual(result["status"], "updated")
        _, body = _parse_frontmatter(ob.files["Knowledge Cortex/Managed/doc.md"])
        self.assertIn("Version two.", body)

    async def test_detects_human_modification_of_managed_note(self):
        ob = FakeObsidian()
        await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "Original body.", default_metadata())
        content = ob.files["Knowledge Cortex/Managed/doc.md"]
        fm, body = _parse_frontmatter(content)
        edited = content.replace(body.strip(), "Original body.\n\nA human added this sentence.")
        ob.files["Knowledge Cortex/Managed/doc.md"] = edited

        result = await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "New generated content.", default_metadata())
        self.assertEqual(result["status"], "conflict")

    async def test_conflict_does_not_alter_the_note(self):
        ob = FakeObsidian()
        await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "Original body.", default_metadata())
        content_before = ob.files["Knowledge Cortex/Managed/doc.md"]
        _, body = _parse_frontmatter(content_before)
        edited = content_before.replace(body.strip(), "Human edit here.")
        ob.files["Knowledge Cortex/Managed/doc.md"] = edited

        await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "New generated content.", default_metadata())
        self.assertEqual(ob.files["Knowledge Cortex/Managed/doc.md"], edited)

    async def test_unchanged_generated_content_skips_write(self):
        ob = FakeObsidian()
        await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "Stable content.", default_metadata())
        write_calls_after_first = len(ob.write_calls)
        result = await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "Stable content.", default_metadata())
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(len(ob.write_calls), write_calls_after_first)

    async def test_frontmatter_provenance_fields_are_preserved(self):
        ob = FakeObsidian()
        meta = default_metadata(source_path="/tmp/kc_notes/report.pdf", source_sha256="deadbeef")
        await write_managed_note(ob, "Knowledge Cortex/Managed/report.md", "Body.", meta)
        fm, _ = _parse_frontmatter(ob.files["Knowledge Cortex/Managed/report.md"])
        self.assertEqual(fm["cortex_source_path"], "/tmp/kc_notes/report.pdf")
        self.assertEqual(fm["cortex_source_sha256"], "deadbeef")
        self.assertEqual(fm["cortex_source_id"], source_id_for("/tmp/kc_notes/report.pdf"))
        self.assertEqual(fm["cortex_structure_model"], "gemma4:12b")
        self.assertEqual(fm["cortex_prompt_version"], "1")
        self.assertIn("cortex_last_write", fm)

    async def test_hash_comparison_ignores_volatile_metadata(self):
        ob = FakeObsidian()
        await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "Same body every time.", default_metadata())
        # A second write with the SAME body but at a later (different)
        # timestamp must still be recognized as unchanged, not a conflict,
        # even though cortex_last_write in the stored frontmatter differs
        # from what a naive whole-file hash would expect.
        result = await write_managed_note(ob, "Knowledge Cortex/Managed/doc.md", "Same body every time.", default_metadata())
        self.assertEqual(result["status"], "skipped")

    async def test_directory_style_multiple_managed_outputs(self):
        ob = FakeObsidian()
        r1 = await write_managed_note(ob, "Knowledge Cortex/Managed/a.md", "Doc A body.", default_metadata(source_path="/tmp/a.txt"))
        r2 = await write_managed_note(ob, "Knowledge Cortex/Managed/b.md", "Doc B body.", default_metadata(source_path="/tmp/b.txt"))
        self.assertEqual(r1["status"], "created")
        self.assertEqual(r2["status"], "created")
        self.assertEqual(len(ob.files), 2)

    async def test_one_conflict_does_not_block_unrelated_safe_writes(self):
        ob = FakeObsidian(files={"Knowledge Cortex/Managed/unmanaged.md": "# Human note"})
        r_conflict = await write_managed_note(ob, "Knowledge Cortex/Managed/unmanaged.md", "New body.", default_metadata(source_path="/tmp/x.txt"))
        r_ok = await write_managed_note(ob, "Knowledge Cortex/Managed/other.md", "Other body.", default_metadata(source_path="/tmp/y.txt"))
        self.assertEqual(r_conflict["status"], "conflict")
        self.assertEqual(r_ok["status"], "created")
        self.assertEqual(ob.files["Knowledge Cortex/Managed/unmanaged.md"], "# Human note")


class SourceIdTests(unittest.TestCase):
    def test_source_id_is_deterministic_and_short(self):
        a = source_id_for("/tmp/doc.txt")
        b = source_id_for("/tmp/doc.txt")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)

    def test_different_paths_produce_different_ids(self):
        self.assertNotEqual(source_id_for("/tmp/a.txt"), source_id_for("/tmp/b.txt"))


class ParseFrontmatterTests(unittest.TestCase):
    def test_no_frontmatter_returns_empty_dict_and_full_body(self):
        fm, body = _parse_frontmatter("Just plain content.")
        self.assertEqual(fm, {})
        self.assertEqual(body, "Just plain content.")

    def test_body_hash_ignores_frontmatter_block(self):
        content_a = "---\ncortex_last_write: 2026-01-01T00:00:00\n---\n\nSame body."
        content_b = "---\ncortex_last_write: 2026-06-06T12:00:00\n---\n\nSame body."
        _, body_a = _parse_frontmatter(content_a)
        _, body_b = _parse_frontmatter(content_b)
        self.assertEqual(_hash_body(body_a), _hash_body(body_b))


if __name__ == "__main__":
    unittest.main()
