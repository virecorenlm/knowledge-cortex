import unittest

from ingest.chunk import build_chunks, chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_splits_into_bounded_pieces_preserving_all_characters(self):
        text = "x" * 2500
        parts = chunk_text(text, max_chars=1000)
        self.assertEqual(len(parts), 3)
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(len(p) <= 1000 for p in parts))

    def test_empty_text_yields_no_chunks(self):
        self.assertEqual(chunk_text(""), [])

    def test_rejects_nonpositive_max_chars(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", max_chars=0)


class BuildChunksTests(unittest.TestCase):
    def test_reconstructs_full_text_in_order(self):
        text = "a" * 50 + "b" * 50 + "c" * 50
        chunks = build_chunks("note.md", text, max_chars=50)
        self.assertEqual(len(chunks), 3)
        rebuilt = "".join(c["text"] for c in sorted(chunks, key=lambda c: c["chunk_index"]))
        self.assertEqual(rebuilt, text)

    def test_whitespace_only_chunks_are_dropped(self):
        text = "real content" + " " * 40 + "more"
        chunks = build_chunks("note.md", text, max_chars=12)
        self.assertTrue(all(c["text"].strip() for c in chunks))

    def test_same_content_same_path_produces_identical_ids(self):
        a = build_chunks("note.md", "hello world", max_chars=1000)
        b = build_chunks("note.md", "hello world", max_chars=1000)
        self.assertEqual([c["id"] for c in a], [c["id"] for c in b])

    def test_different_content_same_path_changes_ids(self):
        a = build_chunks("note.md", "hello world", max_chars=1000)
        b = build_chunks("note.md", "goodbye world", max_chars=1000)
        self.assertNotEqual([c["id"] for c in a], [c["id"] for c in b])

    def test_different_path_same_content_changes_ids(self):
        a = build_chunks("note-a.md", "hello world", max_chars=1000)
        b = build_chunks("note-b.md", "hello world", max_chars=1000)
        self.assertNotEqual([c["id"] for c in a], [c["id"] for c in b])

    def test_rejects_empty_path(self):
        with self.assertRaises(ValueError):
            build_chunks("", "hello", max_chars=1000)

    def test_payload_carries_source_and_hash(self):
        chunks = build_chunks("note.md", "hello", max_chars=1000, source="obsidian")
        self.assertEqual(chunks[0]["source"], "obsidian")
        self.assertEqual(len(chunks[0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
