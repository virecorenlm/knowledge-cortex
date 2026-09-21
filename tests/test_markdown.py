import unittest

from ingest.markdown import to_markdown, split_frontmatter


class SplitFrontmatterTests(unittest.TestCase):
    def test_splits_to_markdown_output_into_frontmatter_and_body(self):
        md = to_markdown("hello world", source="/tmp/x.txt")
        frontmatter, body = split_frontmatter(md)
        self.assertTrue(frontmatter.startswith("---\n"))
        self.assertTrue(frontmatter.endswith("---"))
        self.assertIn("source: /tmp/x.txt", frontmatter)
        self.assertNotIn("source:", body)
        self.assertNotIn("ingested:", body)
        self.assertIn("hello world", body)

    def test_reattaching_frontmatter_and_body_reconstructs_original(self):
        md = to_markdown("some content here", source="/tmp/y.txt")
        frontmatter, body = split_frontmatter(md)
        self.assertEqual(frontmatter + "\n\n" + body, md)

    def test_content_with_no_frontmatter_is_returned_unchanged(self):
        frontmatter, body = split_frontmatter("Just plain content, no frontmatter.")
        self.assertEqual(frontmatter, "")
        self.assertEqual(body, "Just plain content, no frontmatter.")

    def test_ingestion_timestamp_never_leaks_into_the_split_body(self):
        # Two calls to to_markdown() at different (real) timestamps must
        # still produce the SAME body once frontmatter is split off --
        # confirms the volatile "ingested:" timestamp lives only in the
        # frontmatter half, never in what downstream hashing/structuring
        # sees as the document body.
        md_a = to_markdown("identical content", source="/tmp/z.txt")
        md_b = to_markdown("identical content", source="/tmp/z.txt")
        _, body_a = split_frontmatter(md_a)
        _, body_b = split_frontmatter(md_b)
        self.assertEqual(body_a, body_b)


if __name__ == "__main__":
    unittest.main()
