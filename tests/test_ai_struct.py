import os
import unittest

from ingest.ai_struct import structure_markdown, StructureResult, _validate, _significant_tokens


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeOllamaChat:
    """content_fn(text) -> structured markdown string, or raises to simulate failure."""

    def __init__(self, content_fn):
        self.content_fn = content_fn
        self.calls = []
        self.calls_kwargs = []

    def post(self, url, json, timeout):
        self.calls.append(json)
        self.calls_kwargs.append({"url": url, "json": json, "timeout": timeout})
        user_text = next(m["content"] for m in json["messages"] if m["role"] == "user")
        content = self.content_fn(user_text)
        return FakeResponse({"message": {"content": content}})


class StructureMarkdownTests(unittest.TestCase):
    def test_successful_structuring_returns_ok_true_and_structured_text(self):
        original = "Meeting notes: call vendor at 555-1234 on 2026-01-15 for order #4821."
        http = FakeOllamaChat(lambda t: "# Meeting Notes\n\n" + t)
        result = structure_markdown(original, http_client=http, model="test-model")
        self.assertTrue(result.ok)
        self.assertIn("# Meeting Notes", result.text)
        self.assertIn(original, result.text)
        self.assertEqual(result.model, "test-model")
        self.assertIsNone(result.reason)

    def test_network_failure_falls_back_to_original_text(self):
        class BrokenHttp:
            def post(self, *a, **k):
                raise ConnectionError("no route to host")
        original = "Some raw content."
        result = structure_markdown(original, http_client=BrokenHttp())
        self.assertFalse(result.ok)
        self.assertEqual(result.text, original)
        self.assertIn("failed", result.reason)

    def test_empty_output_falls_back_to_original_text(self):
        original = "Some raw content with real information in it."
        http = FakeOllamaChat(lambda t: "")
        result = structure_markdown(original, http_client=http)
        self.assertFalse(result.ok)
        self.assertEqual(result.text, original)

    def test_whitespace_only_output_falls_back(self):
        original = "Some raw content with real information in it."
        http = FakeOllamaChat(lambda t: "   \n\n  ")
        result = structure_markdown(original, http_client=http)
        self.assertFalse(result.ok)
        self.assertEqual(result.text, original)

    def test_drastically_shortened_output_is_rejected(self):
        original = "x" * 500 + " important details here with numbers 12345 and more content padding."
        http = FakeOllamaChat(lambda t: "Summary.")
        result = structure_markdown(original, http_client=http)
        self.assertFalse(result.ok)
        self.assertIn("too short", result.reason)

    def test_output_dropping_a_url_is_rejected(self):
        original = "See https://example.com/docs for reference material and more context padding here."
        http = FakeOllamaChat(lambda t: "# Notes\n\nSee the docs for reference material and context padding here.")
        result = structure_markdown(original, http_client=http)
        self.assertFalse(result.ok)
        self.assertIn("missing", result.reason)

    def test_output_dropping_a_number_is_rejected(self):
        original = "Order quantity: 4821 units, ship by 2026-01-15 with plenty of extra descriptive padding text."
        http = FakeOllamaChat(lambda t: "# Order\n\nShip the units by the deadline with extra descriptive padding text.")
        result = structure_markdown(original, http_client=http)
        self.assertFalse(result.ok)
        self.assertIn("missing", result.reason)

    def test_output_preserving_all_values_with_reorg_is_accepted(self):
        original = "Call 4821 at https://example.com on 2026-01-15 regarding the order."
        http = FakeOllamaChat(lambda t: (
            "# Order Follow-up\n\n## Summary\n\n"
            "Call 4821 at https://example.com on 2026-01-15 regarding the order."
        ))
        result = structure_markdown(original, http_client=http)
        self.assertTrue(result.ok)

    def test_code_block_content_must_survive_unchanged(self):
        original = "Run this:\n\n```bash\ncurl -X POST http://host:1234/api\n```\n\nThat's it."
        http = FakeOllamaChat(lambda t: "# Instructions\n\n" + t)
        result = structure_markdown(original, http_client=http)
        self.assertTrue(result.ok)
        self.assertIn("```bash\ncurl -X POST http://host:1234/api\n```", result.text)

    def test_empty_input_is_not_sent_to_the_model(self):
        http = FakeOllamaChat(lambda t: "should not be called")
        result = structure_markdown("   ", http_client=http)
        self.assertFalse(result.ok)
        self.assertEqual(len(http.calls), 0)

    def test_model_defaults_and_override(self):
        http = FakeOllamaChat(lambda t: "# X\n\n" + t)
        result = structure_markdown("content here", http_client=http, model="custom-model")
        self.assertEqual(result.model, "custom-model")


class TimeoutConfigurationTests(unittest.TestCase):
    """Hardening pass: configurable timeout with precedence
    explicit arg > env var > default (300s)."""

    def setUp(self):
        for var in ("AI_STRUCTURE_TIMEOUT_SECONDS",):
            self._orig = os.environ.pop(var, None)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        if self._orig is not None:
            os.environ["AI_STRUCTURE_TIMEOUT_SECONDS"] = self._orig
        else:
            os.environ.pop("AI_STRUCTURE_TIMEOUT_SECONDS", None)

    def test_default_timeout_is_300_seconds_when_nothing_set(self):
        http = FakeOllamaChat(lambda t: "# X\n\n" + t)
        structure_markdown("content here", http_client=http)
        self.assertEqual(http.calls_kwargs[-1]["timeout"], 300)

    def test_env_var_overrides_default(self):
        os.environ["AI_STRUCTURE_TIMEOUT_SECONDS"] = "45"
        http = FakeOllamaChat(lambda t: "# X\n\n" + t)
        structure_markdown("content here", http_client=http)
        self.assertEqual(http.calls_kwargs[-1]["timeout"], 45.0)

    def test_explicit_argument_overrides_env_var(self):
        os.environ["AI_STRUCTURE_TIMEOUT_SECONDS"] = "45"
        http = FakeOllamaChat(lambda t: "# X\n\n" + t)
        structure_markdown("content here", http_client=http, timeout=7)
        self.assertEqual(http.calls_kwargs[-1]["timeout"], 7)

    def test_invalid_env_var_falls_back_to_default_without_crashing(self):
        os.environ["AI_STRUCTURE_TIMEOUT_SECONDS"] = "not-a-number"
        http = FakeOllamaChat(lambda t: "# X\n\n" + t)
        result = structure_markdown("content here", http_client=http)
        self.assertTrue(result.ok)
        self.assertEqual(http.calls_kwargs[-1]["timeout"], 300)

    def test_zero_or_negative_env_var_falls_back_to_default(self):
        os.environ["AI_STRUCTURE_TIMEOUT_SECONDS"] = "-5"
        http = FakeOllamaChat(lambda t: "# X\n\n" + t)
        structure_markdown("content here", http_client=http)
        self.assertEqual(http.calls_kwargs[-1]["timeout"], 300)


class ValidationHelperTests(unittest.TestCase):
    def test_significant_tokens_extracts_urls_numbers_dates_code(self):
        text = "See https://a.com on 2026-01-15 for item 4821 and `inline_code` and version 1.5"
        tokens = _significant_tokens(text)
        self.assertIn("https://a.com", tokens)
        self.assertIn("2026-01-15", tokens)
        self.assertIn("4821", tokens)
        self.assertIn("`inline_code`", tokens)
        self.assertIn("1.5", tokens)

    def test_validate_accepts_identical_text(self):
        text = "Some content with number 4821 in it that is long enough to pass the ratio check easily."
        self.assertIsNone(_validate(text, text))

    def test_validate_rejects_none_output(self):
        self.assertIsNotNone(_validate("some source text", None))


if __name__ == "__main__":
    unittest.main()
