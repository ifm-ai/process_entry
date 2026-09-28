"""Focused regression tests for model-identity leakage detection."""

import unittest

from identity_check import find_self_id, scan_conversation


class FindSelfIdTest(unittest.TestCase):

    def assert_rejected_by(self, text, rule):
        self.assertIn(rule, {name for name, _snippet in find_self_id(text)})

    def assert_kept(self, text):
        self.assertEqual(find_self_id(text), [])

    def test_plural_self_identification_is_rejected(self):
        cases = (
            "We are ChatGPT.",
            "We're Kimi, ready to help.",
            "The system message says we are ChatGPT.",
            "We are ChatGPT playing the role of Lina Romero.",
            "We are ChatGPT simulating an OASIS account.",
            "We are ChatGPT representing the account.",
            "We're ChatGPT in the role of a moderator.",
            "We are ChatGPT and need to choose an action.",
            "We are the GPT-4 model.",
            "We’re actually a version of Claude Sonnet.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assert_rejected_by(text, "self_id_plural_name")

    def test_plural_topical_mentions_are_kept(self):
        cases = (
            "We are evaluating ChatGPT responses.",
            "We are ChatGPT users testing plugins.",
            "We are ChatGPT and Claude users testing plugins.",
            "We are comparing ChatGPT and Claude.",
            "We are Claude Monet scholars.",
            "The system message says ChatGPT should be discussed.",
            "The answers were ChatGPT.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assert_kept(text)

    def test_model_apposition_is_rejected(self):
        self.assert_rejected_by(
            "The user is instructing the ChatGPT (us) to act as an account.",
            "self_id_model_apposition",
        )
        self.assert_rejected_by(
            "The prompt calls Claude (me) the moderator.",
            "self_id_model_apposition",
        )

    def test_uppercase_us_apposition_is_kept(self):
        self.assert_kept("Please compare ChatGPT (US) and ChatGPT (EU).")

    def test_as_ai_with_model_or_vendor_is_rejected(self):
        """self_id_as_ai requires a MODEL or VENDOR name.

        Narrowed 2026-08-12. The rule originally matched the bare frame
        ("As an AI, I ...") with no model or vendor token, which inverted this
        module's governing principle: a model name never rejects on its own, and
        neither should a first-person frame on its own. See
        test_bare_as_ai_frame_is_kept for the measured cost of the bare form.

        The narrowed rule still covers the gap that motivated it --
        "As an AI developed by OpenAI, I ..." is NOT caught by
        self_id_created_by, which needs "I am/was developed by".
        """
        cases = (
            "As an AI developed by OpenAI, I cannot help with that.",
            "As an AI model from Moonshot AI, we must decline.",
            "As an AI assistant built by Anthropic, I will refuse.",
            "As an artificial intelligence created by Google, my answer is no.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assert_rejected_by(text, "self_id_as_ai")

    def test_bare_as_ai_frame_is_kept(self):
        """"As an AI" on its own is NOT identity leakage.

        These are a model reasoning correctly about its own capability limits, and
        the bare form cost 3 hits in a 35-record hand-verified corpus, 2 of them on
        records hand-labelled GOOD:
          "As an AI, I can't browse, but I might have seen a solution."
              (a competitive-programming record)
          "however, as an AI, I must work with what I have."
              (a generated agentic record)
        """
        cases = (
            "As an AI, I cannot claim personal experience.",
            "As an AI, I can't browse, but I might have seen a solution.",
            "however, as an AI, I must work with what I have.",
            "As an AI assistant, we should be transparent.",
            "Speaking as an artificial intelligence model, I cannot browse.",
            "As an artificial-intelligence agent, our role is to help.",
            # never matched (the noun whitelist lacked 'language'), and under the
            # narrowed rule it correctly should not -- no model or vendor name.
            "As an AI language model, I do not have opinions.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assert_kept(text)

    def test_as_ai_profession_is_kept(self):
        self.assert_kept("As an AI researcher, I study model behavior.")

    def test_existing_direct_self_identification_still_rejects(self):
        self.assert_rejected_by("I am ChatGPT.", "self_id_name")
        self.assert_rejected_by("As ChatGPT, we cannot do that.", "self_id_as_model")

    def test_bare_model_discussion_is_kept(self):
        cases = (
            "ChatGPT is a language model.",
            "The user asks about ChatGPT.",
            "The user asks me to compare ChatGPT and Claude.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assert_kept(text)

    def test_scan_conversation_rejects_plural_identity_in_think(self):
        result = scan_conversation(
            [
                {"role": "user", "content": "Choose an action."},
                {
                    "role": "assistant",
                    "content": "like 1",
                    "think": "We are ChatGPT playing the role of an account.",
                },
            ]
        )
        self.assertEqual(result["provenance"], [])
        self.assertEqual(result["self_id"][0][1:3], ("think", "self_id_plural_name"))


if __name__ == "__main__":
    unittest.main()
