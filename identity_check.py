"""Model-identity leakage detection for SFT data.

WHY THIS EXISTS
---------------
The previous check was `BAD_WORDS = ["chatgpt"]` + a lowercased substring test over
every turn's joined content+think. It was abandoned in practice because of false
positives. Measured on a 51,634-doc sample across all 95 catalogued sources:

  * it CRASHED whenever enabled -- `' '.join(...)` receives list/dict tool content,
    which process_entry explicitly permits, raising TypeError.
  * precision ~23%: it fired on 0.647% of docs, but most hits were users *asking
    about* ChatGPT, retrieved documents, and tool schemas -- not the model's own voice.
  * it missed the largest leak class entirely: generator provenance in the SYSTEM
    prompt ("You are powered by the model named moonshotai/Kimi-K2.6."), present in
    0.873% of docs and in 33.7% of one public agentic dataset.

DESIGN
------
Three ideas carry the precision:

1. SCOPE BY ROLE. Only the assistant's own voice can leak the assistant's identity.
   - assistant content / think / tool_call args -> self-ID    -> REJECT
   - system content                             -> provenance -> REJECT
   - user content, tool content, system tools   -> NEVER SCANNED
   164 of the old check's 334 rejections came from the never-scanned roles alone.

2. NO BARE VENDOR TOKEN MAY REJECT. A model name must sit inside a first-person or
   persona frame. Bare tokens are hopeless: `claude` appears in 3.65% of docs (81%
   with no AI context), `bard` in 5.41% (95% non-AI), `gpt` in 5.31%. A 24-term bare
   list was already tried and abandoned -- see LEGACY/process_oss_sources.py:10.

3. TIGHT FRAMES. The gap between "I am" and the model name is a whitelist of at most
   five connector words, not `[^.!?]{0,40}`. The loose form produced 88 false
   positives per 100 snippets ("This is a classic minimax problem").

Word boundaries everywhere: `anthropic` without \b matches PHIL-anthropic (205 hits
in 35k docs); `copilot` matches `copilots`. Only `chat ?-?gpt` and `moonshotai/` are
safe as substrings.
"""

import json
import re

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Model/product names. Grouped by the vendor families we actively care about:
# OpenAI/ChatGPT, Kimi/Moonshot, GLM/Zhipu -- plus families observed leaking in the
# corpus. Each alternative is inserted into a \b-anchored frame; none of these ever
# rejects on its own.
_MODEL_ALTS = [
    # --- OpenAI family -----------------------------------------------------
    r'chat\s?-?gpt',                 # also catches the "chat GPT" space-evasion
    r'gpt-?[3-5](?:\.\d+)?(?:\s*-?\s*(?:turbo|o|mini|nano))?',
    r'davinci', r'codex',
    # --- Kimi / Moonshot family -------------------------------------------
    r'kimi(?:\s*-?\s*k\d+(?:[.\-]\d+)?)?',
    r'moonshot(?:\s*-?\s*ai)?',
    # --- GLM / Zhipu family ------------------------------------------------
    r'chat\s?-?glm',
    r'glm-?\s?\d+(?:\.\d+)?(?:\s*-?\s*(?:air|flash|plus|turbo))?',
    r'zhipu(?:\s*ai)?',
    # --- other families seen leaking in the corpus -------------------------
    r'claude(?:\s*-?\s*(?:opus|sonnet|haiku|instant)\s*[\d.]*)?',
    r'nemotron', r'minimax(?:\s*-?\s*m\d+(?:\.\d+)?)?',
    r'deepseek(?:\s*-?\s*[vr]\d+(?:\.\d+)?)?',
    r'qwen\s?\d*(?:\.\d+)?', r'gemini', r'bard', r'grok', r'llama\s?\d*',
    r'mistral', r'copilot', r'ernie', r'doubao', r'hunyuan',
]
MODEL = '(?:' + '|'.join(_MODEL_ALTS) + ')'

# Organisations. Same rule: only meaningful inside a "created by" frame.
_VENDOR_ALTS = [
    r'open\s?ai', r'moonshot(?:\s*ai)?', r'zhipu(?:\s*ai)?', r'z\.ai',
    r'anthropic', r'nvidia', r'google(?:\s+deepmind)?', r'deepmind',
    r'meta(?:\s+ai)?', r'microsoft', r'alibaba', r'bytedance', r'baidu',
    r'tencent', r'mistral\s*ai', r'minimax(?:\s*ai)?', r'deepseek(?:\s*ai)?',
    r'x\.ai', r'cohere',
]
VENDOR = '(?:' + '|'.join(_VENDOR_ALTS) + ')'

# Connector words permitted between the first-person opener and the model name.
# Deliberately a closed whitelist -- this is what keeps precision at ~95%.
# NOTE: 'on' was removed from this whitelist 2026-08-08. With it, `self_id_name` matched
# browse-location phrasing ("I'm on Gemini's results page") and measured 95.4% (125/131);
# without it the rule is 100% with no measured recall loss -- "I'm based on GPT-4" still
# matches via 'based'.
_CONN = (r'(?:(?:a|an|the|your|just|actually|really|simply|currently|now|still|'
         r'version|variant|of|based|new|latest|underlying|ai|large|language|'
         r'model|assistant|chatbot|called|named|known|as|by|from)\s+){0,5}')

# The LEADING \b is load-bearing. Without it the `i\s*'?\s*m\b` alternative matches the
# tail of any word ending in "im" -- Prim, optim, trim, sim, dim -- so "Prim minimax"
# read as "i" + "m" + boundary + model name. Reported from a 106,814-trajectory run
# (FIELD_FINDINGS.md 1) where it produced the ONLY identity hit in the whole corpus and
# that hit was a false positive: `"""Prim minimax edge, matching internal_connectivity."""`
# inside generated code. Verified: 4/9 wrong before, 0/9 after, all true positives kept.
# `i\s*(?:['’]\s*)?m` and not `i\s*['’]?\s*m`: the latter puts two \s* next to each
# other with an optional character between them, so on a whitespace run the engine
# splits it every possible way before failing -- O(R^2). Measured on `'i' + ' '*R`:
# 3.6 ms at R=1,000, 881 ms at 16,000, 13.8 SECONDS at 64,000; the form below is
# 0.05 / 0.78 / 3.0 ms. Identical language, verified exhaustively over 402,233
# strings up to length 5 across the alphabet that can distinguish them
# (i m space tab newline ' ’ x I M a . 1): 0 mismatches in span or group.
_I_AM = r"\b(?:i\s*(?:['’]\s*)?m\b|i\s+am\b|my\s+name\s+is\b|you\s+can\s+call\s+me\b|i\s+go\s+by\b|i\s+am\s+called\b)"

# Plural first-person identity needs a tighter ending than the singular form.
# In real data, phrases such as "we are ChatGPT users" and "we are evaluating
# ChatGPT responses" describe people or a topic rather than the assistant's
# identity. Require either punctuation/end-of-line or an immediate model-role
# continuation ("we are ChatGPT playing the role ...").
_WE_ARE = r"\b(?:we\s*['’]\s*re\b|we\s+are\b)"
_PLURAL_SELF_ID_TAIL = (
    r'(?:\*\*|__|["\'])?'
    r'(?=(?:[ \t]*(?:[,.;:!?)\]}]|$)|[ \t]*[\r\n]|[ \t]+'
    r'(?:model|assistant|agent|chatbot|playing|acting|serving|responding|'
    r'tasked|asked|as|with|in|simulating|representing|assuming|taking)\b|'
    r'[ \t]+(?:and|but)\s+(?:i|we|must|need|will|should|can|have|are)\b))'
)

# ---------------------------------------------------------------------------
# Tier B/C -- assistant self-identification  -> REJECT
# ---------------------------------------------------------------------------

SELF_ID_PATTERNS = [
    # "I am ChatGPT", "I'm a version of the ChatGPT model", "my name is Kimi"
    ('self_id_name',
     re.compile(_I_AM + r'\s+' + _CONN + r'(?:\*\*|__|["\'])?' + MODEL + r'\b', re.I)),

    # "We are ChatGPT", "we're a version of Kimi", "we are ChatGPT playing
    # the role ...". The tail guard keeps plural descriptions such as "we are
    # ChatGPT users" and "we are Claude Monet scholars" out.
    ('self_id_plural_name',
     re.compile(_WE_ARE + r'\s+' + _CONN + r'(?:\*\*|__|["\'])?' + MODEL +
                r'\b' + _PLURAL_SELF_ID_TAIL, re.I)),

    # "the ChatGPT (us)" explicitly equates a named model with the assistant.
    # Keep the pronoun case-sensitive so geographical/product labels such as
    # "ChatGPT (US)" remain topical mentions.
    ('self_id_model_apposition',
     re.compile(r'\b(?:the\s+)?' + MODEL +
                r'\b\s*\(\s*(?-i:us|me)\s*\)', re.I)),

    # "As an AI built by OpenAI, I ..." / "As an AI model from Moonshot, we ...".
    #
    # NARROWED 2026-08-12: this rule now REQUIRES a model or vendor name. As
    # originally added it matched the bare frame "As an AI, I ..." with no model or
    # vendor token at all, which inverted this module's governing principle -- a
    # model name never rejects on its own, and neither should a first-person frame
    # on its own. Measured cost of the bare form: 3 hits in a 35-record
    # hand-verified corpus, 2 of them on records hand-labelled GOOD --
    #   "As an AI, I can't browse, but I might have seen a solution."
    #       (a competitive-programming record)
    #   "however, as an AI, I must work with what I have."
    #       (a generated agentic record)
    # Both are a model reasoning correctly about its own capability limits, which is
    # not identity leakage. Per the owner: "As an AI" is fine.
    #
    # The narrowed form still covers the real gap that motivated the rule --
    # "As an AI developed by OpenAI, I ..." is NOT caught by self_id_created_by,
    # which needs "I am/was developed by" rather than "As an AI developed by".
    #
    # Side effect, and it is the correct one: "As an AI language model, I do not
    # have opinions." no longer matches. It never did (the noun whitelist lacked
    # 'language'), and under the narrowed rule it should not -- there is no model or
    # vendor name in it.
    ('self_id_as_ai',
     re.compile(r'\bas\s+an\s+(?:ai|artificial(?:\s+|-)+intelligence)'
                r'(?:\s+(?:assistant|model|agent|chatbot|system|language\s+model))?'
                r'[^.\n!?]{0,40}?\b(?:' + MODEL + '|' + VENDOR + r')\b'
                r'[^.\n!?]{0,40}?,?\s+(?:i|we|my|our)\b', re.I)),

    # "I am/was created|made|developed|trained|built by OpenAI"
    ('self_id_created_by',
     re.compile(r"\b(?:i\s*['’]?\s*m\b|i\s+am\b|i\s+was\b|i\s+have\s+been\b)"
                r'[^.\n!?]{0,60}\b(?:created|made|developed|built|trained|designed|'
                r'produced|fine-?tuned)\s+by\s+(?:the\s+)?' + VENDOR + r'\b', re.I)),

    # "an AI assistant made by Anthropic" / "a large language model developed by OpenAI"
    ('self_id_ai_by_vendor',
     re.compile(r'\b(?:an?|the)\s+(?:ai|large\s+language|language)\s+'
                r'(?:assistant|model|agent|chatbot|system)\b[^.\n!?]{0,40}'
                r'\b(?:created|made|developed|built|trained|designed)\s+by\s+'
                r'(?:the\s+)?' + VENDOR + r'\b', re.I)),

    # "As ChatGPT, I ..." / "as chat GPT, we cannot ..." -- trailing pronoun is what
    # keeps "as Claude Monet" and "as Gemini (the constellation)" out.
    ('self_id_as_model',
     re.compile(r'\b(?:as|being|acting\s+as)\s+(?:\*\*)?' + MODEL +
                r'(?:\*\*)?\s*,?\s+(?:i|we|my|our)\b', re.I)),

    # REMOVED 2026-08-08: `self_id_my_creators` ("my creators at <VENDOR>").
    # Its ONLY hit in 231,275 docs was an HTML <meta> tag -- "Follow my developer journey"
    # followed by `<meta`, where `meta` is the VENDOR token. Precision 0/1. This is the
    # third rule in this detector to fail the same way as the deleted self_id_namespaced:
    # a bare vendor token matching a non-AI string.

    # "I'm running on Kimi-K2.6", "the model behind me is GLM-4" -- the assistant
    # naming its own serving identity.
    #
    # The frame ALONE is not enough. An earlier draft matched bare
    # "I'm running on" / "I'm powered by" and was ~100% false positives, because
    # the phrase overwhelmingly describes the ENVIRONMENT, not the model. Verified
    # from the corpus:
    #   "I don't know what model I'm running on from this context"  (discussing a
    #        tool's modelOverride parameter -- explicitly NOT self-identifying)
    #   "I'm running on Python 3.11"        -> interpreter version
    #   "I'm running on CPU with TF2.21"    -> hardware
    #   "I'm running on a newer Python3.12" -> version
    #   "the system reminder says I'm running on the user's computer"
    #   "the user thinks I'm running on a hotel concierge terminal" (quoting the user)
    #   "I'm powered by chaos"              -> a social-media POLL OPTION
    # So the model/vendor name must actually be present, within a few words.
    # The gate here is the MODEL/VENDOR requirement, not the width of the gap, so a
    # short lazy gap is safe and lets the name follow a comma or "is"
    # ("...the model I'm running on, moonshotai/Kimi-K2.6").
    ('self_id_running_model',
     re.compile(r'\b(?:i\s*[\'’]?\s*m|i\s+am|the\s+model\s+(?:i\s*[\'’]?\s*m\s+|i\s+am\s+)?)'
                r'\s*(?:running\s+on|powered\s+by|behind\s+me|based\s+on)\b'
                r'[^.\n!?]{0,22}?(?:[\w.\-]+/)?'
                r'(?:' + MODEL + r'|' + VENDOR + r')\b', re.I)),

    # REMOVED: `self_id_namespaced` -- matched any `vendor/name` path.
    #
    # It was the single largest rule (702 hits) and measurement showed it was
    # essentially ALL false positives, because `vendor/name` is the shape of
    # HuggingFace repos, GitHub repos, URL paths and package lists. Verified
    # examples, every one legitimate:
    #   'openai/gpt-4o'         -> a modelOverride CONFIG VALUE the agent was setting
    #   'openai/concepts'       -> URL path learn.microsoft.com/.../openai/concepts/...
    #   'openai/python-dotenv'  -> a requirements.txt package list
    #   'OpenAI/GPQA'           -> a DATASET name
    #   'OpenAI/Anthropic'      -> prose listing two companies with a slash
    #   'meta-llama/Llama-3-70b'-> a model id being discussed, not claimed
    # A namespaced id is evidence of TOPIC, not of self-identification. The genuine
    # case it was meant to catch (the assistant reading its own served model id) is
    # covered by self_id_running_model above when a first-person frame is present,
    # and the system-prompt origin of that leak is caught by PROVENANCE_PATTERNS.
]

# ---------------------------------------------------------------------------
# Generator provenance in the SYSTEM prompt  -> REJECT
# ---------------------------------------------------------------------------
# Agentic-harness boilerplate that leaked into generated data. This is common in
# some sources (33.7% of one public agentic dataset), so a calibration run
# will surface it loudly -- that is the intent. The gate does NOT rewrite data;
# `strip_provenance` below is exported for data owners who prefer to clean rather
# than drop, but process_entry never calls it.

# Sentence body that tolerates dots inside version numbers ("Kimi-K2.6",
# "MiniMax-M2.5") but still stops at a real sentence end (dot + space/EOL).
# Without the `\.(?=\d)` escape hatch, stripping "...named moonshotai/Kimi-K2.6."
# leaves a stray "6." behind.
_SENT = r'(?:[^.\n]|\.(?=\d))*\.?'

PROVENANCE_PATTERNS = [
    ('prov_powered_by',
     re.compile(_SENT + r'\byou\s+are\s+powered\s+by\s+the\s+model\s+named\b' + _SENT, re.I)),
    ('prov_exact_model_id',
     re.compile(_SENT + r'\bthe\s+exact\s+model\s+id\s+is\b' + _SENT, re.I)),
    ('prov_model_kv',
     re.compile(_SENT + r'\b(?:default_)?model\s*=\s*[\w./:-]*'
                r'(?:kimi|minimax|moonshot|qwen|deepseek|glm|zhipu|llama|mistral|'
                r'gpt-|claude|gemini|nemotron)[\w./:-]*' + _SENT, re.I)),
    # REMOVED 2026-08-08: `prov_you_are_model` ("You are <MODEL>, ...").
    # Hand-labelled 0/51 -- precision 0.0%. Every hit was a DATASET-AUTHORED synthetic
    # persona (a task that deliberately casts the assistant as a named product), not the
    # generator leaking its own identity. It bought 29 unique corpus rejections, all wrong,
    # concentrated in office-agent and terminal-agent sources. The other three
    # provenance rules are 100% precise (480/480) and cover the real harness banners.
]


# ---------------------------------------------------------------------------
# Prefilters
# ---------------------------------------------------------------------------
# Every SELF_ID / PROVENANCE pattern requires at least one of: a model name, a
# vendor name, or one of two fixed phrases. A single literal pass that looks for
# *any* of those lets us skip the ~11 expensive frame regexes on the overwhelming
# majority of documents, which contain none of them.
#
# This measurably matters: on the 512k bucket the unfiltered scan cost 141 ms/doc,
# i.e. 34% on top of the chat-template + encode that process_entry already pays.
#
# CORRECTNESS REQUIREMENT: the prefilter must be a strict SUPERSET of what the
# real patterns can match, or we get silent false negatives. It is built from the
# same MODEL / VENDOR alternations, so it cannot drift out of sync.
# Implemented as one str.lower() plus C-level substring searches rather than a
# regex alternation: measured 3x faster on the same input (0.04s vs 0.12s over
# 1.3 MB), because Python's `re` walks a 40-way alternation position by position
# while `in` uses an optimised substring search.
#
# Each entry must be a case-folded SUBSTRING of something the real patterns can
# match, so the filter can never produce a false negative:
#   'gpt'      covers chatgpt / chat-gpt / chat gpt / gpt-4 / gpt-5 ...
#   vendors    cover the "... created by X" frames
#   phrases    cover frames that need no model/vendor token
_TRIGGERS = (
    # model families
    'gpt', 'kimi', 'moonshot', 'glm', 'zhipu', 'claude', 'nemotron', 'minimax',
    'deepseek', 'qwen', 'gemini', 'bard', 'grok', 'llama', 'mistral', 'copilot',
    'ernie', 'doubao', 'hunyuan', 'davinci', 'codex',
    # vendors
    'openai', 'open ai', 'anthropic', 'nvidia', 'google', 'deepmind', 'meta',
    'microsoft', 'alibaba', 'bytedance', 'baidu', 'tencent', 'cohere',
    'x.ai', 'z.ai',
    # frames that contain no model/vendor token
    'running on', 'behind me',
    'my creator', 'my developer', 'my maker', 'my trainer', 'parent company',
    # 'as an ai' / 'as an artificial' removed 2026-08-12 with the narrowing of
    # self_id_as_ai: that rule now requires a model or vendor token, so the model
    # and vendor triggers above already reach every text it can match.
)

_PROV_TRIGGERS = ('powered by', 'exact model id', 'model=', 'model =', 'you are')


def _may_contain(text, triggers):
    low = text.lower()
    for t in triggers:
        if t in low:
            return True
    return False


def find_self_id(text):
    """Return list of (rule_name, matched_snippet) for assistant-voice identity leaks."""
    if not isinstance(text, str) or not text:
        return []
    if not _may_contain(text, _TRIGGERS):
        return []
    hits = []
    for name, rx in SELF_ID_PATTERNS:
        m = rx.search(text)
        if m:
            hits.append((name, m.group(0)[:120]))
    return hits


def find_provenance(text):
    """Return list of (rule_name, matched_sentence) for system-prompt provenance."""
    if not isinstance(text, str) or not text:
        return []
    if not _may_contain(text, _PROV_TRIGGERS):
        return []
    hits = []
    for name, rx in PROVENANCE_PATTERNS:
        for m in rx.finditer(text):
            hits.append((name, m.group(0)[:200]))
    return hits


def strip_provenance(text):
    """Remove provenance sentences from a system prompt.

    Returns (new_text, n_removed). Collapses the whitespace left behind so we do not
    manufacture the very trailing-whitespace artifacts the repetition checks hunt for.
    """
    if not isinstance(text, str) or not text:
        return text, 0
    n = 0
    out = text
    for _name, rx in PROVENANCE_PATTERNS:
        out, k = rx.subn('', out)
        n += k
    if n:
        out = re.sub(r'[ \t]{2,}', ' ', out)
        out = re.sub(r'\n{3,}', '\n\n', out)
        out = '\n'.join(ln.rstrip() for ln in out.split('\n')).strip()
    return out, n


def scan_conversation(conversation, think_key='think'):
    """Scan one conversation for identity problems.

    Returns dict:
      'self_id':    [(turn_idx, field, rule, snippet)]   -> caller should REJECT
      'provenance': [(turn_idx, rule, snippet)]          -> caller should REJECT

    Scanned: assistant content / think_key / tool_calls (self-ID); system content
    (provenance). NOT scanned: user content, tool content, system tools -- a user
    asking about ChatGPT is not the model leaking its identity.
    """
    self_id = []
    provenance = []
    for i, turn in enumerate(conversation):
        if not isinstance(turn, dict):
            continue
        role = turn.get('role')
        if role == 'assistant':
            for field in ('content', think_key):
                val = turn.get(field)
                if isinstance(val, str) and val:
                    for rule, snip in find_self_id(val):
                        self_id.append((i, field, rule, snip))
            tcs = turn.get('tool_calls')
            if tcs:
                try:
                    blob = json.dumps(tcs, ensure_ascii=False)
                except (TypeError, ValueError):
                    blob = ''
                for rule, snip in find_self_id(blob):
                    self_id.append((i, 'tool_calls', rule, snip))
        elif role == 'system':
            val = turn.get('content')
            if isinstance(val, str) and val:
                for rule, snip in find_provenance(val):
                    provenance.append((i, rule, snip))
    return {'self_id': self_id, 'provenance': provenance}
