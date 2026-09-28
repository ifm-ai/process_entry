"""Pure-function quality detectors for SFT reasoning/content.

DESIGN CONTRACT
---------------
* stdlib only, no tokenizer, no I/O  -> importable in a sweep over millions of
  docs without paying AutoTokenizer.from_pretrained (~seconds) per worker.
* every detector is a pure function of a string (or of a conversation) and
  returns a number. NO thresholds live in this module -- policy is applied by
  the caller (process_entry) so that thresholds can be swept independently.
* metric names are stable; they become keys in entry['quality_metrics'].

Metric orientation: for every metric here, LARGER == MORE SUSPICIOUS, except
`zlib_ratio` / `conv_zlib_ratio` / `distinct_line_ratio` where SMALLER == MORE
SUSPICIOUS. This is called out per-function and encoded in METRIC_DIRECTION.
"""

import bisect as _bisect
import json
import re
import zlib
from collections import Counter

# --------------------------------------------------------------------------
# tunables that are NOT thresholds -- they only gate when a metric is defined
# --------------------------------------------------------------------------

# Below this many bytes a compression ratio is dominated by zlib header
# overhead and is meaningless. Short strings report the neutral value 1.0.
ZLIB_MIN_BYTES = 200

# zlib level. 6 is the default; kept explicit so sweeps are reproducible.
ZLIB_LEVEL = 6

# Sentences shorter than this are ignored by the sentence-repeat detector:
# "Yes." or "OK." legitimately recur and would swamp the signal.
MIN_SENTENCE_CHARS = 25

# Character-run / token-run detectors only look for runs at least this long.
# Keeping the regex sparse makes it fast on megabyte strings.
MIN_CHAR_RUN = 8

# Truncation cap for the single-string scans. `None` == scan everything, which is
# the production setting. Every use below is a slice `s[:SCAN_CAP_CHARS]`, and
# `s[:None]` is the whole string, so None simply disables truncation.
#
# THIS WAS 400_000 AND THAT WAS A REAL RECALL HOLE. The old rationale -- "the first
# slice is more than enough to characterize them, a loop shows up early" -- is false,
# and was refuted by measurement 2026-08-08:
#
#   * With clean non-repetitive padding pushing the defect past 400k, EVERY class of
#     degeneracy became invisible. All five of trailing-space wall, glyph wall, token
#     wall, ellipsis wall and a sentence looped 300x were ACCEPTED when capped and
#     rejected when uncapped. The cap did not weaken detection, it removed it.
#   * It is not a rare shape. 1.17% of catalogue records (3,592 of 306,090 scanned)
#     have a single think/content field over 400k. It concentrates precisely where the
#     risk is: 86% of one long-proof source, 21% of a short-math source.
#   * It was actively hiding defects, not just theoretically. Uncapping newly rejects
#     51 of those 306,090 records, ALL in that long-proof source, and all hand-read as genuine:
#     single-turn records with 600k-1.2M char `think` that loop and never terminate --
#       e.g.     "so the statement is false." x26, "thus, i'm stuck." x8
#                "now, i'll write." x19, "now, i'll produce." x13
#                "maybe we can use the inequality:" x33
#                1.2M chars, "consider the convex hull." x30
#     Every one ends mid-deliberation ("Now, I'll produce the final message."). This is
#     non-terminating reasoning that loops -- so the cap was blindest in exactly the
#     population where a repetition gate has the most to catch.
#   * Removing it is nearly free: 1.02x gate time over 306,090 real records
#     (52.72 -> 53.64 ms/record), because 98.8% of records are under the cap anyway.
#
# Set it to an int only for a research sweep where bounded CPU matters more than
# recall. Production must leave it None.
SCAN_CAP_CHARS = None

# --------------------------------------------------------------------------
# regexes (compiled once)
# --------------------------------------------------------------------------

_CHAR_RUN_RE = re.compile(r'(.)\1{%d,}' % (MIN_CHAR_RUN - 1), re.DOTALL)
_ELLIPSIS_RUN_RE = re.compile(r'(?:\.\.\.|…)(?:\s*(?:\.\.\.|…))+')
_ELLIPSIS_TOK_RE = re.compile(r'\.\.\.|…')
_TRAIL_WS_RE = re.compile(r'\s+$')
_WS_ONLY_LINE_RE = re.compile(r'^[ \t]+$')
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
_WS_COLLAPSE_RE = re.compile(r'\s+')

# Unicode blocks that indicate a non-Latin script. Deliberately a coarse
# range check rather than `unicodedata.name()` (which is ~100x slower).
_NON_LATIN_RANGES = (
    (0x0400, 0x04FF),   # Cyrillic
    (0x0500, 0x052F),   # Cyrillic supplement
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0700, 0x074F),   # Syriac
    (0x0900, 0x097F),   # Devanagari
    (0x0E00, 0x0E7F),   # Thai
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0x3400, 0x4DBF),   # CJK ext A
    (0x4E00, 0x9FFF),   # CJK unified
    (0xA000, 0xA4CF),   # Yi
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK compat
    (0xFF66, 0xFF9F),   # halfwidth katakana
)


def _is_non_latin(cp):
    for lo, hi in _NON_LATIN_RANGES:
        if lo <= cp <= hi:
            return True
        if cp < lo:
            return False
    return False


# Named script ranges for the script-COUNT detector below. Kept separate from
# _NON_LATIN_RANGES because here we need to know WHICH script, not just "not Latin".
_SCRIPT_RANGES = (
    ('cjk', 0x4E00, 0x9FFF), ('cjk', 0x3400, 0x4DBF), ('cjk', 0xF900, 0xFAFF),
    ('kana', 0x3040, 0x30FF), ('kana', 0xFF66, 0xFF9F),
    ('hangul', 0xAC00, 0xD7AF), ('hangul', 0x1100, 0x11FF),
    ('cyrillic', 0x0400, 0x052F),
    ('arabic', 0x0600, 0x06FF), ('syriac', 0x0700, 0x074F),
    ('hebrew', 0x0590, 0x05FF),
    ('devanagari', 0x0900, 0x097F), ('bengali', 0x0980, 0x09FF),
    ('tamil', 0x0B80, 0x0BFF), ('telugu', 0x0C00, 0x0C7F),
    ('thai', 0x0E00, 0x0E7F), ('greek', 0x0370, 0x03FF),
    ('armenian', 0x0530, 0x058F), ('georgian', 0x10A0, 0x10FF),
    ('ethiopic', 0x1200, 0x137F), ('khmer', 0x1780, 0x17FF),
)

# A script must contribute at least this many characters before it counts. Without
# it, one stray glyph (a name, a symbol, a quoted character) would register as a
# whole writing system and destroy the signal.
MIN_CHARS_PER_SCRIPT = 5


def script_counts(s):
    """Character counts per non-Latin writing system present in `s`."""
    c = Counter()
    if not s:
        return c
    for ch in s[:SCAN_CAP_CHARS]:
        cp = ord(ch)
        if cp < 128:
            continue
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                c[name] += 1
                break
    return c


def n_distinct_scripts(s, min_chars=MIN_CHARS_PER_SCRIPT):
    """Number of distinct non-Latin writing systems with >= min_chars characters.

    LARGER == more suspicious. This is the sharpest gibberish detector measured so
    far, and it is preferable to `non_latin_letter_frac` because it does not punish
    legitimately non-English data.

    Evidence (a smoke test on 400 trajectories a colleague had
    already flagged by hand as containing gibberish/language mixing):

        degenerate token-salad turns : 7-12 distinct scripts
        clean turns in the same runs : 0-1
        multilingual SFT data (real) : max 3 (one doc at 4) out of 1800 sampled
        every other source sampled   : <= 2

    So >= 5 separates them with a wide empty margin either side. The failure mode it
    targets is INTERLEAVING of unrelated scripts (CJK + Cyrillic + Arabic + Hebrew +
    Devanagari + Thai in one field), which is what degenerate sampling produces;
    genuine multilingual text uses one script consistently.
    """
    return sum(1 for v in script_counts(s).values() if v >= min_chars)


# --------------------------------------------------------------------------
# single-string detectors
# --------------------------------------------------------------------------

def zlib_ratio(s):
    """Compressed size / raw size. SMALLER == more repetitive.

    Returns the neutral 1.0 for strings below ZLIB_MIN_BYTES, so callers can
    threshold without special-casing short turns.

    Caveat that motivated the rest of this module: zlib's window is 32 KiB, so
    a loop whose period exceeds that will NOT compress. Use alongside the
    structural detectors below rather than on its own.
    """
    if not s:
        return 1.0
    b = s.encode('utf-8', 'ignore')
    if len(b) < ZLIB_MIN_BYTES:
        return 1.0
    return len(zlib.compress(b, ZLIB_LEVEL)) / len(b)


def max_char_run(s):
    """Longest run of one repeated character. LARGER == more suspicious.

    Catches '====', '......', and the space/newline walls seen in the
    dsv32 sources.
    """
    if not s:
        return 0
    s = s[:SCAN_CAP_CHARS]
    best = 0
    for m in _CHAR_RUN_RE.finditer(s):
        n = len(m.group(0))
        if n > best:
            best = n
    return best


def max_token_run(s):
    """Longest run of the SAME whitespace-separated token repeated back to back.

    LARGER == more suspicious. This is the general form of the ellipsis
    pathology ('... ... ... ...' -> token '...' repeated N times).
    """
    if not s:
        return 0
    toks = s[:SCAN_CAP_CHARS].split()
    if not toks:
        return 0
    best = 1
    cur = 1
    prev = toks[0]
    for t in toks[1:]:
        if t == prev:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
            prev = t
    return best


def max_ellipsis_run(s):
    """Longest consecutive ellipsis-token run ('...' or '…', whitespace allowed).

    LARGER == more suspicious. Kept as its own metric (rather than relying on
    max_token_run) so numbers are directly comparable to the existing
    ellipsis audit, and because it tolerates newlines between tokens.
    """
    if not s:
        return 0
    s = s[:SCAN_CAP_CHARS]
    best = 0
    for m in _ELLIPSIS_RUN_RE.finditer(s):
        n = len(_ELLIPSIS_TOK_RE.findall(m.group(0)))
        if n > best:
            best = n
    return best


def line_stats(s):
    """(max_line_repeat, distinct_line_ratio) over non-blank lines.

    max_line_repeat LARGER == suspicious; distinct_line_ratio SMALLER ==
    suspicious. Blank lines are excluded so prose paragraphs don't skew it.
    """
    if not s:
        return 0, 1.0
    lines = [ln.strip() for ln in s[:SCAN_CAP_CHARS].split('\n')]
    lines = [ln for ln in lines if ln]
    if not lines:
        return 0, 1.0
    c = Counter(lines)
    return c.most_common(1)[0][1], len(c) / len(lines)


def max_sentence_repeat(s):
    """Max multiplicity of any normalized sentence >= MIN_SENTENCE_CHARS.

    LARGER == more suspicious. This is the detector aimed at circular
    reasoning: "We are stuck. Let's think if there is any other tool or data
    source we haven't considered." restated over and over.
    """
    if not s:
        return 0
    parts = _SENT_SPLIT_RE.split(s[:SCAN_CAP_CHARS])
    norm = []
    for p in parts:
        p = _WS_COLLAPSE_RE.sub(' ', p).strip().lower()
        if len(p) >= MIN_SENTENCE_CHARS:
            norm.append(p)
    if not norm:
        return 0
    return Counter(norm).most_common(1)[0][1]


_WORD_RE = re.compile(r'[A-Za-z]{2,}')
_CODEY_RE = re.compile(r'[(){}\[\];=<>|&$#@\\/]')

# A repeated unit only counts as PROSE if it carries real words. Without this filter,
# `max_sentence_repeat` is dominated by non-prose that legitimately repeats:
#   ARC grid rows  '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'  -> reaches x87
#   markdown fences '```'                                          -> x19
#   code lines     'for (int i = 0; i < n; ++i) {'                 -> x6
# Those forced the threshold up to 100, which permitted 99 repeats of a real sentence.
# Filtering to prose drops the legitimate ceiling dramatically and lets the threshold
# be tightened by roughly an order of magnitude.
MIN_PROSE_WORDS = 4
MIN_PROSE_ALPHA_FRAC = 0.45
MAX_PROSE_SYMBOLS = 4


# Code slips past a symbol-count test alone: 'def _generate(cls, strategy, params):'
# has 5 words, 78% alpha and only 3 symbols, yet is obviously not prose. Measured in
# the [10,20) band, code was still the dominant occupant. These three tests remove it.
# ONLY keywords that essentially never open an English sentence. Words like `if`,
# `for`, `return`, `use`, `type`, `class`, `do`, `then`, `let` were tried and removed:
# `let` alone destroyed the canonical loop phrase "Let's think if there is any other
# tool or data source we haven't considered." The structural tests below (call, snake
# _case, assignment, symbol density) catch the code those keywords were covering.
# Two more code shapes that leaked through and produced 4 of the 6 measured false
# positives at prose_line_repeat>=25 (agent-labelled sample):
#   'for nb in adj[node]:'                  -> subscript/index
#   'align: "center", valign: "middle"'     -> object-literal key:value
#   "category: 'trademarked product name'," -> same, in generated JS rule objects
_INDEX_RE = re.compile(r'\w\s*\[[^\]]*\]')
_KV_RE = re.compile(r'^\s*["\']?[\w.\-]+["\']?\s*:\s*["\'\[{]')

_CODE_KEYWORD_RE = re.compile(
    r'^\s*(?:def|elif|lambda|async|await|const|var|function|struct|enum|printf|'
    r'println|impl|fn|import|package|namespace|static|void|public|private|'
    r'protected|typedef|extern|goto)\b', re.I)
_CALL_RE = re.compile(r'\w\s*\([^)]*\)')          # foo(...) / foo (a, b)
_SNAKE_RE = re.compile(r'\b[a-z]+(?:_[a-z0-9]+)+\b')  # snake_case identifier
_ASSIGN_RE = re.compile(r'[^=!<>]=[^=]|->|::|\+=|=>')


# HARDENING PATCHES (2026-08-08). The surviving false positives of
# max_prose_sentence_repeat were is_prose leaks in two kinds:
#   CODE the original tests miss --
#     'body.append(new Paragraph('    unbalanced paren, so _CALL_RE never matches
#     'slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {'   _SNAKE_RE is lowercase-only
#     'for ing in self.ingredients:'  attribute access, no closing paren
#     'non-compliance detail: [Mandatory if No]'  _KV_RE handles single-token keys only
# Measured over 277 labelled rows, legacy -> hardened:
#   max_prose_sentence_repeat >= 20   precision 0.753 -> 0.831, 0 true positives lost
#   max_prose_line_repeat     >= 25   precision 0.855 -> 0.879, 1 true positive lost
#
# TRIED AND REJECTED -- a sixth patch rejecting MARKDOWN LIST ITEMS (`^[-*+] `).
# It targets real leaks (list items recurring once per distinct table row, e.g.
# '- All values appear valid.' x32) and on prose_sentence it looks free. On
# prose_line it is not: it costs 17 of 59 true positives against 1 without it,
# for +0.034 precision. Every one of those 17 is the teacher-student dialogue family
# that writes its degenerate loop AS bullets --
#     '*   *wait*, i'll write the response.'   x62
#     '*   **okay, i'll go with **ethylamine**.**'  x52
#     '*   i'll write the response.'           x87
# which is the single largest real defect family in the catalogue (that source
# loses 6.67%). Rejecting bullets would blind the gate to it. NOT APPLIED.
#
# ALSO TRIED AND REJECTED: "reject sentences ending in ':'" -- costs 3 true
# positives ('maybe we can use the inequality:' x23/x24) for 1 false positive.
_SNAKE_I = re.compile(r'\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b')   # case-insensitive snake
_DOTTED = re.compile(r'\b[A-Za-z_]\w*\.[A-Za-z_]\w*')        # attribute access
_KV_LOOSE = re.compile(r'^\s*[\w.\- ]{1,60}:\s*["\'\[{]')     # multi-word key: [val]


def is_prose(s):
    """True iff `s` reads as a natural-language sentence rather than data or code."""
    words = _WORD_RE.findall(s)
    if len(words) < MIN_PROSE_WORDS:
        return False
    if sum(len(w) for w in words) / max(len(s), 1) < MIN_PROSE_ALPHA_FRAC:
        return False
    if len(_CODEY_RE.findall(s)) >= MAX_PROSE_SYMBOLS:
        return False
    # code-shape tests
    if _CODE_KEYWORD_RE.match(s):
        return False
    if _CALL_RE.search(s):
        return False
    if _SNAKE_RE.search(s):
        return False
    if _ASSIGN_RE.search(s):
        return False
    if _INDEX_RE.search(s):
        return False
    if _KV_RE.match(s):
        return False
    # hardening patches -- see the note above
    if _SNAKE_I.search(s):
        return False
    if (s.count('(') != s.count(')') or s.count('[') != s.count(']')
            or s.count('{') != s.count('}')):
        return False
    if _DOTTED.search(s):
        return False
    if _KV_LOOSE.match(s):
        return False
    return True


def max_prose_sentence_repeat(s):
    """Max multiplicity of any repeated PROSE sentence. LARGER == more suspicious.

    The prose-only counterpart of `max_sentence_repeat`. This is the detector aimed at
    circular reasoning, and it catches both canonical loops:
      "Let's think if there is any other tool or data source we haven't considered."
      "Looking at the current state and need to fix the duplicated code."
    while ignoring grids, fences and code that repeat for legitimate reasons.
    """
    if not s:
        return 0
    parts = _SENT_SPLIT_RE.split(s[:SCAN_CAP_CHARS])
    norm = []
    for p in parts:
        p = _WS_COLLAPSE_RE.sub(' ', p).strip()
        if len(p) >= MIN_SENTENCE_CHARS and is_prose(p):
            norm.append(p.lower())
    if not norm:
        return 0
    return Counter(norm).most_common(1)[0][1]


def max_prose_line_repeat(s):
    """Max multiplicity of any repeated PROSE line. LARGER == more suspicious.

    Same rationale as above, applied to whole lines. Catches the short-filler loops
    ("Okay." x580 is excluded as non-prose, but is caught by max_line_repeat) while
    ignoring '```' fences and grid rows.
    """
    if not s:
        return 0
    lines = [ln.strip() for ln in s[:SCAN_CAP_CHARS].split('\n')]
    lines = [ln for ln in lines if ln and is_prose(ln)]
    if not lines:
        return 0
    return Counter(l.lower() for l in lines).most_common(1)[0][1]


def max_wordy_line_repeat(s):
    """Max multiplicity of a repeated line that contains at least one WORD.

    LARGER == more suspicious. This is the detector for SHORT-FILLER loops, which the
    prose variants cannot see (too short / not sentence-like) and which the raw
    `max_line_repeat` cannot isolate from legitimate punctuation repetition.

    The discriminator is simply "does the repeated line contain letters":

        FLAGGED (defects)              SPARED (legitimate, measured)
        'Okay.'   x495                 '}'      x236   closing braces
        '*Okay.'  x489                 '*****'  x239   separator rules
        'Row?'    x10247               '\\['     x633   LaTeX display math
                                       '```'    x19    markdown fences
                                       '0 0 0'  x28    ARC grid rows

    Raw `max_line_repeat` fired on every item in the right-hand column, which is why
    it is not usable as a gate.
    """
    if not s:
        return 0
    lines = [ln.strip() for ln in s[:SCAN_CAP_CHARS].split('\n')]
    lines = [ln for ln in lines if ln and _WORD_RE.search(ln)]
    if not lines:
        return 0
    return Counter(l.lower() for l in lines).most_common(1)[0][1]


def whitespace_stats(s):
    """Whitespace-abuse metrics. All LARGER == more suspicious.

    Returns dict:
      trailing_ws_len   - length of the trailing whitespace run
      trailing_ws_spaces- spaces/tabs within that run (the dsv32 signature)
      ws_only_line_frac - fraction of lines that are whitespace-only-but-nonempty
      max_space_run     - longest run of spaces/tabs anywhere

    `strip()` repairs the trailing case but NOT interior whitespace walls,
    which is why ws_only_line_frac / max_space_run exist.
    """
    if not s:
        return {'trailing_ws_len': 0, 'trailing_ws_spaces': 0,
                'ws_only_line_frac': 0.0, 'max_space_run': 0}
    capped = s[:SCAN_CAP_CHARS]
    m = _TRAIL_WS_RE.search(s)
    run = m.group(0) if m else ''
    lines = capped.split('\n')
    ws_only = sum(1 for ln in lines if _WS_ONLY_LINE_RE.match(ln))
    max_sp = 0
    for mm in re.finditer(r'[ \t]{4,}', capped):
        n = len(mm.group(0))
        if n > max_sp:
            max_sp = n
    return {
        'trailing_ws_len': len(run),
        'trailing_ws_spaces': run.count(' ') + run.count('\t'),
        'ws_only_line_frac': (ws_only / len(lines)) if lines else 0.0,
        'max_space_run': max_sp,
    }


def non_latin_letter_frac(s):
    """Fraction of *letters* that belong to a non-Latin script.

    LARGER == more suspicious for an English-intended source. Deliberately
    letter-relative (not char-relative) so that code, punctuation, digits and
    whitespace -- which dominate SWE data -- do not dilute the signal.
    Returns 0.0 when there are too few letters to judge.
    """
    if not s:
        return 0.0
    letters = 0
    non_latin = 0
    for ch in s[:SCAN_CAP_CHARS]:
        cp = ord(ch)
        if cp < 128:
            if ch.isalpha():
                letters += 1
            continue
        if _is_non_latin(cp):
            letters += 1
            non_latin += 1
        elif ch.isalpha():
            letters += 1
    if letters < 20:
        return 0.0
    return non_latin / letters


# Metrics that no shipped gate reads. Skipping them saves 14% of the bundle
# (measured on 475 real catalogue records: 39.9 -> 34.5 ms/record). Less than the
# profile suggests at first glance -- non_latin_letter_frac shows a large cumtime,
# but most of it is script_counts, which n_distinct_scripts still needs.
# They stay computed BY DEFAULT so research sweeps and every recorded measurement
# keep their meaning; only the production gate asks for the lean set.
GATE_SKIP = ('zlib_ratio', 'non_latin_letter_frac')


# ===========================================================================
# REFORMULATED RUN DETECTORS (2026-08-08)
# ===========================================================================
# Each replaces a gate whose measured precision was poor. The originals are kept
# (research and the recorded measurements depend on them); only the POLICY in
# repetition_check.py switched over.
#
#   char_run  -> glyph_run        precision 0.35 -> 1.00, corpus 49 -> 6 docs
#   space_run -> trailing_space   precision 0.20 -> 0.80, corpus 394 -> 133 docs
#   token_run -> unfenced_token   precision 0.65 -> 1.00, corpus 34 -> 20 docs

# ASCII rule / leader / underline characters. A long run of these is typography
# (a `# =====` banner, a `___` form blank, a `...` dot leader), not generation.
RULE_CHARS = set('=-_*#~+.|<>/')


def max_glyph_run(s):
    """Longest run of a repeated character that is neither whitespace nor an ASCII
    rule/leader/underline character. LARGER == more suspicious.

    WHY THIS REPLACED max_char_run: at char_run >= 500 the corpus flagged 49 docs at
    precision 0.35, and 42 of the 49 (86%) had run character ' ' -- char_run was
    operating as a lower-precision duplicate of the space gate and inheriting its
    entire pasted-DataFrame false-positive population. The two non-space false
    positives were typography: '=' x761 (a banner the task ASKED the model to emit)
    and '_' x632 (a form blank in a pasted template).

    Exhaustive corpus at >= 1000: SIX docs, all hand-verified defects --
      '\\' x60040, '0' x10626, U+200F x2345 (RLM wall), '\u6816' x1880, '5' x1589.
    Longest legitimate glyph run measured anywhere is 426 ('a' x426, a regex
    catastrophic-backtracking test string), then 379 and 307. 1000 sits in a 2.3x
    empty margin either side.
    """
    if not s:
        return 0
    best = 0
    for m in _CHAR_RUN_RE.finditer(s[:SCAN_CAP_CHARS]):
        c = m.group(1)
        if c.isspace() or c in RULE_CHARS:
            continue
        n = len(m.group(0))
        if n > best:
            best = n
    return best


def max_trailing_space_run(s):
    """Longest run of spaces/tabs TERMINATING a line that carries printable content.
    LARGER == more suspicious.

    WHY THIS REPLACED max_space_run: space_run >= 100 flagged 394 corpus docs at
    precision 0.20 -- the worst gate in the set. 207 of the 394 came from
    file-manipulation and software-engineering sources, and every one sampled was GOOD: the model
    QUOTES a pandas or openpyxl render inside its reasoning and the run is column
    padding ("...NaN<623 spaces>NaN..."), LaTeX \\begin{array} padding, or deep code
    indentation. The actual defect is a run that TERMINATES a line -- the dsv32
    emission wall ("Let's try `...`.<2069 spaces>").

    Measured on the labelled rows, the INTERIOR component of max_space_run (printable
    content on BOTH sides) has 0 true positives and 16 false positives at >= 40. It has
    never once been right.

    Whitespace-only lines are skipped, so pasted indentation cannot trigger it.
    Exhaustive corpus at >= 100: 133 docs (from 394), 124 of them
    single-generator emission walls -- 97% of the defect family preserved while 201 of the 207 paste
    false positives disappear.

    TRIED AND REJECTED: a ">= 4 alphabetic words on the line" guard. It removes only 3
    of the 6 residual false positives and destroys 36 real walls that follow short
    sentences ("Continue."<316 spaces>).
    """
    if not s:
        return 0
    best = 0
    for ln in s[:SCAN_CAP_CHARS].split('\n'):
        st = ln.rstrip(' \t')
        if not st:
            continue
        n = len(ln) - len(st)
        if n > best:
            best = n
    return best


_NULL_LIT_RE = re.compile(r'^(?:None|nan|NaN|NULL|null|nil|NA|N/A)[,;]?$')
_SHORT_DOTS_RE = re.compile(r'^\.{1,2}$')
_FENCE_RE = re.compile(r'```')
_TOKEN_RE = re.compile(r'\S+')


def _pasted_token(tok):
    """Filler tokens whose long runs are table structure, never generation."""
    return bool(_NULL_LIT_RE.match(tok) or _SHORT_DOTS_RE.match(tok))


def max_unfenced_token_run(s):
    """Longest back-to-back repeat of one whitespace-separated token, IGNORING runs
    inside a ``` fence and runs of pasted-table filler tokens.

    WHY THIS REPLACED max_token_run: of the 34 corpus docs at token_run >= 200, 13 were
    false positives and every one was a QUOTED data structure -- either inside a fence
    (a markdown table's '|' column, a C macro concatenation example, a
    `new double[] { 0, 0, ... }` literal, an ASCII Ishikawa diagram) or unfenced pasted
    table filler ('None,' x2201 from an openpyxl row repr, '.' x846 dot leaders from a
    PDF table of contents).

    Exhaustive corpus at >= 200: 20 docs (from 34), ALL hand-verified defects.
    Precision 1.00 on both the corpus and the labelled sample.

    Excluding '.'/'..' costs nothing -- the '...' walls that matter are 3+ dots and are
    independently caught by max_ellipsis_run. The fence rule costs exactly one true
    positive ('Digital' x432), which max_wordy_line_repeat still rejects at 436.

    TRIED AND REJECTED: the "repeated token must be word-like" analogue of the
    wordy_line_repeat fix. It reaches only 0.67 because most real walls are punctuation
    ('`' x15878, '"?":' x23305, ')' x201).
    """
    if not s:
        return 0
    s = s[:SCAN_CAP_CHARS]
    fences = [m.start() for m in _FENCE_RE.finditer(s)]
    toks = [(m.group(0), m.start()) for m in _TOKEN_RE.finditer(s)]
    best = 0
    i = 0
    n = len(toks)
    while i < n:
        j = i
        while j + 1 < n and toks[j + 1][0] == toks[i][0]:
            j += 1
        cnt = j - i + 1
        if (cnt > best and not _pasted_token(toks[i][0])
                and _bisect.bisect_right(fences, toks[i][1]) % 2 == 0):
            best = cnt
        i = j + 1
    return best


# ===========================================================================
# SHINGLE DUPLICATION -- the delimiter-independent repetition detector
# ===========================================================================
# EVERY other repetition detector here keys on a delimiter: a repeated CHARACTER
# (char/glyph run), a WHITESPACE-DELIMITED TOKEN (token run), a LINE, or a
# >=25-char SENTENCE. A substring that repeats without any of those boundaries
# matches none of them and is completely invisible.
#
# The case that motivated this, measured:
#   a record from a multilingual SFT source:
#   817,794 chars of assistant content, "DataGridViewTextBoxColumn" x32,630, no
#   whitespace anywhere in the wall. Metrics: max_char_run 68, max_token_run 1,
#   max_prose_sentence_repeat 0, max_wordy_line_repeat 2 -> scan_repetition() = [].
#   A record that is 100% degenerate passed every gate cleanly.
# It is not a one-off: a census of 2,000 records of that source
# found 7 more gate-accepted pure walls.
#
# Shingling is delimiter-free, so it sees all of them: the 8 confirmed walls score
# 0.963-0.997 duplicate fraction.
#
# NOTE ON zlib: this class IS very compressible (the 817k record is 0.0047), so a
# low zlib gate would also catch it. Shingling is preferred because it degrades
# gracefully -- a wall with jitter defeats compression-style thinking less
# predictably, and zlib was already measured useless at separating legitimate code
# (0.109) from a known-bad loop (0.202). See the rejected-detector notes in
# repetition_check.py.
SHINGLE_K = 24          # substring length; >= the length of a typical identifier
SHINGLE_STRIDE = 8      # sampling stride, so cost is len/8 not len
SHINGLE_MIN_CHARS = 4000  # below this, ordinary prose repeats enough to be noisy
SHINGLE_MIN_COUNT = 32    # need enough shingles for the fraction to mean anything


def max_shingle_dup_frac(s, k=SHINGLE_K, stride=SHINGLE_STRIDE,
                         min_chars=SHINGLE_MIN_CHARS):
    """Fraction of sampled k-char shingles that are duplicates. LARGER == worse.

    1.0 means every sampled window is a repeat of one seen earlier; 0.0 means all
    distinct. Returns 0.0 for anything shorter than `min_chars`, so short fields
    -- where ordinary prose legitimately repeats -- can never trip it.

    Measured on the 8 confirmed no-delimiter walls: 0.963 to 0.997. Ordinary
    records sit far below; see the threshold note in repetition_check.py for the
    false-positive census.
    """
    if not s or len(s) < min_chars:
        return 0.0
    n = len(s) - k + 1
    if n <= 0:
        return 0.0
    total = 0
    seen = set()
    for i in range(0, n, stride):
        seen.add(s[i:i + k])
        total += 1
    if total < SHINGLE_MIN_COUNT:
        return 0.0
    return 1.0 - len(seen) / total


def text_metrics(s, skip=()):
    """All single-string metrics for one field, as a flat dict.

    `skip` names metrics to omit. Omitted keys are ABSENT from the result rather
    than None, so a caller that thresholds on a skipped metric silently gates
    nothing instead of comparing against a wrong value.
    """
    mlr, dlr = line_stats(s)
    ws = whitespace_stats(s)
    out = {
        'len': len(s or ''),
        'max_char_run': max_char_run(s),
        'max_glyph_run': max_glyph_run(s),
        'max_trailing_space_run': max_trailing_space_run(s),
        'max_token_run': max_token_run(s),
        'max_unfenced_token_run': max_unfenced_token_run(s),
        'max_ellipsis_run': max_ellipsis_run(s),
        'max_line_repeat': mlr,
        'distinct_line_ratio': dlr,
        'max_sentence_repeat': max_sentence_repeat(s),
        'max_prose_sentence_repeat': max_prose_sentence_repeat(s),
        'max_prose_line_repeat': max_prose_line_repeat(s),
        'max_wordy_line_repeat': max_wordy_line_repeat(s),
        'max_shingle_dup_frac': max_shingle_dup_frac(s),
        'n_distinct_scripts': n_distinct_scripts(s),
        **ws,
    }
    if 'zlib_ratio' not in skip:
        out['zlib_ratio'] = zlib_ratio(s)
    if 'non_latin_letter_frac' not in skip:
        out['non_latin_letter_frac'] = non_latin_letter_frac(s)
    return out


# --------------------------------------------------------------------------
# conversation-level detectors
# --------------------------------------------------------------------------

def _canon_args(args):
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(args)


def _turn_call_sig(turn):
    """Byte-identical signature of every tool call on one turn, as a tuple."""
    out = []
    for tc in (turn.get('tool_calls') or []):
        if not isinstance(tc, dict):
            continue
        fn = tc.get('function') if isinstance(tc.get('function'), dict) else tc
        out.append(f'{fn.get("name")}|{_canon_args(fn.get("arguments"))}')
    return tuple(out)


def tool_call_repetition(conversation):
    """Repeated tool calls. LARGER == more suspicious.

    max_identical_tool_call      - max multiplicity of one (name, exact args)
                                   across the whole trajectory
    max_consec_identical_call    - longest back-to-back run of the same
                                   (name, exact args)
    n_tool_calls                 - denominator

    NOTE: this is genuinely new. The existing Rule 3 in tool_call_recovery.py
    only fires for schema-FAILING calls whose response errored, only compares
    the immediately-following assistant turn, and matches argument KEYS rather
    than values -- so a healthy trajectory repeating one identical call 50x is
    invisible to it.
    """
    sigs = []
    for turn in conversation:
        if not isinstance(turn, dict) or turn.get('role') != 'assistant':
            continue
        for tc in (turn.get('tool_calls') or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get('function') if isinstance(tc.get('function'), dict) else tc
            name = fn.get('name')
            args = fn.get('arguments')
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (TypeError, ValueError):
                    pass
            sigs.append((name, _canon_args(args)))
    n_assist = sum(1 for t in conversation
                   if isinstance(t, dict) and t.get('role') == 'assistant')
    if not sigs:
        return {'max_identical_tool_call': 0, 'max_consec_identical_call': 0,
                'n_tool_calls': 0, 'identical_call_frac': 0.0}
    counts = Counter(sigs)
    best_consec = 1
    cur = 1
    for i in range(1, len(sigs)):
        if sigs[i] == sigs[i - 1]:
            cur += 1
            if cur > best_consec:
                best_consec = cur
        else:
            cur = 1
    top = counts.most_common(1)[0][1]
    return {
        'max_identical_tool_call': top,
        'max_consec_identical_call': best_consec,
        'n_tool_calls': len(sigs),
        # Fraction of the trajectory spent re-issuing ONE identical call.
        #
        # This exists because the absolute count cannot separate the two
        # populations -- measured overlap in the 5-15 band:
        #   LEGITIMATE iterate-and-check      DEGENERATE loop
        #   go build   9 / 107 turns (0.08)   click{index:209} 48 / 52  (0.92)
        #   read file 11 / 105 turns (0.10)   make -j4         47 / 110 (0.43)
        #   pytest    13 /  73 turns (0.18)   pnpm install     47 / 133 (0.35)
        #   verify.py  5 /  83 turns (0.06)   create_po        40 /  13 (3.08)
        # The ratio splits them cleanly at ~0.30 where the raw count does not.
        # Can exceed 1.0 when a turn fires several calls (batch loops).
        'identical_call_frac': (top / n_assist) if n_assist else 0.0,
    }


def think_dup_stats(conversation, think_key='think'):
    """(max_think_dup, dup_unit_chars, dup_frac, dup_call_share).

    The extra three features exist because the RAW COUNT cannot separate the two
    populations -- both live at 10-30. max_think_dup >= 10 flagged 160 corpus docs at
    precision 0.41, and every false positive was a terse action label repeated while the
    trajectory ADVANCED: "let me check.", "open result 0.", "continue.".

    The hand labels name the discriminator: a defect re-issues the SAME call on the
    duplicated turns (dead reasoning channel), while terse narration issues a DIFFERENT
    call each time. So the missing feature is not the reasoning at all -- it is the
    ACTION taken on the turns that carry the duplicated reasoning. Over the 126 labelled
    rows with count >= 8:
      24 of 40 BAD  have dup_call_share == 1.00 with exactly ONE distinct signature
      53 of 55 GOOD have dup_call_share <= 0.62, the mass sitting at 0.04-0.31
    Two near-identical records from one STEM agent source make the point:
      record A "let me check:" x24, all 24 turns read /app/result.json   -> BAD
      record B x25, every duplicated turn issues a different call        -> GOOD

    dup_call_share is forced to 0.0 when the modal signature is EMPTY -- otherwise every
    non-agentic record with a repeated think would score 1.00. That guard costs nothing:
    all 6 labelled BAD records with an empty modal signature have unit length 255-1272
    and are caught by the length clause instead.

    TRIED AND REJECTED (negative result worth keeping): excluding ADJACENT duplicates as
    a harness artifact. Collapsing adjacent repeats drops precision to 0.06 and recall to
    0.08 -- the real defects ARE adjacent ("i'll answer." x808 with an adjacent run of
    808) while the false positives are interleaved.
    """
    norm, sigs, thinks = [], [], []
    n_assist = 0
    for turn in conversation:
        if not isinstance(turn, dict) or turn.get('role') != 'assistant':
            continue
        n_assist += 1
        th = turn.get(think_key) or ''
        t = _WS_COLLAPSE_RE.sub(' ', th).strip().lower() if isinstance(th, str) else ''
        norm.append(t)
        sigs.append(_turn_call_sig(turn))
        if t:
            thinks.append(t)
    if not thinks:
        return 0, 0, 0.0, 0.0
    unit, n = Counter(thinks).most_common(1)[0]
    dup_sigs = [sg for t, sg in zip(norm, sigs) if t == unit]
    share = 0.0
    if len(dup_sigs) >= 2:
        modal, top = Counter(dup_sigs).most_common(1)[0]
        if modal:
            share = top / len(dup_sigs)
    return n, len(unit), (n / n_assist) if n_assist else 0.0, share


def content_dup_run_stats(conversation):
    """(max_run, unit_chars, ctx_modal_share, n_ctx_distinct) for CONTENT across turns.

    The `think_dup_stats` of the answer channel, with two deliberate differences.

    KEYED ON THE BACK-TO-BACK RUN, not the modal count. A raw cross-turn count
    cannot separate a loop from terse narration repeated while the trajectory
    ADVANCES -- the same finding that put CONSEC_IDENTICAL_CONJUNCT on the
    tool-call rule. Measured: keying on the count instead of the run admits 189
    extra records whose duplicated turns are interleaved with real work (count up
    to 68 at a longest run of 3).

    THE SECOND RETURN PAIR IS THE ENVIRONMENT, NOT THE ACTION. For the reasoning
    channel, `dup_call_share` asks "did the agent re-issue the same call?". The
    obvious content analogue -- "did the agent make no call at all?" -- is WRONG,
    and this is the negative result this detector exists to record: the harness
    that produces this defect (terminus-2) puts its command batch INSIDE the
    content JSON, so `tool_calls` is empty on every record in the subset, including
    records where the agent is actively driving the terminal. Measured: empty-call
    share is 1.00 on 650 of 650 flagged records, so the conjunct is inoperative --
    it cannot tell "no action" from "advancing action". Using it would have shipped
    a rule whose only real condition was the run length.

    What does discriminate is whether the ENVIRONMENT moved. So for each turn of
    the winning run we take the normalized content of the immediately preceding
    NON-assistant turn (the observation the agent was responding to) and return
    how concentrated those observations are. A wedged agent sees the same dead
    terminal every turn; an agent polling a long-running job sees a different
    observation every turn even though its own status line is identical.

    Separation is bimodal, with nothing in between:
        stuck loops              ctx_modal_share 0.90 - 1.00   (median 1.00)
        legitimate polling       ctx_modal_share 0.11 - 0.58
    The clean class this rescues is real: one agentic-coding record emits 8 identical
    `{"analysis": "Progressing within ~94%. Continue.", ...}` turns while issuing
    `sleep 55; tail -2 /tmp/pytest_full2.log` each time and watching the pytest dot
    count advance -- ctx_modal_share 0.12, correctly spared. It also spares the
    anti-sycophancy class (a model correctly HOLDING its answer against 8 different
    user challenges scores 0.125) WITHOUT any length threshold, which is why this
    detector has no min-unit-chars knob: a floor was measured at 2 true defects lost
    against 1 clean class saved, and it was upper-bounded at 111 chars by the very
    record it exists to catch.

    USER AND TOOL TURNS ARE READ HERE ONLY AS EXCULPATORY CONTEXT. They can raise
    ctx_modal_share but the rule only ever fires on repetition in the ASSISTANT's
    own content, so a repetitive user or tool turn can never by itself cause a
    rejection. The "never gate user/tool turns" rule is intact.
    """
    norm, ctx = [], []
    last_obs = ''
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        if turn.get('role') != 'assistant':
            v = turn.get('content')
            last_obs = (_WS_COLLAPSE_RE.sub(' ', v).strip().lower()
                        if isinstance(v, str) else '')
            continue
        co = turn.get('content')
        norm.append(_WS_COLLAPSE_RE.sub(' ', co).strip().lower()
                    if isinstance(co, str) else '')
        ctx.append(last_obs)

    best_len, best_end, cur_len = 0, -1, 0
    cur_unit = None
    for i, s in enumerate(norm):
        if s and s == cur_unit:
            cur_len += 1
        else:
            cur_unit, cur_len = (s if s else None), (1 if s else 0)
        if cur_len > best_len:
            best_len, best_end = cur_len, i
    if best_len == 0:
        return 0, 0, 0.0, 0

    window = ctx[best_end - best_len + 1:best_end + 1]
    modal_share = 0.0
    if window:
        modal_share = Counter(window).most_common(1)[0][1] / len(window)
    return (best_len, len(norm[best_end]), modal_share, len(set(window)))


def cross_turn_metrics(conversation, think_key='think'):
    """Trajectory-level repetition across assistant turns.

    max_think_dup   - max multiplicity of one normalized reasoning block
                      (LARGER == suspicious). Catches a model restating the
                      same paragraph turn after turn.
    conv_zlib_ratio - zlib ratio over all assistant text concatenated
                      (SMALLER == suspicious). Catches globally-repetitive
                      trajectories whose individual turns each look fine.
    n_assistant_turns
    """
    thinks = []
    all_text = []
    n_assist = 0
    for turn in conversation:
        if not isinstance(turn, dict) or turn.get('role') != 'assistant':
            continue
        n_assist += 1
        th = turn.get(think_key) or ''
        co = turn.get('content') or ''
        if isinstance(th, str) and th.strip():
            thinks.append(_WS_COLLAPSE_RE.sub(' ', th).strip().lower())
            all_text.append(th)
        if isinstance(co, str) and co:
            all_text.append(co)
    max_dup = Counter(thinks).most_common(1)[0][1] if thinks else 0
    return {
        'max_think_dup': max_dup,
        'conv_zlib_ratio': zlib_ratio('\n'.join(all_text)),
        'n_assistant_turns': n_assist,
    }


def conversation_metrics(conversation, think_key='think', skip=()):
    """Full metric bundle for one record.

    Per-field metrics are reduced across assistant turns by taking the WORST
    value seen (max for larger-is-worse metrics, min for ratio metrics), so a
    single degenerate turn is not averaged away by many healthy ones. The
    reducing turn index is reported for auditability.
    """
    worst = {}
    worst_turn = {}
    for idx, turn in enumerate(conversation):
        if not isinstance(turn, dict) or turn.get('role') != 'assistant':
            continue
        for field in ('think', 'content'):
            key = think_key if field == 'think' else 'content'
            val = turn.get(key)
            if not isinstance(val, str) or not val:
                continue
            m = text_metrics(val, skip)
            for name, v in m.items():
                mk = f'{field}_{name}'
                lower_is_worse = name in ('zlib_ratio', 'distinct_line_ratio')
                if mk not in worst:
                    worst[mk] = v
                    worst_turn[mk] = idx
                elif (v < worst[mk]) if lower_is_worse else (v > worst[mk]):
                    worst[mk] = v
                    worst_turn[mk] = idx
    out = dict(worst)
    out.update(cross_turn_metrics(conversation, think_key))
    _n, _L, _fr, _sh = think_dup_stats(conversation, think_key)
    out.update({'dup_unit_chars': _L, 'dup_frac': _fr, 'dup_call_share': _sh})
    out.update(tool_call_repetition(conversation))
    out['_worst_turn'] = worst_turn
    return out


# Orientation map so sweep/report code never has to guess which way a
# threshold points. 'lo' = flag when value <= threshold; 'hi' = flag when >=.
METRIC_DIRECTION = {
    'zlib_ratio': 'lo',
    'conv_zlib_ratio': 'lo',
    'distinct_line_ratio': 'lo',
}


def direction_for(metric_name):
    base = metric_name.split('_', 1)[1] if metric_name.startswith(('think_', 'content_')) else metric_name
    return METRIC_DIRECTION.get(base, METRIC_DIRECTION.get(metric_name, 'hi'))
