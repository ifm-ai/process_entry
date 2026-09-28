"""Degenerate-repetition gate for SFT records.  REJECT ONLY -- never repairs.

This is the POLICY layer over quality_checks.py (which holds the pure
detectors and no thresholds). Everything here is a number that was chosen from
measurement, and each one carries the measurement that chose it. If you change a
number, re-run `random/process_entry/verify_the_uglies.py` -- it asserts that a
13-file corpus of real defects still gets caught and that 17 real good records
still pass.

WHAT THIS CATCHES
-----------------
Training data that has degenerated into a loop: a model that emitted the same
sentence 174 times, a `think` field that is 1.9M characters of newlines, a
trajectory that called the same tool with the same arguments 400 times.

A model trained on this learns to imitate it, which is the whole reason the gate
exists. Note the converse, which is a deliberate design position: LENGTH IS NOT A
DEFECT. Long reasoning with no loop in it is fine, and there is no length, ratio
or growth threshold anywhere in this module. Every rule here measures repetition
or a run -- a model going round without progressing -- never size.

WHAT IT DELIBERATELY DOES NOT TOUCH
-----------------------------------
* USER turns and TOOL responses. A user pasting a repetitive log, or a tool
  returning 10k identical rows, is not a defect in the target -- and gating on
  it was the single largest false-rejection source in early versions.
* CONTENT is gated, but on only four metrics at 2-3x looser thresholds than
  reasoning (see CONTENT_THRESHOLDS). Measured: content is 3-7x cleaner than
  reasoning for loop-shaped repetition, but its character/token-flood tail is
  actually FATTER, so those two gates stay.
* Nothing is rewritten. A repair would hide the defect from the person who owns
  the source; a rejection surfaces it in their calibration run, where they can
  fix the generator, pre-clean, or relax the check for that source.

CALIBRATION
-----------
Validated 2026-08-08 by running the gate over 378,976 real records across all 95
catalogue sources, ARC repetition-exempt. This is the WHOLE gate, repetition plus
identity:

    doc-weighted    0.156%      token-weighted   0.266%
    32 of 95 sources lose NOTHING

Read the split before reacting to that number -- it is mostly not repetition:

    identity_provenance                3,318   71% of all rejections
    identity_self_id                     299
    all repetition rules combined      1,074   23%

So the repetition gate itself costs roughly 0.03% doc-weighted. The identity
majority is one narrow, source-side problem: generator banners left in system
prompts (one public agentic dataset loses 32.6%, a file-manipulation source ~9.5%),
which is a fix for the data owner, not a threshold to loosen.

Per-rule firing rates, same scan, as a share of records scanned:

    think_max_prose_sentence_repeat    0.093%    content_max_prose_sentence_repeat 0.013%
    think_max_wordy_line_repeat        0.090%    content_max_wordy_line_repeat     0.006%
    identical_tool_call                0.053%    content_max_char_run              0.005%
    think_max_prose_line_repeat        0.032%    content_max_token_run             0.003%
    max_think_dup                      0.016%    content_max_shingle_dup_frac      0.015%
    consecutive_identical_tool_call    0.014%    think_max_shingle_dup_frac        0.003%
    think_max_trailing_space_run       0.007%    think_max_unfenced_token_run      0.002%
    think_max_glyph_run                0.000%    think_n_distinct_scripts          0.000%

The reformulated run gates firing at ~0 is the intended result, not a broken rule:
they were rewritten to stop matching quoted data structures, so what remains is
the small set of genuine walls.

Rejections concentrate rather than spread, which is the point -- they are a signal
about specific generators, not a tax on the corpus. Sources losing most, and why:

    public agentic dataset           32.6%  generator banner in the system prompt
    browser-agent source             13.3%  browser agent stuck on one element
    file-manipulation source         ~9.5%  generator banner
    teacher-student dialogues         5.7%  reasoning is "*   Okay." x320
    long-proof source                 2.3%  long proofs that loop and never terminate

All hand-read and true positives. Run a calibration pass before a big generation
run and READ the rejection reasons -- a source losing >1% is telling you something
about its generator.

COST: ~35 ms/record for the metric bundle, a few percent on top of process_entry
(tokenization dominates). Rejected records are cheaper than accepted ones, because
the gate runs before the chat-template render.

SOURCES THAT NEED AN EXEMPTION
------------------------------
ARC-AGI sources: reasoning quotes puzzle grids
verbatim, so row and token repetition is inherent to the task, not a defect --
one record repeats a grid row 10,247 times and is CORRECT. Pass
`repetition_check=False` for those sources. Exempting them is what let
think_max_token_run tighten from 1000 to 200 for everything else.
"""

from quality_checks import (conversation_metrics, think_dup_stats,
                            content_dup_run_stats, GATE_SKIP)

# --------------------------------------------------------------------------
# REASONING thresholds. Tighter than content: reasoning is where loops start,
# and reasoning has no formatting contract with the user.
# --------------------------------------------------------------------------
REASONING_THRESHOLDS = {
    # p999 = 19 over the catalogue once non-prose is excluded. The prose filter
    # is what makes this usable: the raw sentence-repeat metric was dominated by
    # things that legitimately repeat (ARC grid rows x87, '```' fences x19, code
    # lines), which is what forced the earlier absurd threshold of 100.
    'think_max_prose_sentence_repeat': 20,

    # 25, NOT 15. An earlier pass set 15 from BAND-stratified labels, which
    # overweight the low bands. Re-sampled UNIFORMLY from the flagged population,
    # precision at 15 is 61.2% and at 25 it is 82.5%; at 15 this was the single
    # largest false-positive source in the whole gate set (183 fired, 46 false,
    # and 37 of its 75 SOLO firings wrong). Two independent measurements agreed.
    'think_max_prose_line_repeat': 25,

    # Short filler loops ("Okay.", "Let me check.") that the prose-line metric
    # misses because they are too short to be prose. Requires >=2 words, so the
    # raw metric's false hits ('}' x236, '*****' x239, '\\[' x633) do not apply.
    'think_max_wordy_line_repeat': 100,

    # THREE REFORMULATED RUN GATES (2026-08-08). Each replaces a raw run metric
    # whose measured precision was poor, because the raw metric could not tell a
    # QUOTED data structure from a generated wall. Detector docstrings in
    # quality_checks.py carry the full diagnosis and the rejected alternatives.
    #
    #   was think_max_char_run  >= 500   precision 0.35, 49 corpus docs
    #   now think_max_glyph_run >= 1000  precision 1.00,  6 corpus docs
    # 86% of the old hits had run character ' ' -- it was a duplicate of the space
    # gate. Excluding whitespace and ASCII rule glyphs leaves only real walls.
    'think_max_glyph_run': 1000,

    #   was think_max_space_run          >= 100  precision 0.20, 394 corpus docs
    #   now think_max_trailing_space_run >= 100  precision 0.80, 133 corpus docs
    # The worst gate in the set. The INTERIOR component has 0 true positives and 16
    # false positives on the labelled rows -- it has never once been right. The real
    # defect always TERMINATES a line.
    'think_max_trailing_space_run': 100,

    #   was think_max_token_run          >= 200  precision 0.65, 34 corpus docs
    #   now think_max_unfenced_token_run >= 200  precision 1.00, 20 corpus docs
    # All 13 old false positives were quoted data structures, fenced or pasted-table
    # filler.
    'think_max_unfenced_token_run': 200,

    'think_max_ellipsis_run': 200,       # legitimate dot leaders top out at 157

    # DELIMITER-FREE repetition. Every other detector here needs a boundary -- a
    # repeated character, a whitespace-delimited token, a line, a >=25-char
    # sentence. A substring that repeats with NO delimiter matches none of them.
    # 817,794 chars of "DataGridViewTextBoxColumn" x32,630 scored max_char_run 68,
    # max_token_run 1, prose_sentence_repeat 0, wordy_line_repeat 2 -- a 100%
    # degenerate record that passed every gate. See max_shingle_dup_frac.
    # 0.90: the 8 confirmed walls score 0.963-0.997.
    'think_max_shingle_dup_frac': 0.90,

    # max_think_dup is NOT a plain threshold any more -- see THINK_DUP_* below and
    # the note there. It stays out of this dict on purpose.
}

# THE IMPORTANT ONE, reformulated. Near-duplicate reasoning ACROSS turns is the
# only detector that sees the canonical failure: a DeepSeek-V3.2 tool-use record
# repeats "We are stuck. Let's think if there is any other tool or data source we
# haven't considered." once per turn for 174 turns. Every per-turn metric reads ~1
# on it, because within any single turn nothing repeats.
#
# But the raw count >= 10 measured precision 0.41 (160 corpus docs): terse action
# labels repeated while the trajectory ADVANCES look identical to a dead channel if
# you only count. The discriminator is the ACTION on the duplicated turns. Combined
# rule: count >= 8 AND (long block OR dead channel OR stuck action).
#   precision 0.41 -> 0.92, recall 0.90 -> 0.85, corpus 160 -> 102 docs
THINK_DUP_MIN_COUNT = 8
THINK_DUP_MIN_UNIT_CHARS = 200   # B_DUPPARA -- a long paragraph re-emitted verbatim
THINK_DUP_MIN_FRAC = 0.50        # B_FILLER  -- dead channel ("i'll answer." 808/817)
THINK_DUP_MIN_CALL_SHARE = 0.80  # B_DUPSTUCK-- same think AND same call, no progress

# Distinct Unicode scripts in one reasoning field: the gibberish / token-salad
# detector. Threshold 7, raised from 5 on hand labels: at >=5 the gate is right
# once in 27 on the real catalogue (26 legitimate multilingual docs rejected per
# actual defect); at >=7 it is right once in 5 and loses NOTHING, because
# scripts 2-6 is an empty band in the known-bad population (degenerate floor is
# 7, median 11) while legitimate multilingual data sits at <=3.
#
# REASONING ONLY. Applying this to `content` was a measured regression: 0 true
# positives in 1.25M docs and 1 confirmed false positive (a Wiktionary-style
# language table -- a good answer that lists many scripts by design).
SCRIPT_THRESHOLD = 7

# --------------------------------------------------------------------------
# CONTENT thresholds. 2-3x looser, and only four metrics. Not redundant with the
# reasoning gate: 61% of content flags are reasoning-clean, i.e. a final answer
# that degenerated while the reasoning stayed fine. Union cost ~0.026% of corpus,
# with 0 false positives on hand inspection of all 65 flagged records.
#
# DELIBERATELY NOT GATED on content: space_run (0 defects in 1.25M docs;
# legitimate table padding reaches 883), ellipsis_run (1 doc, already caught by
# token_run), n_distinct_scripts (see above).
# --------------------------------------------------------------------------
CONTENT_THRESHOLDS = {
    # Same threshold as reasoning, deliberately: a wall is a wall, and the worst
    # instance found was in `content`, not `think`.
    'content_max_shingle_dup_frac': 0.90,
    'content_max_prose_sentence_repeat': 40,
    'content_max_wordy_line_repeat': 200,
    'content_max_char_run': 1500,
    'content_max_token_run': 200,
}

# Repeated identical tool call is a COMBINED rule, not a single threshold, and
# this is the whole point: many legitimate tasks call one tool dozens of times
# with DIFFERENT arguments (read 40 files, grep 30 patterns), and some
# legitimately repeat an identical call while polling. The absolute count cannot
# separate iterate-and-check from a loop -- measured overlap across the entire
# 5-15 band. What separates them is how much of the trajectory is spent on the
# one identical call. Both conditions must hold.
IDENTICAL_CALL_MIN = 10
IDENTICAL_CALL_FRAC = 0.30

# ...and a CONSECUTIVE-run condition, added 2026-08-08 after measuring the rule on
# the 139 hand labels. Two changes, both large:
#
# 1. `max_consec_identical_call >= 3` as a CONJUNCT to the rule above.
#       precision 0.646 -> 0.859, keeping 55 of 62 true positives (n 96 -> 64)
#    This was the least precise rule in the gate and the largest single source of
#    rejections. What it was mostly catching was the browser-agent OBSERVE step:
#    `browser_get_state` issued 14 times, but each one AFTER a different action on
#    a different page -- interleaved, never consecutive. Requiring even a 3-long
#    consecutive run removes that class outright while keeping the real loops.
#
# 2. A STANDALONE consecutive rule, which needs no fraction at all:
#       max_consec_identical_call >= 8   precision 0.978 (44/45 labels)
#       max_consec_identical_call >= 10  precision 1.000 (33/33 labels)
#    8 consecutive identical calls is a loop under any reading. This is what
#    catches the case the fraction rule structurally cannot: a LONG trajectory with
#    a long stuck run, e.g. a coding-agent record with 30 consecutive
#    identical failing `npm run build`, empty think and empty content on every one,
#    frac 0.280 so the combined rule misses it by 0.02.
#
# TRIED AND REJECTED: switching identical_call_frac's denominator from assistant
# turns to n_tool_calls. Measured 0.646 -> 0.648 precision while LOSING 5 true
# positives -- it is not the fix the batch-call false positives looked like they
# needed; the consecutive conjunct is.
CONSEC_IDENTICAL_CONJUNCT = 3
CONSEC_IDENTICAL_STANDALONE = 8

# CROSS-TURN CONTENT DUPLICATION (2026-08-19). The answer-channel counterpart of
# max_think_dup, and the only detector that sees a WEDGED AGENT: an environment
# that has died (terminal hung, or a harness pinning tool_choice=none and never
# releasing) while the model keeps emitting a byte-identical no-op envelope.
#
# The canonical record, from a tool-use source, has 1,241 assistant
# turns, the same 111-char '{"current_assessment": "The terminal remains stuck.",
# "next_steps": "Stop.", "batch": []}' emitted 819 times, 63 back-to-back. Full
# process_entry ACCEPTED it. Every existing rule is blind for a STRUCTURAL reason,
# not by accident:
#   * the harness puts the command batch inside the content JSON, so tool_calls is
#     empty and both identical-call rules read 0;
#   * the model writes FRESH reasoning on every looping turn (measured: up to 75
#     distinct normalized `think` values across 75 duplicated turns), so
#     max_think_dup reads ~1;
#   * each individual content is a short well-formed JSON object, so every
#     per-turn content metric reads clean.
# Verified across the whole 4,021,234-record shipped corpus: scan_repetition
# returns [] on 650 of 650 flagged records. This rule is ~100% additive.
#
# THRESHOLDS. Two conditions, both load-bearing, both measured on the full corpus
# plus ~9.1M catalogue records:
#   run   >= 8      matches CONSEC_IDENTICAL_STANDALONE, which measured 0.978
#                   precision at 8 on the tool-call side. Smooth, no cliff:
#                   6->732, 7->677, 8->644, 9->604, 10->566, 12->504.
#   ctx_modal_share >= 0.80   the environment must be frozen too. Plateau at
#                   0.6/0.7/0.8 (all 644); population is bimodal with nothing
#                   between the fired band (0.90-1.00) and the spared band
#                   (0.11-0.58). 0.80 matches THINK_DUP_MIN_CALL_SHARE.
# Fire rate 644 / 4,021,234 = 0.016%, entirely in an agentic-coding source (640)
# and a tool-use source (10); the other six subsets fire zero and four of them are structurally immune
# (single-turn). Precision on hand reads: 31/32 by the proposing pass, then 10/10
# and 12/12 on two independent uniform re-draws by adversarial reviewers.
#
# TRIED AND REJECTED, all measured -- do not re-propose these:
#   * A "dead action" conjunct (empty tool_calls, or the modal call signature) as
#     the analogue of dup_call_share. INOPERATIVE: empty-call share is 1.00 on
#     650/650 fires because this harness encodes actions inside `content`, so it
#     cannot separate "no action" from "advancing action". Note this INVERTS the
#     guard in think_dup_stats, which forces share=0 on an empty modal signature
#     precisely so non-agentic records do not score 1.00.
#   * A min-unit-chars floor. Ledger over 10.1M records: it LOST two true defects
#     (a 57-char "the human agent will be with you shortly" hold
#     loop; and a 38-char refusal loop in an SFT source) to SAVE one clean class, and
#     it split one defect family by sentence length -- the same tau-bench stuck
#     transfer fired at 134 chars and passed at 57. It was also upper-bounded at
#     111 by the motivating record itself, leaving an 11-char margin. The ctx
#     conjunct spares the clean class on the right axis and costs no recall.
#   * Keying on the modal COUNT instead of the run: admits 189 records whose
#     duplicated turns are interleaved with real work.
#   * A prefix-normalized near-duplicate variant: +4 records on the tool-use source, of which
#     2 have essentially no exact duplication at all (a shared opening sentence,
#     not a loop).
#
# HONEST SCOPE. Every one of the 650 shipped fires is source_family "terminus-2",
# 219 of them from a single experiment, and in 207 of 226 sampled the
# tool_choice=none directive is not anywhere in the record. This is one generator's
# bug, and the rejection is a message to its owner -- not a corpus-wide tax.
#
# KNOWN RECALL HOLE, deliberately not closed here: a single varied turn splits the
# run. One agentic SFT record ping-pongs "Understood. Standing by..." for
# ~23 turns, but one 41-char variant turn splits it into runs of 7 and 5, so this
# rule is silent. Closing it needs a count-keyed branch WITH the ctx conjunct, and
# that combination has not been measured. Left as a documented gap, not shipped.
CONTENT_DUP_MIN_RUN = 8
CONTENT_DUP_MIN_CTX_SHARE = 0.80

# Metrics that were TRIED AND REJECTED, recorded so they are not re-proposed:
#
#   zlib_ratio / conv_zlib_ratio -- the obvious idea, and it does not work.
#     Legitimate code (0.109) and JSON (0.107) are MORE compressible than the
#     known-bad dsv32 loop record (0.202). Every document the 0.03 setting
#     caught was already caught structurally.
#   think_trailing_ws_spaces -- process_entry strips reasoning whitespace before
#     any gate runs, so this reads 0 by construction. Measured 0/60 still over
#     20 after the strip. It is a pre-strip source diagnostic, not a gate.
#   think_ws_only_line_frac -- cosmetic. Flagged records are uniformly indented
#     by ONE space, so paragraph separators are ' ' instead of ''. Would reject
#     several SFT sources, ARC-AGI and the multilingual sources wholesale.
#   raw max_sentence_repeat / max_line_repeat -- replaced by the prose-only
#     variants; see the notes on the thresholds above.
#   raw max_char_run / max_space_run / max_token_run on REASONING -- replaced by
#     the glyph / trailing-space / unfenced variants above. They are still computed
#     and still gate CONTENT, where they measured 97.9% and 100% on full censuses.
#   adjacent-duplicate exclusion for think_dup -- refuted, precision 0.06.
#   "repeated token must be word-like" for token_run -- 0.67; most real walls are
#     punctuation.
#   ">= 4 alphabetic words on the line" guard for trailing_space_run -- removes 3
#     false positives and destroys 36 real walls.
#   identical_call_frac over n_tool_calls instead of assistant turns -- 0.646 ->
#     0.648 precision while losing 5 true positives. See the note above.
#   non_latin_letter_frac -- punishes legitimately non-English data. Script
#     COUNT separates gibberish (7-12 scripts) from real multilingual text (<=3)
#     without caring whether the data is English.


def scan_repetition(conversation, think_key='think',
                    reasoning_thresholds=None, content_thresholds=None,
                    script_threshold=SCRIPT_THRESHOLD):
    """Return a list of (rule, value, threshold) for every gate that fires.

    Empty list == clean. Pure: never mutates `conversation`.

    `reasoning_thresholds` / `content_thresholds`, when given, REPLACE the
    defaults wholesale -- pass a dict with a metric omitted to disable that gate
    for a source, rather than editing this module.
    """
    rt = REASONING_THRESHOLDS if reasoning_thresholds is None else reasoning_thresholds
    ct = CONTENT_THRESHOLDS if content_thresholds is None else content_thresholds

    hits = []
    # GATE_SKIP drops the two metrics no threshold here reads (zlib_ratio,
    # non_latin_letter_frac). See the note on GATE_SKIP in quality_checks.py.
    m = conversation_metrics(conversation, think_key, skip=GATE_SKIP)
    m.pop('_worst_turn', None)

    for name, thr in list(rt.items()) + list(ct.items()):
        v = m.get(name)
        if v is not None and v >= thr:
            hits.append((name, v, thr))

    n_dup, dup_len, dup_frac, dup_share = think_dup_stats(conversation, think_key)
    if n_dup >= THINK_DUP_MIN_COUNT and (dup_len >= THINK_DUP_MIN_UNIT_CHARS
                                         or dup_frac >= THINK_DUP_MIN_FRAC
                                         or dup_share >= THINK_DUP_MIN_CALL_SHARE):
        hits.append(('max_think_dup', n_dup, THINK_DUP_MIN_COUNT))

    # Cross-turn CONTENT duplication with a frozen environment -- the wedged-agent
    # rule. See the long note on CONTENT_DUP_MIN_RUN for why the conjunct is the
    # ENVIRONMENT and not the action.
    _run, _unit, _ctx_share, _ctx_n = content_dup_run_stats(conversation)
    if _run >= CONTENT_DUP_MIN_RUN and _ctx_share >= CONTENT_DUP_MIN_CTX_SHARE:
        hits.append(('max_content_dup_run', _run, CONTENT_DUP_MIN_RUN))

    _consec = m.get('max_consec_identical_call') or 0
    if ((m.get('max_identical_tool_call') or 0) >= IDENTICAL_CALL_MIN
            and (m.get('identical_call_frac') or 0.0) >= IDENTICAL_CALL_FRAC
            and _consec >= CONSEC_IDENTICAL_CONJUNCT):
        hits.append(('identical_tool_call', m['max_identical_tool_call'],
                     IDENTICAL_CALL_MIN))
    elif _consec >= CONSEC_IDENTICAL_STANDALONE:
        hits.append(('consecutive_identical_tool_call', _consec,
                     CONSEC_IDENTICAL_STANDALONE))

    # REASONING only -- `content_n_distinct_scripts` is deliberately not read.
    # conversation_metrics already reduced this to the worst assistant `think`
    # turn, so there is nothing to recompute here.
    if script_threshold:
        worst_k = m.get('think_n_distinct_scripts') or 0
        if worst_k >= script_threshold:
            hits.append(('think_n_distinct_scripts', worst_k, script_threshold))

    return hits
