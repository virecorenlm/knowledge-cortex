"""Deterministic line-based 3-way merge (diff3 semantics, conservative).

    merge3(base, left, right) -> MergeResult(clean, text, regions)

Each side's edits are the non-equal opcodes of difflib.SequenceMatcher
(autojunk off) against BASE, as base line ranges [i1, i2) plus replacement
lines. All edits are sorted by base position and grouped while they overlap
OR touch (an edit ending at line k and another starting at k), so edits to
adjacent lines, or two insertions at the same point, land in one group.
  - a group with edits from one side only is applied
  - a group with edits from both sides is applied only if both sides produce
    byte-identical text for the group's span; otherwise it is a conflict
    region (base/left/right text recorded) and the merge is NOT clean
Treating "touching" as overlapping is deliberately conservative (as diff3
does): anything uncertain is a conflict, never a silent choice.

Markdown guard: a merge that would leave an unbalanced ``` / ~~~ fence when
all three inputs are balanced is reported as a conflict, because a line
merge can otherwise splice text into or out of a code block.

No model is ever involved; identical inputs always give identical output.
"""

import difflib
import re
from dataclasses import dataclass, field

_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")


@dataclass
class MergeResult:
    clean: bool
    text: str | None
    regions: list = field(default_factory=list)
    reason: str | None = None


def _lines(text):
    return text.splitlines(keepends=True)


def _edits(base, other, side):
    matcher = difflib.SequenceMatcher(a=base, b=other, autojunk=False)
    return [(i1, i2, other[j1:j2], side) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def _apply(base, lo, hi, edits):
    """Base[lo:hi] with one side's edits (all inside [lo, hi]) applied."""
    out, pos = [], lo
    for i1, i2, repl, _ in sorted(edits, key=lambda e: (e[0], e[1])):
        out.extend(base[pos:i1])
        out.extend(repl)
        pos = i2
    out.extend(base[pos:hi])
    return out


def _fences_balanced(text):
    return sum(1 for line in text.splitlines() if _FENCE_RE.match(line)) % 2 == 0


def merge3(base, left, right):
    if left == right:
        return MergeResult(True, left)
    if base == left:
        return MergeResult(True, right)
    if base == right:
        return MergeResult(True, left)
    b, l_, r_ = _lines(base), _lines(left), _lines(right)
    edits = sorted(_edits(b, l_, "left") + _edits(b, r_, "right"), key=lambda e: (e[0], e[1], e[3]))
    groups = []
    for edit in edits:
        if groups and edit[0] <= groups[-1]["hi"]:
            groups[-1]["edits"].append(edit)
            groups[-1]["hi"] = max(groups[-1]["hi"], edit[1])
        else:
            groups.append({"lo": edit[0], "hi": edit[1], "edits": [edit]})

    out, pos, regions = [], 0, []
    for g in groups:
        out.extend(b[pos:g["lo"]])
        sides = {e[3] for e in g["edits"]}
        left_text = _apply(b, g["lo"], g["hi"], [e for e in g["edits"] if e[3] == "left"])
        right_text = _apply(b, g["lo"], g["hi"], [e for e in g["edits"] if e[3] == "right"])
        if len(sides) == 1:
            out.extend(left_text if "left" in sides else right_text)
        elif left_text == right_text:
            out.extend(left_text)
        else:
            regions.append({"base_start_line": g["lo"] + 1, "base_end_line": g["hi"],
                            "base": "".join(b[g["lo"]:g["hi"]]), "left": "".join(left_text),
                            "right": "".join(right_text)})
            out.extend(b[g["lo"]:g["hi"]])  # placeholder; text is discarded when not clean
        pos = g["hi"]
    out.extend(b[pos:])
    if regions:
        return MergeResult(False, None, regions, reason=f"{len(regions)} overlapping region(s) changed on both sides")
    merged = "".join(out)
    if all(_fences_balanced(t) for t in (base, left, right)) and not _fences_balanced(merged):
        return MergeResult(False, None, [{"base_start_line": 1, "base_end_line": len(b), "base": base,
                                          "left": left, "right": right}],
                           reason="merge would break Markdown code-fence structure")
    return MergeResult(True, merged)
