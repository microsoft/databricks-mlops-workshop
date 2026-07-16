"""Generate the gapped ``src/`` notebooks from the complete source (``solution/``).

The ``solution/`` tree is the single source of truth: complete, runnable notebooks.
Each solution block is wrapped in markers::

    # TODO-BEGIN: <one-line summary of what to build>
    # HINT: <optional guidance kept in the gapped version>
    # HINT: <as many hint lines as you like>
    <solution code>
    # TODO-END

This script copies ``solution/`` to ``src/``. For each marked block it keeps the
``TODO`` summary and any ``# HINT:`` lines, removes the solution code, and inserts a
placeholder, so participants fill in the gaps guided by the hints. Files with no
markers (helper modules, the instructor-only ``data_ingestion`` notebooks, READMEs)
are copied verbatim.

``src/`` is what the bundle deploys on the default ``personal`` target, so attendees
get the gapped notebooks. The clean ``dev`` / ``staging`` / ``prod`` targets point at
``solution/`` instead (via the ``code_root`` bundle variable), so instructor and CI
runs execute the complete notebooks. ``src/`` is a generated artifact; never hand-edit
it, regenerate it.

Usage::

    python scripts/generate_src.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLUTION_DIR = REPO_ROOT / "solution"
OUT_DIR = REPO_ROOT / "src"

BEGIN = "# TODO-BEGIN"
END = "# TODO-END"
HINT = "# HINT"
PLACEHOLDER = "# <-- Your code here"

# Width of the banner rules that wrap each generated TODO block so the gaps stand out
# clearly from the surrounding provided code.
_BANNER = "-" * 20
TODO_TOP = f"# {_BANNER} TODO {_BANNER}"
TODO_BOTTOM = "# " + "-" * (len(TODO_TOP) - 2)


def strip_solution(text: str) -> str:
    """Replace every TODO-BEGIN..TODO-END block with the hint, any retained HINT
    lines, and a placeholder.

    Inside a ``# TODO-BEGIN`` .. ``# TODO-END`` block:
      * the text after ``TODO-BEGIN:`` becomes the ``# TODO:`` summary line,
      * any ``# HINT:`` comment lines are kept verbatim (guidance for participants),
      * all other (solution) lines are removed and replaced by a single placeholder.
    Each generated block is wrapped in a banner and blank lines so the gap stands out
    from the provided code around it. Files with no markers pass through unchanged."""
    lines = text.splitlines()
    out: list[str] = []
    in_block = False
    indent = ""
    swallow_blank = False
    for line in lines:
        stripped = line.lstrip()
        # Drop a single blank line that immediately follows a block: we add our own
        # trailing blank, so this avoids a double gap before the next line.
        if swallow_blank:
            swallow_blank = False
            if stripped == "":
                continue
        if stripped.startswith(BEGIN):
            in_block = True
            indent = line[: len(line) - len(stripped)]
            hint = stripped[len(BEGIN) :].lstrip(": ").rstrip()
            # Blank line before the banner for a clear separation from preceding code.
            if out and out[-1].strip() != "":
                out.append("")
            out.append(f"{indent}{TODO_TOP}")
            out.append(f"{indent}# TODO: {hint}" if hint else f"{indent}# TODO")
            continue
        if stripped.startswith(END):
            in_block = False
            out.append(f"{indent}{PLACEHOLDER}")
            out.append(f"{indent}{TODO_BOTTOM}")
            out.append("")
            swallow_blank = True
            continue
        if in_block:
            # Retain author-provided hints; drop everything else (the solution).
            if stripped.startswith(HINT):
                out.append(line)
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main() -> None:
    if not SOLUTION_DIR.exists():
        raise SystemExit(f"Source directory not found: {SOLUTION_DIR}")

    # Overwrite files in place and prune stale ones rather than removing the whole
    # tree: rmtree is unreliable on OneDrive-synced folders (files/dirs get locked).
    written: set[Path] = set()
    count = 0
    for src_path in SOLUTION_DIR.rglob("*"):
        if src_path.is_dir():
            continue
        if "__pycache__" in src_path.parts:
            continue
        rel = src_path.relative_to(SOLUTION_DIR)
        dest = OUT_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src_path.suffix == ".py":
            dest.write_text(
                strip_solution(src_path.read_text(encoding="utf-8")),
                encoding="utf-8",
                newline="\n",
            )
        else:
            shutil.copy2(src_path, dest)
        written.add(dest.resolve())
        count += 1

    # Prune generated files that no longer have a matching source (best-effort).
    if OUT_DIR.exists():
        for existing in OUT_DIR.rglob("*"):
            if existing.is_file() and existing.resolve() not in written:
                try:
                    existing.unlink()
                except OSError as err:
                    print(f"[warn] could not remove stale file {existing}: {err}")

    print(f"Generated {count} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
