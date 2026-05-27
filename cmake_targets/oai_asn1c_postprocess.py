#!/usr/bin/env python3
"""Repair older asn1c output for OAI RRC generated sources."""

from __future__ import annotations

import re
import sys
from pathlib import Path


INLINE_SEQUENCE_RE = re.compile(
    r"\n(?P<indent>\t+)A_SEQUENCE_OF\(struct (?P<name>[A-Za-z0-9_]+) \{\n"
    r"(?P<body>.*?)\n(?P=indent)\} Member\) list;",
    re.DOTALL,
)

VALIDATOR_RE = re.compile(
    r"static int (?P<name>asn_validate_[A-Za-z0-9_]+)"
    r"\(const asn_TYPE_descriptor_t \*td,\n"
    r"\s+const void \*sptr,\n"
    r"\s+asn_app_constraint_failed_f \*ctfailcb,\n"
    r"\s+void\* app_key\) \{\n"
    r"\s+if\(! sptr\) \{ return -1; \}\n"
    r"\s+e_[A-Za-z0-9_]+ value = \*\(e_[A-Za-z0-9_]+\*\)sptr;\n"
    r"\s+switch\(value\) \{\n"
    r".*?\n\s+return 0;\n"
    r"\s+\}\n"
    r"\s+return -1;\n"
    r"\}",
    re.DOTALL,
)

ALIASES = {
    "NR_SystemInformation_IEs__sib_TypeAndInfo__Member": (
        "SystemInformation_IEs__sib_TypeAndInfo__Member"
    ),
}


def dedent_body(body: str, indent: str) -> str:
    lines = []
    for line in body.splitlines():
        if line.startswith(indent):
            line = line[len(indent) :]
        lines.append(line)
    return "\n".join(lines)


def insertion_point(text: str, before: int) -> int:
    typedef_pos = text.rfind("\ntypedef struct ", 0, before)
    if typedef_pos == -1:
        return before

    comment_pos = text.rfind("\n/* ", 0, typedef_pos)
    if comment_pos != -1 and "*/" in text[comment_pos:typedef_pos]:
        return comment_pos
    return typedef_pos


def transform_header(text: str) -> str:
    matches = list(INLINE_SEQUENCE_RE.finditer(text))
    if not matches:
        return text

    edits: list[tuple[int, int, str]] = []
    inserts: list[tuple[int, str]] = []
    seen: set[str] = set()

    for match in matches:
        indent = match.group("indent")
        name = match.group("name")
        body = dedent_body(match.group("body"), indent)

        replacement = f"\n{indent}A_SEQUENCE_OF(struct {name}) list;"
        edits.append((match.start(), match.end(), replacement))

        if name not in seen:
            seen.add(name)
            alias = ALIASES.get(name)
            alias_text = f"typedef struct {name} {name};\n"
            if alias:
                alias_text += f"typedef {name} {alias};\n"

            definition = (
                f"\n/* Forward definitions */\n"
                f"struct {name} {{\n"
                f"{body}\n"
                f"}};\n"
                f"{alias_text}"
            )
            inserts.append((insertion_point(text, match.start()), definition))

    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    for pos, insertion in sorted(inserts, reverse=True):
        text = text[:pos] + insertion + text[pos:]
    return text


def validator_replacement(name: str) -> str:
    return f"""static int {name}(const asn_TYPE_descriptor_t *td,
                       const void *sptr,
                       asn_app_constraint_failed_f *ctfailcb,
                       void* app_key) {{
    if(! sptr) {{ return -1; }}
    const asn_INTEGER_specifics_t *specs = (const asn_INTEGER_specifics_t *)td->specifics;
    const long value = *(const long *)sptr;
    return specs && INTEGER_map_value2enum(specs, value) ? 0 : -1;
}}"""


def transform_validators(text: str) -> str:
    matches = list(VALIDATOR_RE.finditer(text))
    if not matches:
        return text

    pieces: list[str] = []
    counters: dict[str, int] = {}
    cursor = 0

    for index, match in enumerate(matches):
        old_name = match.group("name")
        counters[old_name] = counters.get(old_name, 0) + 1
        new_name = f"{old_name}_oai_{counters[old_name]}"

        pieces.append(text[cursor : match.start()])
        pieces.append(validator_replacement(new_name))

        segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.end() : segment_end]
        segment, replacements = re.subn(
            rf"\b{re.escape(old_name)}\b", new_name, segment, count=1
        )
        if replacements != 1:
            print(f"warning: no descriptor reference found for {old_name}", file=sys.stderr)
        pieces.append(segment)
        cursor = segment_end

    return "".join(pieces)


def transform_source(text: str) -> str:
    text = transform_validators(text)
    return text


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} GENERATED_ASN1_DIR", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    for header in root.glob("*.h"):
        original = header.read_text(encoding="latin-1")
        updated = transform_header(original)
        if updated != original:
            header.write_text(updated, encoding="latin-1")
    for source in root.glob("*.c"):
        original = source.read_text(encoding="latin-1")
        updated = transform_source(original)
        if updated != original:
            source.write_text(updated, encoding="latin-1")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
