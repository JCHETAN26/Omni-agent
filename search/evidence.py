"""High-signal identifiers and conservative detection of scoped setting conflicts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal

from backend.core.models import Chunk

_IDENTIFIER = re.compile(
    r"(?<![\w-])(?:0x[0-9a-f]+|[a-z][a-z0-9]*(?:[-_][a-z0-9]+)+"
    r"|[a-z]{2,}\d+)(?![\w-])",
    re.IGNORECASE,
)
_SETTING = re.compile(
    r"(?<![\w.-])([a-z][\w.-]*)\s*`?\s*([:=])\s*`?\s*"
    r"([+-]?\d+(?:\.\d+)?|true|false|enabled|disabled|allow|deny)"
    r"(?:\s*((?:ms|s|m|h|kib|mib|gib|kb|mb|gb|%)(?![\w-]|\.\w)))?"
    r"(?![\w+/-]|\.\d)",
    re.IGNORECASE,
)
_ENTITY = re.compile(r"(?<![\w-])[a-z][a-z0-9]*(?:-[a-z0-9]+)*-\d+(?![\w-])")
_ENV = re.compile(
    r"\b(?:non[- ]prod(?:uction)?|production|prod|staging|development|dev|test|"
    r"legacy|sandbox|canary|uat|qa)\b",
    re.I,
)
_CLAUSE = re.compile(r"[;\n]+|(?<=[.!?])\s+")
_ILLUSTRATIVE = re.compile(r"\b(?:example|formerly|previously|do not|never set)\b", re.I)


def exact_identifiers(text: str) -> list[str]:
    return list(dict.fromkeys(_IDENTIFIER.findall(text.casefold())))


def _environments(text: str) -> set[str]:
    aliases = {"prod": "production", "dev": "development", "non-prod": "non-production"}
    return {
        aliases.get(value, value) for value in _ENV.findall(text.casefold().replace("non ", "non-"))
    }


def matches_requested_environment(question: str, chunk: Chunk) -> bool:
    """Exclude passages that explicitly belong to a different environment."""
    requested = _environments(question) - {"test"}
    if len(requested) != 1:
        return True
    metadata = f"{chunk.document_title}\n{chunk.section_heading}"
    declared = _environments(metadata)
    for line in chunk.text.splitlines():
        if "scope:" in line.casefold() or _SETTING.search(line):
            declared.update(_environments(line))
    return len(declared) != 1 or bool(requested & declared)


def setting_conflict(question: str, chunks: Sequence[Chunk]) -> tuple[str, list[str]] | None:
    """Return cited, scoped conflicting assignments and their chunk IDs.

    Compare explicit scalar assignments with matching keys, scope and units.
    Require positive scope evidence and skip clauses with ambiguous scope.
    """
    query_entities = set(_ENTITY.findall(question))
    question = question.casefold()
    question_words = " ".join(re.findall(r"[a-z0-9]+", question))
    query_envs = _environments(question)
    groups: dict[tuple[str, str, str, str], dict[Decimal | str, tuple[str, str]]] = {}
    for chunk in chunks:
        context_scope = None
        for clause in _CLAUSE.split(chunk.text):
            for match in _SETTING.finditer(clause):
                if _ILLUSTRATIVE.search(clause[: match.start()]):
                    continue
                key, separator, value, unit = (
                    part.casefold() if part else "" for part in match.groups()
                )
                # Avoid interpreting the last word of a prose label as a key.
                if separator == ":" and clause[: match.start()].strip(" \t`*-"):
                    continue
                key_words = " ".join(re.findall(r"[a-z0-9]+", key))
                if f" {key_words} " not in f" {question_words} " and not (
                    key == "max_connections" and "connection limit" in question_words
                ):
                    continue
                if context_scope is None:
                    context = f"{chunk.document_title}\n{chunk.section_heading}\n{chunk.text}"
                    context_scope = (set(_ENTITY.findall(context)), _environments(context))
                context_entities, context_envs = context_scope
                entities = set(_ENTITY.findall(clause)) or context_entities
                envs = _environments(clause) or context_envs
                if len(entities) > 1 or len(envs) > 1 or not (entities or envs):
                    continue
                entity = next(iter(entities), "")
                env = next(iter(envs), "")
                if (query_entities and entity not in query_entities) or (
                    query_envs and env not in query_envs
                ):
                    continue
                normalized = Decimal(value) if value[0] in "+-0123456789" else value
                display = f"{value} {unit}" if unit else value
                group = groups.setdefault((key, entity, env, unit), {})
                group.setdefault(normalized, (display, chunk.chunk_id))

    for (key, entity, env, _), group in groups.items():
        if len(group) < 2:
            continue
        # Two distinct values suffice to report disagreement. Each citation maps
        # directly to the sources frame, even when both values share one chunk.
        refs: list[str] = []
        cited = []
        for value, chunk_id in list(group.values())[:2]:
            if chunk_id not in refs:
                refs.append(chunk_id)
            cited.append(f"`{value}` [{refs.index(chunk_id) + 1}]")
        scope = " in ".join(filter(None, (entity, env)))
        return (
            f"The sources disagree about `{key}` for {scope}. They report "
            f"{' and '.join(cited)}. Confirm the authoritative policy before changing the setting.",
            refs,
        )
    return None
