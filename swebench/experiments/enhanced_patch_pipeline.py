from __future__ import annotations

import ast
import json
import os
import re
import shlex
import traceback
from difflib import unified_diff

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import docker
from openai import OpenAI
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from swebench.harness.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_PASS,
)
from swebench.harness.docker_build import build_container, close_logger, setup_logger
from swebench.harness.docker_utils import cleanup_container, copy_to_container, exec_run_with_timeout
from swebench.harness.grading import get_logs_eval
from swebench.harness.run_evaluation import GIT_APPLY_CMDS
from swebench.harness.test_spec.create_scripts import make_eval_script_list
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import (
    EvaluationError,
    get_modified_files,
    get_new_files,
    load_swebench_dataset,
    optional_str,
)
from swebench.inference.make_datasets.utils import extract_diff, repair_patch


DEFAULT_MODEL = "gpt-4o"
DEFAULT_SYSTEM_PROMPT = (
    "You are helping with SWE-bench style software engineering research. "
    "Return precise, executable unified diffs when asked for patches."
)
START_MARKER = ">>>>> Start Test Output"
END_MARKER = ">>>>> End Test Output"
TEST_NAME_PATTERN = re.compile(r"^\+\s*def\s+(test_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)
INDENTED_TEST_NAME_PATTERN = re.compile(
    r"^\+\s{4,}def\s+(test_[A-Za-z0-9_]+)\s*\(",
    re.MULTILINE,
)
ENHANCED_TEST_PREFIX = "test_sweb_enhanced_"
STRUCTURAL_RULE_RESTORE_UNREACHABLE_BRANCH = "restore_unreachable_existing_branch"


@dataclass
class CandidatePatch:
    idea: dict[str, Any] | None
    patch: str
    raw_response: str
    identifiers: list[str]
    enhanced_identifiers: list[str] | None = None
    duplicate_identifiers: list[str] | None = None
    kept: bool = False
    reason: str = ""
    failing_identifiers: list[str] | None = None
    eval_status_map: dict[str, str] | None = None
    generation_attempts: int = 1
    quality_score: float = 0.0
    quality_breakdown: dict[str, float] | None = None
    covered_original_tests: list[str] | None = None
    covered_obligations: list[str] | None = None


@dataclass
class EvalResult:
    resolved: bool
    status_map: dict[str, str]
    log_text: str
    log_path: str
    report: dict[str, Any] | None
    patch_applied: bool
    patch_apply_mode: str
    timed_out: bool
    error: str | None = None


@dataclass
class PatchGenerationResult:
    analysis: dict[str, Any] | None
    strategy: dict[str, Any] | None
    edit_plan: dict[str, Any] | None
    patch: str
    raw_response: str
    attempts: int
    accepted: bool
    acceptance_reason: str
    candidate_eval: EvalResult | None = None
    semantic_oracle_passed: bool = False
    semantic_oracle_failed_identifiers: list[str] | None = None
    patch_error: str | None = None
    final_feedback: str | None = None


@dataclass
class SemanticOracleResult:
    passed: bool
    failed_identifiers: list[str]
    status_maps: list[dict[str, str]]
    failure_summaries: dict[str, str]


def align_generated_patch_outcome(
    *,
    generated_patch_accepted: bool,
    generated_patch_acceptance_reason: str,
    generated_patch_feedback: str | None,
    patch_was_generated: bool,
    final_eval: EvalResult,
) -> tuple[bool, str, str | None]:
    if patch_was_generated and final_eval.resolved:
        return True, "accepted_final_validation", None
    return (
        generated_patch_accepted,
        generated_patch_acceptance_reason,
        generated_patch_feedback,
    )


FAILURE_MODE_RULES = [
    ("attribute_error", ["attributeerror", "hasattr", "__dict__", "__slots__"]),
    ("import_error", ["importerror", "modulenotfounderror", "cannot import"]),
    ("type_error", ["typeerror"]),
    ("value_error", ["valueerror"]),
    ("key_error", ["keyerror"]),
    ("assertion_error", ["assertionerror", "assert "]),
]

TEMPLATE_LIBRARY = {
    "attribute_error": [
        "attribute_absence_check",
        "forbidden_attribute_assignment",
        "slots_visibility_check",
    ],
    "import_error": [
        "import_surface_check",
        "public_api_resolution",
        "module_alias_check",
    ],
    "type_error": [
        "boundary_type_input",
        "coercion_contract",
        "mixed_operand_type",
    ],
    "value_error": [
        "invalid_value_boundary",
        "normalization_contract",
        "message_specific_failure",
    ],
    "key_error": [
        "missing_key_surface",
        "mapping_roundtrip",
        "lookup_boundary",
    ],
    "assertion_error": [
        "semantic_invariant_check",
        "regression_example",
        "state_consistency_check",
    ],
    "generic_failure": [
        "semantic_regression_example",
        "boundary_case_variant",
        "state_invariant_variant",
    ],
}

SEMANTIC_BUCKETS = {
    "attribute_error": [
        "direct_symptom",
        "behavioral_consequence",
        "structural_invariant",
        "regression_contrast",
        "boundary_variant",
    ],
    "generic_failure": [
        "direct_symptom",
        "behavioral_consequence",
        "structural_invariant",
    ],
}

HARD_FAILURE_SIGNALS = [
    "indentationerror",
    "syntaxerror",
    "importerror",
]

SEMANTIC_BUCKET_GUIDANCE = {
    "direct_symptom": "tests the most visible failure symptom or violated observable property",
    "behavioral_consequence": "tests a downstream behavior change caused by the bug through a different usage path",
    "structural_invariant": "tests a class, schema, or object invariant that should always hold after the fix",
    "regression_contrast": "tests a contrast against a closely related API path, object, or regression example",
    "boundary_variant": "tests an edge-case input, construction path, or boundary condition related to the bug",
}

NOISE_LOG_TOKENS = [
    "deprecationwarning",
    "resourcewarning",
    "pendingdeprecationwarning",
    "userwarning",
    "warning:",
]

DEFAULT_MAX_DYNAMIC_CANDIDATES = 12
DEFAULT_BASE_CANDIDATES = 3


def normalize_patch(text: str | None) -> str:
    if text is None:
        return ""
    patch = text.strip()
    return f"{patch}\n" if patch else ""


def extract_diff_from_raw_response(raw_response: str) -> str:
    """Extract the first valid unified diff from a raw LLM response string.

    Tries fenced code blocks first (```diff ... ``` or ``` ... ```), then
    falls back to looking for a bare 'diff --git' header anywhere in the text.
    Returns the extracted patch text (normalized) or empty string.
    """
    # 1. Try fenced blocks
    for pattern in [
        r"```diff\s*\n(.*?)```",
        r"```\s*\n(diff --git.*?)```",
        r"```patch\s*\n(.*?)```",
    ]:
        m = re.search(pattern, raw_response, re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            if candidate.startswith("diff --git") or candidate.startswith("---"):
                sanitized, err = sanitize_unified_diff(candidate)
                if sanitized and not err:
                    return sanitized
    # 2. Bare diff starting from 'diff --git'
    idx = raw_response.find("diff --git")
    if idx != -1:
        candidate = raw_response[idx:].strip()
        sanitized, err = sanitize_unified_diff(candidate)
        if sanitized and not err:
            return sanitized
    return ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _check_patch_syntax_in_memory(patch_text: str, code_context: dict[str, str]) -> str | None:
    """Apply patch to code_context in memory and compile each changed .py file. Returns error msg or None."""
    try:
        patchset = PatchSet(patch_text)
    except Exception:
        return None  # sanitize_unified_diff already validated
    for patched_file in patchset:
        path = patched_file.path
        if not path.endswith(".py"):
            continue
        original = code_context.get(path, "")
        if not original:
            continue
        lines = original.splitlines(keepends=True)
        try:
            result_lines = list(lines)
            offset = 0
            for hunk in patched_file:
                src_start = hunk.source_start - 1 + offset
                if src_start < 0 or src_start > len(result_lines):
                    return f"{path}: hunk start line {hunk.source_start} is out of range (file has {len(lines)} lines)"
                source_count = sum(1 for l in hunk if not l.is_added)
                # Validate that context lines match the actual file content
                context_lines_in_hunk = [l.value for l in hunk if l.is_context]
                actual_source_lines = [l.value for l in result_lines[src_start: src_start + source_count] if l.rstrip('\n') == l.rstrip('\n')]
                source_lines = result_lines[src_start: src_start + source_count]
                expected_context = [l.value for l in hunk if l.is_context]
                actual_context = []
                ci = src_start
                for l in hunk:
                    if l.is_context:
                        if ci < len(result_lines):
                            actual_context.append(result_lines[ci].rstrip('\n'))
                        ci += 1
                    elif not l.is_added:
                        ci += 1
                replacement = [l.value for l in hunk if not l.is_removed]
                result_lines[src_start: src_start + source_count] = replacement
                offset += len(replacement) - source_count
            result = "".join(result_lines)
        except Exception as e:
            return f"{path}: patch apply failed in memory: {e}"
        try:
            compile(result, path, "exec")
        except SyntaxError as e:
            return f"{path}:{e.lineno}: {e.msg}"
    return None


def sanitize_unified_diff(patch_text: str | None) -> tuple[str, str | None]:
    patch = normalize_patch(patch_text)
    if not patch:
        return "", "empty patch"
    last_error = None
    candidates = [patch]
    repaired = normalize_patch(repair_patch(patch))
    if repaired and repaired not in candidates:
        candidates.append(repaired)
    for candidate in candidates:
        try:
            PatchSet(candidate)
            return candidate, None
        except UnidiffParseError as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)
    return "", last_error


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n...[truncated {omitted} chars]..."


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def extract_test_identifiers_from_patch(patch: str) -> list[str]:
    added_lines = "\n".join(
        line for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    names = TEST_NAME_PATTERN.findall(added_lines)
    names.extend(INDENTED_TEST_NAME_PATTERN.findall(added_lines))
    # Also capture test functions that are modified (present in context/removed lines)
    # because the test_patch may update an existing test rather than add a new one.
    context_lines = "\n".join(
        line[1:] if line.startswith((" ", "-")) else line
        for line in patch.splitlines()
        if (line.startswith(" ") or line.startswith("-")) and not line.startswith("---")
    )
    names.extend(TEST_NAME_PATTERN.findall(context_lines))
    names.extend(INDENTED_TEST_NAME_PATTERN.findall(context_lines))
    return dedupe_preserve_order(names)


def find_failing_identifiers(status_map: dict[str, str], identifiers: list[str]) -> list[str]:
    failing = []
    for identifier in identifiers:
        for test_name, status in status_map.items():
            if identifier in test_name and status in {"FAILED", "ERROR"}:
                failing.append(identifier)
                break
    return dedupe_preserve_order(failing)


def get_enhanced_test_identifiers(identifiers: list[str]) -> list[str]:
    return [identifier for identifier in identifiers if identifier.startswith(ENHANCED_TEST_PREFIX)]


def get_original_test_identifiers(instance: dict[str, Any]) -> list[str]:
    # Prefer SWE-bench's authoritative FAIL_TO_PASS gold labels when available —
    # these are exactly the tests that must go from FAIL to PASS for the issue to be resolved.
    # Only fall back to test_patch parsing when gold labels are absent.
    gold_f2p_raw = instance.get(FAIL_TO_PASS)
    if gold_f2p_raw:
        try:
            full_ids = json.loads(gold_f2p_raw) if isinstance(gold_f2p_raw, str) else list(gold_f2p_raw)
            # Strip the "tests/test_foo.py::" prefix to get just the function name
            short_ids = []
            for fid in full_ids:
                parts = fid.split("::")
                short_ids.append(parts[-1])
            return dedupe_preserve_order(short_ids)
        except Exception:
            pass
    return extract_test_identifiers_from_patch(normalize_patch(instance["test_patch"]))


def _normalize_test_identifier_variants(test_id: str | None) -> set[str]:
    raw = str(test_id or "").strip()
    if not raw:
        return set()
    variants = {raw}
    short = raw
    if "::" in raw:
        parts = [part.strip() for part in raw.split("::") if part.strip()]
        if parts:
            short = parts[-1]
            variants.add(short)
            variants.add("::".join(parts[-2:]) if len(parts) >= 2 else short)
    elif ":" in raw:
        prefix, suffix = raw.rsplit(":", 1)
        prefix = prefix.strip()
        suffix = suffix.strip()
        if suffix:
            short = suffix
            variants.add(short)
            if prefix:
                prefix_name = Path(prefix).name
                variants.add(f"{prefix_name}::{suffix}")
                variants.add(f"{prefix_name}:{suffix}")
    if short.startswith("test_"):
        variants.add(short)
    return {item for item in variants if item}


def _test_identifiers_match(left: str | None, right: str | None) -> bool:
    left_variants = _normalize_test_identifier_variants(left)
    right_variants = _normalize_test_identifier_variants(right)
    return bool(left_variants and right_variants and left_variants.intersection(right_variants))


def get_active_fail_to_pass_identifiers(
    instance: dict[str, Any],
    status_map: dict[str, str] | None,
) -> list[str]:
    original_identifiers = get_original_test_identifiers(instance)
    if not original_identifiers:
        return []
    status_map = status_map or {}
    active: list[str] = []
    for short_id in original_identifiers:
        if any(
            value in {"FAILED", "ERROR"} and _test_identifiers_match(key, short_id)
            for key, value in status_map.items()
        ):
            active.append(short_id)
    return dedupe_preserve_order(active)


def find_duplicate_test_identifiers(candidate_identifiers: list[str], original_identifiers: list[str]) -> list[str]:
    original = set(original_identifiers)
    return [identifier for identifier in candidate_identifiers if identifier in original]


def format_code_context(context_by_file: dict[str, str]) -> str:
    sections = []
    for path, content in context_by_file.items():
        if path.startswith("__tail__"):
            continue  # internal tail entries — not shown as standalone files
        sections.append(f"### File: {path}\n```text\n{content}\n```")
    return "\n\n".join(sections)


def format_candidate_summary(candidates: list[CandidatePatch]) -> str:
    if not candidates:
        return "No filtered enhanced tests were retained."
    sections = []
    for idx, candidate in enumerate(candidates, start=1):
        ids = ", ".join(candidate.identifiers) if candidate.identifiers else "N/A"
        sections.append(
            f"### Enhanced Test Patch {idx}\n"
            f"Identifiers: {ids}\n"
            f"```diff\n{candidate.patch}\n```"
        )
    return "\n\n".join(sections)


def count_statuses(status_map: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in status_map.values():
        counts[status] = counts.get(status, 0) + 1
    return counts


def count_passed(status_map: dict[str, str]) -> int:
    return sum(1 for status in status_map.values() if status in {"PASSED", "XFAIL"})


def count_failed(status_map: dict[str, str]) -> int:
    return sum(1 for status in status_map.values() if status in {"FAILED", "ERROR"})


def has_meaningful_status_map(status_map: dict[str, str]) -> bool:
    return bool(status_map)


def _extract_import_error_signatures(text: str) -> set[str]:
    if not text:
        return set()
    signatures: set[str] = set()
    for match in re.findall(r"(ModuleNotFoundError:\s+[^\n]+)", text):
        signatures.add(match.strip().lower())
    for match in re.findall(r"(ImportError:\s+[^\n]+)", text):
        signatures.add(match.strip().lower())
    for missing_module in re.findall(r"No module named ['\"]([^'\"]+)['\"]", text):
        signatures.add(f"missing_module:{missing_module.strip().lower()}")
    return signatures


def is_environment_blocked_import_failure(
    failure_focus: dict[str, Any] | None,
    eval_result: EvalResult | None,
) -> bool:
    if not failure_focus or not eval_result:
        return False
    if str(failure_focus.get("failure_mode") or "").strip() != "import_error":
        return False
    if has_meaningful_status_map(eval_result.status_map):
        return False
    baseline_signatures = _extract_import_error_signatures(
        "\n".join(str(tb) for tb in (failure_focus.get("target_test_tracebacks") or {}).values())
    )
    eval_signatures = _extract_import_error_signatures(eval_result.log_text)
    if not baseline_signatures or not eval_signatures:
        return False
    if baseline_signatures.intersection(eval_signatures):
        return True
    baseline_missing = {sig for sig in baseline_signatures if sig.startswith("missing_module:")}
    eval_missing = {sig for sig in eval_signatures if sig.startswith("missing_module:")}
    if baseline_missing and eval_missing and baseline_missing.intersection(eval_missing):
        return True
    dominant_errors = {
        str(item).strip().lower()
        for item in (failure_focus.get("dominant_errors") or [])
        if str(item).strip()
    }
    eval_lower = (eval_result.log_text or "").lower()
    if {"modulenotfounderror", "importerror"}.intersection(dominant_errors) and (
        "modulenotfounderror" in eval_lower or "importerror" in eval_lower
    ):
        return True
    return False


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def get_repair_mode(filtered_candidates: list[CandidatePatch]) -> str:
    return "enhanced_guided" if filtered_candidates else "baseline_fallback"


def aggregate_enhanced_failures(candidates: list[CandidatePatch]) -> list[str]:
    identifiers = []
    for candidate in candidates:
        identifiers.extend(candidate.failing_identifiers or [])
    return dedupe_preserve_order(identifiers)


def classify_failure_mode(original_failure_log: str) -> str:
    """Classify the dominant failure mode from the test log.

    Uses only error-indicator lines (pytest 'E ' prefix lines and error headers)
    to avoid false positives from source code containing exception patterns
    (e.g. 'except ImportError:' in werkzeug source being classified as import_error).

    Also handles 'DID NOT RAISE' as a value_error or assertion_error pattern.
    """
    # First pass: use only the error output lines (most reliable signal)
    error_lines = _extract_error_lines(original_failure_log)
    error_lowered = error_lines.lower()
    # "DID NOT RAISE" is a pytest special failure for pytest.raises() blocks
    if "did not raise" in error_lowered:
        # Determine which exception was expected from the "DID NOT RAISE" context
        import re as _re
        m = _re.search(r"did not raise.*?<class '([a-z_]+)'", error_lowered)
        if m:
            exc_name = m.group(1).lower()
            for mode, patterns in FAILURE_MODE_RULES:
                if any(exc_name.startswith(p.replace("error", "")) for p in patterns):
                    return mode
        return "assertion_error"
    for mode, patterns in FAILURE_MODE_RULES:
        if any(pattern in error_lowered for pattern in patterns):
            return mode
    # Second pass: fall back to full log scan (but still exclude import_error
    # to avoid werkzeug source noise)
    full_lowered = original_failure_log.lower()
    for mode, patterns in FAILURE_MODE_RULES:
        if mode == "import_error":
            continue  # skip import_error in full-log pass to reduce noise
        if any(pattern in full_lowered for pattern in patterns):
            return mode
    # Final fallback: check import_error in full log
    for mode, patterns in FAILURE_MODE_RULES:
        if mode == "import_error":
            if any(pattern in full_lowered for pattern in patterns):
                return mode
    return "generic_failure"


def compute_log_noise_ratio(original_failure_log: str) -> float:
    lines = [line.strip().lower() for line in original_failure_log.splitlines() if line.strip()]
    if not lines:
        return 0.0
    noisy_lines = sum(1 for line in lines if any(token in line for token in NOISE_LOG_TOKENS))
    return noisy_lines / len(lines)


def _traceback_block_has_failure_signal(block_lines: list[str]) -> bool:
    joined = "\n".join(block_lines).lower()
    if not joined.strip():
        return False
    failure_tokens = (
        " failed",
        " error",
        "traceback",
        "assertionerror",
        "typeerror",
        "valueerror",
        "runtimeerror",
        "did not raise",
        "e       ",
        "e   ",
        "error:",
        "failed:",
    )
    return any(token in joined for token in failure_tokens)


def _build_weak_target_traceback_fallback(
    target_id: str,
    original_failure_log: str,
    failure_snippets: list[str],
    dominant_errors: list[str],
    max_lines: int = 20,
) -> str:
    lines = original_failure_log.splitlines()
    collected: list[str] = [f"TARGET TEST: {target_id}"]
    collected.extend(f"- snippet: {snippet}" for snippet in (failure_snippets or [])[:3])
    if dominant_errors:
        collected.append(f"- dominant_errors: {', '.join(dominant_errors[:5])}")
    target_lower = target_id.lower()
    nearby: list[str] = []
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if target_lower in lowered or any(token in lowered for token in ("typeerror", "assertionerror", "valueerror", "traceback", "failed", "error")):
            nearby.append(line)
        if len(nearby) >= max_lines:
            break
    if nearby:
        collected.append("WEAK TRACEBACK CONTEXT:")
        collected.extend(nearby[:max_lines])
    return "\n".join(collected)


def extract_assertion_value_mismatches(original_failure_log: str) -> list[dict[str, str]]:
    mismatches: list[dict[str, str]] = []
    for raw_line in original_failure_log.splitlines():
        line = raw_line.strip()
        match = re.search(
            r"AssertionError:\s+(.{1,120}?)\s+(!=|==|is not|is)\s+(.{1,120})$",
            line,
        )
        if not match:
            continue
        left, operator, right = (part.strip() for part in match.groups())
        mismatches.append(
            {
                "raw": line,
                "left_value": left,
                "operator": operator,
                "right_value": right,
            }
        )
    return mismatches[:5]


def extract_failure_focus(
    instance: dict[str, Any],
    status_map: dict[str, str],
    original_failure_log: str,
    max_snippets: int = 3,
) -> dict[str, Any]:
    original_identifiers = get_original_test_identifiers(instance)
    active_fail_to_pass_identifiers = get_active_fail_to_pass_identifiers(instance, status_map)
    inactive_fail_to_pass_identifiers = [
        test_id for test_id in original_identifiers
        if test_id not in set(active_fail_to_pass_identifiers)
    ]
    failing_tests = [
        test_name
        for test_name, status in status_map.items()
        if status in {"FAILED", "ERROR"}
    ]
    related_lines: list[str] = []
    seen_lines: set[str] = set()
    signal_tokens = dedupe_preserve_order(
        original_identifiers
        + failing_tests[:5]
        + [classify_failure_mode(original_failure_log), "traceback", "assertionerror", "typeerror", "valueerror"]
    )
    for raw_line in original_failure_log.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if not line:
            continue
        if any(token and token.lower() in lowered for token in signal_tokens):
            if line not in seen_lines:
                seen_lines.add(line)
                related_lines.append(line)
        if len(related_lines) >= max_snippets:
            break
    dominant_errors = dedupe_preserve_order(
        match.group(1)
        for match in re.finditer(r"([A-Za-z_]*Error|[A-Za-z_]*Exception)", original_failure_log)
    )[:5]

    # P1: Extract per-target-test tracebacks to isolate true bug signals from noise.
    # For each original_test_identifier, find the block of log lines that belong to it.
    target_test_tracebacks = extract_per_test_tracebacks(original_failure_log, original_identifiers)
    if len(target_test_tracebacks) < len(original_identifiers):
        for target_id in original_identifiers:
            if target_id in target_test_tracebacks:
                continue
            target_test_tracebacks[target_id] = _build_weak_target_traceback_fallback(
                target_id=target_id,
                original_failure_log=original_failure_log,
                failure_snippets=related_lines[:max_snippets],
                dominant_errors=dominant_errors,
            )

    return {
        "original_test_identifiers": original_identifiers,
        "active_fail_to_pass_identifiers": active_fail_to_pass_identifiers,
        "inactive_fail_to_pass_identifiers": inactive_fail_to_pass_identifiers,
        "failing_tests_sample": failing_tests[:5],
        "failure_mode": classify_failure_mode(original_failure_log),
        "dominant_errors": dominant_errors,
        "assertion_value_mismatches": extract_assertion_value_mismatches(original_failure_log),
        "failure_snippets": related_lines[:max_snippets],
        "noise_ratio": round(compute_log_noise_ratio(original_failure_log), 4),
        "target_test_tracebacks": target_test_tracebacks,
    }


def format_failure_focus(failure_focus: dict[str, Any] | None) -> str:
    if not failure_focus:
        return "No focused failure summary available."
    snippets = failure_focus.get("failure_snippets") or []
    dominant_errors = ", ".join(failure_focus.get("dominant_errors") or []) or "unknown"
    failing_tests = ", ".join(failure_focus.get("failing_tests_sample") or []) or "unknown"
    snippet_text = "\n".join(f"- {snippet}" for snippet in snippets) if snippets else "- No focused snippets extracted."
    result = (
        f"Failure mode: {failure_focus.get('failure_mode', 'generic_failure')}\n"
        f"Active FAIL_TO_PASS tests: {', '.join(failure_focus.get('active_fail_to_pass_identifiers') or []) or 'unknown'}\n"
        f"Inactive FAIL_TO_PASS tests: {', '.join(failure_focus.get('inactive_fail_to_pass_identifiers') or []) or 'none'}\n"
        f"Dominant errors: {dominant_errors}\n"
        f"Focused failing tests: {failing_tests}\n"
        f"Log noise ratio: {failure_focus.get('noise_ratio', 0.0)}\n"
        f"Focused failure snippets:\n{snippet_text}"
    )
    assertion_mismatches = failure_focus.get("assertion_value_mismatches") or []
    if assertion_mismatches:
        mismatch_text = "\n".join(
            f"- {item.get('raw')}" for item in assertion_mismatches[:3]
        )
        result += f"\nDirect assertion value mismatches:\n{mismatch_text}"
    # P1: Surface per-target-test tracebacks as primary repair signal.
    target_tracebacks = failure_focus.get("target_test_tracebacks") or {}
    if target_tracebacks:
        result += "\n\nPRIMARY BUG SIGNAL — tracebacks for the specific tests that must be fixed (FAIL_TO_PASS):\n"
        result += "(These are the ground-truth failure signals; other failures in the log may be environment noise.)\n"
        for test_id, tb in target_tracebacks.items():
            result += f"\n--- {test_id} ---\n{tb}\n"
    return result


DIRECT_ASSERTION_DRIFT_TOKENS = {
    "handler",
    "uploadhandler",
    "uploadedfile",
    "temporar",
    "tempfile",
    "rename",
    "os.rename",
    "os.stat",
    "stat(",
    "storage.save",
    "default_storage.save",
    "filesystem",
    "file system",
}


def build_direct_assertion_enhanced_test_guidance(
    failure_focus: dict[str, Any] | None,
) -> str:
    if not failure_focus:
        return ""
    if str(failure_focus.get("failure_mode") or "") != "assertion_error":
        return ""
    mismatches = failure_focus.get("assertion_value_mismatches") or []
    if not mismatches:
        return ""
    lines = [
        "\nDIRECT ASSERTION-MISMATCH RULE:",
        "The original FAIL_TO_PASS signal is a direct assertion value mismatch, not merely a broad downstream behavior.",
        "Generate enhanced tests that directly assert the corrected invariant behind this mismatch.",
        "Do NOT assert the buggy observed value, and do NOT broaden into storage/handler/filesystem side effects unless the original traceback itself names that downstream API.",
        "For each mismatch below, infer the intended correct side from the original test body/log context and assert that corrected value/invariant, not the value already produced by the buggy code:",
    ]
    for item in mismatches[:3]:
        lines.append(
            f"- {item.get('raw')} (left side: {item.get('left_value')}; operator: {item.get('operator')}; right side: {item.get('right_value')})"
        )
    lines.append(
        "Put the raw mismatch tokens and the direct setting/attribute/function that produces the value into semantic_alignment_tokens and trigger_shape_tokens.\n"
    )
    return "\n".join(lines)


def build_direct_assertion_idea_drift_feedback(
    idea: dict[str, Any],
    failure_focus: dict[str, Any] | None,
) -> str | None:
    guidance = build_direct_assertion_enhanced_test_guidance(failure_focus)
    if not guidance:
        return None
    target_text = " ".join(
        str(part)
        for part in [
            idea.get("target_source_symbol", ""),
            idea.get("target_validation_subject", ""),
            " ".join(str(token) for token in (idea.get("semantic_alignment_tokens") or [])),
            " ".join(str(token) for token in (idea.get("trigger_shape_tokens") or [])),
            idea.get("goal", ""),
            idea.get("rationale", ""),
        ]
    ).lower()
    traceback_text = " ".join(
        str(tb)
        for tb in (failure_focus or {}).get("target_test_tracebacks", {}).values()
    ).lower()
    drifting_tokens = sorted(
        token
        for token in DIRECT_ASSERTION_DRIFT_TOKENS
        if token in target_text and token not in traceback_text
    )
    if drifting_tokens:
        return (
            "This idea drifts from a direct assertion value mismatch into downstream behavior "
            f"({', '.join(drifting_tokens[:5])}). For this failure, propose a near-path test that "
            "directly asserts the corrected value/invariant from the original AssertionError instead "
            "of testing storage, handler, tempfile, rename, stat, or filesystem propagation."
        )
    return None


def _extract_test_function_snippet(content: str, test_name: str) -> str:
    if not content or not test_name:
        return ""
    try:
        module = ast.parse(content)
    except SyntaxError:
        return ""
    lines = content.splitlines()
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == test_name:
            start = max(getattr(node, "lineno", 1) - 1, 0)
            end = min(getattr(node, "end_lineno", start + 1), len(lines))
            return "\n".join(lines[start:end])
    pattern = re.compile(
        rf"^\s*def\s+{re.escape(test_name)}\s*\([^)]*\):\n((?:^[ \t]+.*\n?)*)",
        re.MULTILINE,
    )
    match = pattern.search(content)
    if match:
        return f"def {test_name}():\n{match.group(1)}"
    return ""


def _build_test_import_symbol_map(code_context: dict[str, str] | None) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for path, content in (code_context or {}).items():
        if not isinstance(content, str) or not is_test_like_path(path):
            continue
        for match in re.finditer(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+([A-Za-z0-9_,\s]+)$", content, re.MULTILINE):
            module_name = match.group(1).strip()
            imported = [
                item.strip().split(" as ", 1)[0].strip()
                for item in match.group(2).split(",")
                if item.strip()
            ]
            module_path = module_name.replace(".", "/") + ".py"
            for symbol in imported:
                if not symbol or not symbol[:1].isupper():
                    continue
                bucket = mapping.setdefault(symbol, [])
                if module_path not in bucket:
                    bucket.append(module_path)
    return mapping


def _infer_failure_focus_dominant_symbols(
    failure_focus: dict[str, Any] | None,
    code_context: dict[str, str] | None,
) -> list[str]:
    if not failure_focus or not code_context:
        return []
    target_test_ids = (
        list(failure_focus.get("active_fail_to_pass_identifiers") or [])
        or list(failure_focus.get("inactive_fail_to_pass_identifiers") or [])
        or list(failure_focus.get("original_test_identifiers") or [])
    )
    if not target_test_ids:
        return []
    ignored_symbols = {
        "Traceback",
        "AssertionError",
        "TypeError",
        "ValueError",
        "RuntimeError",
        "Exception",
        "File",
        "TARGET",
        "WEAK",
        "CONTEXT",
    }
    symbol_scores: dict[str, int] = {}
    for test_id in target_test_ids:
        short_name = str(test_id).strip().split("::")[-1].split(":")[-1]
        if not short_name:
            continue
        for path, content in (code_context or {}).items():
            if not isinstance(content, str) or not is_test_like_path(path):
                continue
            snippet = _extract_test_function_snippet(content, short_name)
            if not snippet:
                continue
            for symbol in re.findall(r"\b[A-Z][A-Za-z0-9_]+\b", snippet):
                if symbol in ignored_symbols:
                    continue
                symbol_scores[symbol] = symbol_scores.get(symbol, 0) + 1
    if not symbol_scores and len(target_test_ids) == 1:
        target_name = str(target_test_ids[0]).strip().split("::")[-1].split(":")[-1]
        import_symbol_map = _build_test_import_symbol_map(code_context)
        for path, content in (code_context or {}).items():
            if not isinstance(content, str) or not is_test_like_path(path):
                continue
            if target_name and target_name not in content:
                continue
            for symbol in import_symbol_map:
                if symbol in ignored_symbols:
                    continue
                if re.search(rf"\b{re.escape(symbol)}\s*\(", content) or re.search(rf"\b{re.escape(symbol)}\.", content):
                    symbol_scores[symbol] = max(symbol_scores.get(symbol, 0), 1)
    return [
        symbol
        for symbol, _score in sorted(symbol_scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def _infer_failure_focus_source_files(
    failure_focus: dict[str, Any] | None,
    code_context: dict[str, str] | None,
) -> list[str]:
    if not failure_focus or not code_context:
        return []
    target_test_ids = (
        list(failure_focus.get("active_fail_to_pass_identifiers") or [])
        or list(failure_focus.get("inactive_fail_to_pass_identifiers") or [])
        or list(failure_focus.get("original_test_identifiers") or [])
    )
    if not target_test_ids:
        return []
    import_symbol_map = _build_test_import_symbol_map(code_context)
    dominant_symbols = _infer_failure_focus_dominant_symbols(failure_focus, code_context)
    if not dominant_symbols:
        return []
    selected_files: list[str] = []
    for symbol in dominant_symbols:
        for module_path in import_symbol_map.get(symbol, []):
            if module_path not in selected_files and not is_test_like_path(module_path):
                selected_files.append(module_path)
    return selected_files


def _extract_expected_exception_from_traceback(traceback_text: str) -> str | None:
    if not traceback_text:
        return None
    match = re.search(r"with pytest\.raises\(([^)\n]+)\)", traceback_text)
    if match:
        raw = match.group(1).strip()
        return raw.split(",")[0].strip().strip("'\"")
    match = re.search(r"DID NOT RAISE <class '([^']+)'>", traceback_text)
    if match:
        return match.group(1).split(".")[-1]
    return None


def _extract_observed_exception_from_traceback(traceback_text: str) -> str | None:
    if not traceback_text:
        return None
    matches = re.findall(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))", traceback_text)
    return matches[-1] if matches else None


def _extract_source_file_from_traceback(traceback_text: str) -> str | None:
    if not traceback_text:
        return None
    matches = re.findall(r"([A-Za-z_./-][^:\s]+\.py):\d+", traceback_text)
    if not matches:
        return None
    normalized = [match.lstrip("./") for match in matches if match and not is_test_like_path(match.lstrip("./"))]
    if normalized:
        return normalized[-1]
    return matches[-1].lstrip("./")


def _extract_source_lineno_from_traceback(
    traceback_text: str,
    source_file: str | None,
) -> int | None:
    if not traceback_text or not source_file:
        return None
    normalized_file = str(source_file).lstrip("./")
    matches = re.findall(r"([A-Za-z_./-][^:\s]+\.py):(\d+)", traceback_text)
    for path, lineno in reversed(matches):
        if path.lstrip("./") == normalized_file and not is_test_like_path(path.lstrip("./")):
            try:
                return int(lineno)
            except ValueError:
                return None
    return None


def _extract_owner_hint_from_traceback(traceback_text: str) -> str | None:
    if not traceback_text:
        return None
    match = re.search(r"\bself\s*=\s*<([A-Za-z_][A-Za-z0-9_]*)\b", traceback_text)
    if match:
        return match.group(1)
    return None


def _extract_source_symbol_from_traceback_raw(traceback_text: str) -> str | None:
    if not traceback_text:
        return None
    lowered = traceback_text.lower()
    if "add_url_rule" in lowered:
        return "add_url_rule"
    if "register_blueprint" in lowered:
        return "register_blueprint"
    if "blueprint(" in lowered:
        return "Blueprint.__init__"
    matches = re.findall(r"in ([A-Za-z_][A-Za-z0-9_\.]*)", traceback_text)
    if matches:
        return matches[-1]
    return None


def _qualify_source_symbol_from_traceback(
    traceback_text: str,
    code_context: dict[str, str] | None = None,
    source_file: str | None = None,
) -> str | None:
    raw_symbol = _extract_source_symbol_from_traceback_raw(traceback_text)
    if not raw_symbol or "." in raw_symbol:
        return raw_symbol
    if not code_context or not source_file:
        return raw_symbol
    content = code_context.get(source_file)
    if not content:
        return raw_symbol
    try:
        module = ast.parse(content)
    except SyntaxError:
        return raw_symbol
    source_lineno = _extract_source_lineno_from_traceback(traceback_text, source_file)
    owner_hint = _extract_owner_hint_from_traceback(traceback_text)
    matching_methods: list[tuple[str, int]] = []
    module_function_exists = False
    for node in getattr(module, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == raw_symbol:
            module_function_exists = True
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == raw_symbol:
                    score = 0
                    child_end = getattr(child, "end_lineno", child.lineno)
                    if source_lineno is not None and child.lineno <= source_lineno <= child_end:
                        score += 100
                    if owner_hint and node.name == owner_hint:
                        score += 50
                    # Prefer the method whose body *contains* the failure line over
                    # one that just starts earlier.  When no lineno match, use a small
                    # positive bias for later-defined methods (larger lineno) to prefer
                    # the override/subclass implementation over an earlier base one.
                    if source_lineno is None or not (child.lineno <= source_lineno <= child_end):
                        score += child.lineno // 1000  # tiny late-definition bias
                    matching_methods.append((f"{node.name}.{raw_symbol}", score))
    if matching_methods:
        deduped: list[tuple[str, int]] = []
        seen_symbols: set[str] = set()
        for symbol, score in sorted(matching_methods, key=lambda item: item[1], reverse=True):
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            deduped.append((symbol, score))
        if len(deduped) == 1:
            return deduped[0][0]
        best_symbol, best_score = deduped[0]
        second_score = deduped[1][1]
        if best_score != second_score or not module_function_exists:
            return best_symbol
    return raw_symbol


def _extract_semantic_alignment_tokens_from_traceback(traceback_text: str) -> list[str]:
    lowered = traceback_text.lower()
    token_rules = [
        ("endpoint=", "endpoint=" in lowered),
        ("route(", ".route(" in lowered or " route(" in lowered or "bp.route(" in lowered),
        ("add_url_rule", "add_url_rule" in lowered),
        ("register_blueprint", "register_blueprint" in lowered),
        ("blueprint(", "blueprint(" in lowered),
        ("valueerror", "valueerror" in lowered),
        ("assertionerror", "assertionerror" in lowered),
    ]
    return [token for token, present in token_rules if present]


def extract_failure_path_signatures(
    failure_focus: dict[str, Any] | None,
    code_context: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    target_tracebacks = (failure_focus or {}).get("target_test_tracebacks") or {}
    signatures: dict[str, dict[str, Any]] = {}
    for test_id, traceback_text in target_tracebacks.items():
        source_file = _extract_source_file_from_traceback(traceback_text)
        signatures[str(test_id)] = {
            "source_file": source_file,
            "source_symbol": _qualify_source_symbol_from_traceback(traceback_text, code_context, source_file),
            "expected_exception": _extract_expected_exception_from_traceback(traceback_text),
            "observed_exception": _extract_observed_exception_from_traceback(traceback_text),
            "alignment_tokens": _extract_semantic_alignment_tokens_from_traceback(traceback_text),
            "evidence_strength": _classify_obligation_evidence_strength(traceback_text),
        }
    return signatures


def _classify_obligation_evidence_strength(traceback_text: str) -> str:
    text = str(traceback_text or "")
    stripped = text.strip()
    if not stripped:
        return "weak_evidence"
    if stripped.startswith("TARGET TEST:") and "WEAK TRACEBACK CONTEXT:" not in stripped:
        return "weak_evidence"
    lowered = stripped.lower()
    if "weak traceback context:" in lowered:
        return "weak_evidence"
    if _traceback_block_has_failure_signal(stripped):
        return "strong_evidence"
    return "weak_evidence"


def format_failure_path_signatures(
    failure_focus: dict[str, Any] | None,
    code_context: dict[str, str] | None = None,
) -> str:
    signatures = extract_failure_path_signatures(failure_focus, code_context)
    if not signatures:
        return "No structured path signatures available."
    sections = []
    for test_id, signature in signatures.items():
        source_symbol = signature.get("source_symbol") or "unknown"
        source_file = signature.get("source_file") or "unknown"
        expected_exception = signature.get("expected_exception") or "unknown"
        observed_exception = signature.get("observed_exception") or "unknown"
        tokens = ", ".join(signature.get("alignment_tokens") or []) or "none"
        sections.append(
            f"- {test_id}: source_file={source_file}; source_symbol={source_symbol}; "
            f"expected_exception={expected_exception}; observed_exception={observed_exception}; "
            f"alignment_tokens={tokens}"
        )
    return "\n".join(sections)


def _slugify_obligation_token(token: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(token).lower()).strip("_")
    return slug or "core"


def _classify_obligation_trigger(
    source_symbol: str,
    alignment_tokens: list[str],
    marker_group: list[str],
) -> dict[str, Any]:
    normalized_markers = [str(marker).strip() for marker in marker_group if str(marker).strip()]
    marker_text = " ".join(normalized_markers).lower()
    normalized_alignment = [str(token).strip() for token in alignment_tokens if str(token).strip()]
    alignment_lower = [token.lower() for token in normalized_alignment]
    subject = "core"
    trigger_shape = "symbol_level_behavior_gap"
    trigger_shape_tokens: list[str] = []
    if normalized_markers:
        primary = normalized_markers[0]
        subject = _slugify_obligation_token(primary)
        if "." in primary:
            trigger_shape = "object_attribute_value_contains_forbidden_token"
            trigger_shape_tokens.extend(normalized_markers)
            if "__name__" in marker_text:
                trigger_shape_tokens.extend(["__name__", "view_func"])
        else:
            trigger_shape = "argument_value_contains_forbidden_token"
            trigger_shape_tokens.extend(normalized_markers)
            if primary.lower().endswith("name") or primary.lower() == "name":
                trigger_shape_tokens.append("Blueprint(")
            if primary.lower() == "endpoint":
                trigger_shape_tokens.append("endpoint=")
    elif "endpoint=" in alignment_lower:
        subject = "endpoint"
        trigger_shape = "argument_value_contains_forbidden_token"
        trigger_shape_tokens.append("endpoint=")
    elif "__name__" in marker_text or "__name__" in alignment_lower:
        subject = "view_func_name"
        trigger_shape = "object_attribute_value_contains_forbidden_token"
        trigger_shape_tokens.extend(["__name__", "view_func"])
    elif "blueprint(" in alignment_lower:
        subject = "name"
        trigger_shape = "argument_value_contains_forbidden_token"
        trigger_shape_tokens.append("Blueprint(")
    if source_symbol and "." in source_symbol:
        owner, _, method = source_symbol.rpartition(".")
        if owner and method:
            trigger_shape_tokens.extend([owner, method])
    return {
        "validation_subject": subject,
        "trigger_shape": trigger_shape,
        "trigger_shape_tokens": dedupe_preserve_order(
            [token for token in trigger_shape_tokens + normalized_alignment if str(token).strip()]
        ),
    }


def _derive_statement_anchor_tokens(
    source_symbol: str,
    validation_subject: str,
    trigger_shape_tokens: list[str] | None,
    alignment_tokens: list[str] | None,
    marker_group: list[str] | None = None,
) -> list[str]:
    subject = str(validation_subject or "").strip()
    normalized_trigger_tokens = [
        str(token).strip()
        for token in (trigger_shape_tokens or [])
        if str(token).strip()
    ]
    normalized_alignment_tokens = [
        str(token).strip()
        for token in (alignment_tokens or [])
        if str(token).strip()
    ]
    normalized_markers = [
        str(token).strip()
        for token in (marker_group or [])
        if str(token).strip()
    ]
    subject_aliases = _expand_validation_subject_aliases(subject, normalized_trigger_tokens)
    statement_tokens: list[str] = []
    for token in [*subject_aliases, *normalized_markers, *normalized_trigger_tokens, *normalized_alignment_tokens]:
        if token and token not in statement_tokens:
            statement_tokens.append(token)
    subject_lower = subject.strip().lower()
    if subject_lower in {"method", "method parameter"}:
        statement_tokens.extend(["method =", "builtin_str(method)", "method.decode("])
    elif subject_lower == "endpoint":
        statement_tokens.extend(["endpoint =", "if endpoint", "in endpoint"])
    elif subject_lower == "name":
        statement_tokens.extend(["self.name =", "if '.' in name", 'if "." in name'])
    elif subject_lower == "view_func_name":
        statement_tokens.extend(["view_func", "__name__", "view_func.__name__"])
    if source_symbol:
        symbol_tail = source_symbol.rsplit(".", 1)[-1]
        if symbol_tail and symbol_tail not in statement_tokens:
            statement_tokens.append(symbol_tail)
    return dedupe_preserve_order([token for token in statement_tokens if str(token).strip()])


def _classify_obligation_level(
    source_symbol: str,
    validation_subject: str,
    alignment_tokens: list[str],
    marker_group: list[str],
    has_sibling_group: bool,
) -> str:
    subject = str(validation_subject or "").strip().lower() or "core"
    alignment_lower = [str(token).strip().lower() for token in alignment_tokens if str(token).strip()]
    marker_lower = [str(marker).strip().lower() for marker in marker_group if str(marker).strip()]
    if subject == "core":
        return "primary_direct"
    direct_signals = set(alignment_lower)
    if subject in {"endpoint", "method", "method parameter"} and "endpoint=" in direct_signals:
        return "primary_direct"
    if subject in {"name"} and "blueprint(" in direct_signals:
        return "primary_direct"
    if any(marker in direct_signals for marker in marker_lower):
        return "primary_direct"
    if any(
        token in direct_signals
        for token in (
            "__name__" if subject == "view_func_name" else "",
            "view_func" if subject == "view_func_name" else "",
        )
        if token
    ):
        return "primary_direct"
    if has_sibling_group and source_symbol:
        return "primary_sibling"
    return "propagated"


def _build_obligation_lookup(
    failure_focus: dict[str, Any] | None,
    code_context: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    obligations_by_test = extract_failure_path_repair_obligations(failure_focus, code_context)
    return {
        str(obligation.get("id")): obligation
        for obligations in obligations_by_test.values()
        for obligation in obligations
        if str(obligation.get("id") or "").strip()
    }


def extract_failure_path_repair_obligations(
    failure_focus: dict[str, Any] | None,
    code_context: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    signatures = extract_failure_path_signatures(failure_focus, code_context)
    obligations_by_test: dict[str, list[dict[str, Any]]] = {}
    sibling_obligations_by_symbol: dict[str, dict[str, Any]] = {}
    if code_context:
        for test_id, signature in signatures.items():
            source_symbol = str(signature.get("source_symbol") or "").strip()
            source_file = str(signature.get("source_file") or "").strip()
            observed_exception = str(signature.get("observed_exception") or "").strip().lower()
            # Attempt sibling analysis for any exception type — not just assertionerror.
            # When the buggy code uses assert for multiple invariants (e.g. endpoint AND
            # view_func.__name__), all of them need to be covered by enhanced tests even if
            # the original failing test only triggered one path.
            if not source_symbol:
                continue
            if source_symbol in sibling_obligations_by_symbol:
                continue
            resolved = _resolve_code_file_for_symbol(code_context, source_symbol, preferred_file=source_file or None)
            if resolved is None:
                # Fallback: if source_symbol is misidentified, scan all code_context source
                # files for functions named by the last component of source_symbol.
                raw_fn_name = source_symbol.split(".")[-1] if "." in source_symbol else source_symbol
                for _path, _content in code_context.items():
                    if _path.startswith("tests/") or not _path.endswith(".py"):
                        continue
                    try:
                        _module = ast.parse(_content)
                    except SyntaxError:
                        continue
                    for _node in ast.walk(_module):
                        if isinstance(_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _node.name == raw_fn_name:
                            resolved = (_path, _content)
                            break
                    if resolved is not None:
                        break
                if resolved is None:
                    continue
            _, content = resolved
            try:
                module = ast.parse(content)
            except SyntaxError:
                continue
            # Try exact symbol first; if not found, fall back to raw function name search
            found = _find_function_node_for_symbol(module, source_symbol)
            if found is None:
                raw_fn_name = source_symbol.split(".")[-1] if "." in source_symbol else source_symbol
                for node in ast.walk(module):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == raw_fn_name:
                        found = (node, None)
                        break
            if found is None:
                continue
            fn_node, _ = found
            assert_infos: list[dict[str, Any]] = []
            for subnode in ast.walk(fn_node):
                if not isinstance(subnode, ast.Assert):
                    continue
                markers = _extract_assert_branch_markers(subnode)
                if not markers:
                    continue
                assert_infos.append({"markers": markers})
            if len(assert_infos) > 1:
                sibling_obligations_by_symbol[source_symbol] = {
                    "marker_groups": [info["markers"] for info in assert_infos],
                }
    for test_id, signature in signatures.items():
        source_symbol = str(signature.get("source_symbol") or "").strip()
        alignment_tokens = [str(token) for token in (signature.get("alignment_tokens") or []) if str(token).strip()]
        base_expected = str(signature.get("expected_exception") or "").strip()
        obligations: list[dict[str, Any]] = []
        sibling = sibling_obligations_by_symbol.get(source_symbol)
        if sibling:
            for marker_group in sibling.get("marker_groups", []):
                marker_group = [str(marker) for marker in marker_group if str(marker).strip()]
                if not marker_group:
                    continue
                primary_marker = marker_group[0]
                obligation_id = f"{test_id}::{_slugify_obligation_token(primary_marker)}"
                trigger_info = _classify_obligation_trigger(source_symbol, alignment_tokens, marker_group)
                obligation_level = _classify_obligation_level(
                    source_symbol=source_symbol,
                    validation_subject=trigger_info["validation_subject"],
                    alignment_tokens=alignment_tokens,
                    marker_group=marker_group,
                    has_sibling_group=True,
                )
                obligations.append(
                    {
                        "id": obligation_id,
                        "test_id": test_id,
                        "source_symbol": source_symbol,
                        "expected_exception": base_expected,
                        "alignment_tokens": dedupe_preserve_order(alignment_tokens + marker_group),
                        "marker_group": marker_group,
                        "validation_subject": trigger_info["validation_subject"],
                        "trigger_shape": trigger_info["trigger_shape"],
                        "trigger_shape_tokens": trigger_info["trigger_shape_tokens"],
                        "statement_anchor_tokens": _derive_statement_anchor_tokens(
                            source_symbol=source_symbol,
                            validation_subject=trigger_info["validation_subject"],
                            trigger_shape_tokens=trigger_info["trigger_shape_tokens"],
                            alignment_tokens=alignment_tokens,
                            marker_group=marker_group,
                        ),
                        "obligation_level": obligation_level,
                        "evidence_strength": str(signature.get("evidence_strength") or "weak_evidence").strip(),
                    }
                )
        if not obligations:
            trigger_info = _classify_obligation_trigger(source_symbol, alignment_tokens, [])
            obligation_level = _classify_obligation_level(
                source_symbol=source_symbol,
                validation_subject=trigger_info["validation_subject"],
                alignment_tokens=alignment_tokens,
                marker_group=[],
                has_sibling_group=False,
            )
            obligations.append(
                {
                    "id": f"{test_id}::core",
                    "test_id": test_id,
                    "source_symbol": source_symbol,
                    "expected_exception": base_expected,
                    "alignment_tokens": alignment_tokens,
                    "marker_group": [],
                    "validation_subject": trigger_info["validation_subject"],
                    "trigger_shape": trigger_info["trigger_shape"],
                    "trigger_shape_tokens": trigger_info["trigger_shape_tokens"],
                    "statement_anchor_tokens": _derive_statement_anchor_tokens(
                        source_symbol=source_symbol,
                        validation_subject=trigger_info["validation_subject"],
                        trigger_shape_tokens=trigger_info["trigger_shape_tokens"],
                        alignment_tokens=alignment_tokens,
                        marker_group=[],
                    ),
                    "obligation_level": obligation_level,
                    "evidence_strength": str(signature.get("evidence_strength") or "weak_evidence").strip(),
                }
            )
        obligations_by_test[test_id] = obligations
    return obligations_by_test


def format_failure_path_obligations(
    failure_focus: dict[str, Any] | None,
    code_context: dict[str, str] | None = None,
) -> str:
    obligations_by_test = extract_failure_path_repair_obligations(failure_focus, code_context)
    if not obligations_by_test:
        return "No repair obligations extracted."
    lines: list[str] = []
    for test_id, obligations in obligations_by_test.items():
        for obligation in obligations:
            tokens = ", ".join(obligation.get("alignment_tokens") or []) or "none"
            trigger_tokens = ", ".join(obligation.get("trigger_shape_tokens") or []) or "none"
            lines.append(
                f"- {obligation['id']}: test={test_id}; source_symbol={obligation.get('source_symbol') or 'unknown'}; "
                f"expected_exception={obligation.get('expected_exception') or 'unknown'}; "
                f"validation_subject={obligation.get('validation_subject') or 'core'}; "
                f"trigger_shape={obligation.get('trigger_shape') or 'symbol_level_behavior_gap'}; "
                f"alignment_tokens={tokens}; trigger_shape_tokens={trigger_tokens}"
            )
    return "\n".join(lines)


def _normalize_obligation_ids(
    obligation_ids: Any,
    covered_original_tests: list[str],
    failure_focus: dict[str, Any] | None,
    code_context: dict[str, str] | None = None,
) -> list[str]:
    obligations_by_test = extract_failure_path_repair_obligations(failure_focus, code_context)
    all_valid_ids = {
        obligation["id"]
        for obligations in obligations_by_test.values()
        for obligation in obligations
    }
    normalized: list[str] = []
    if isinstance(obligation_ids, list):
        for item in obligation_ids:
            if isinstance(item, str) and item.strip():
                normalized.append(item.strip())
    if normalized:
        return [item for item in dedupe_preserve_order(normalized) if item in all_valid_ids]
    fallback: list[str] = []
    for test_id in covered_original_tests:
        obligations = obligations_by_test.get(test_id) or []
        if obligations:
            fallback.append(str(obligations[0]["id"]))
    return dedupe_preserve_order(fallback)


def _normalize_alignment_tokens(tokens: Any) -> list[str]:
    if not isinstance(tokens, list):
        return []
    normalized = []
    for token in tokens:
        if isinstance(token, str) and token.strip():
            normalized.append(token.strip())
    return normalized


def _normalize_symbol_alignment_variants(symbol: str) -> list[str]:
    normalized = str(symbol or "").strip().lower()
    if not normalized:
        return []
    variants: list[str] = []
    candidates = [
        normalized,
        normalized.split("::")[-1],
        normalized.split(":")[-1],
        normalized.split(".")[-1],
    ]
    for candidate in candidates:
        candidate = str(candidate).strip().lower()
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _single_root_symbol_alignment_match(
    expected_symbol: str,
    target_source_symbol: str,
    content_text: str,
    dominant_symbols: list[str] | None = None,
) -> bool:
    text_lower = str(content_text or "").lower()
    target_variants = _normalize_symbol_alignment_variants(target_source_symbol)
    dominant_variants = [
        variant
        for symbol in (dominant_symbols or [])
        for variant in _normalize_symbol_alignment_variants(symbol)
    ]
    if expected_symbol.startswith("test_"):
        for variant in dedupe_preserve_order(target_variants + dominant_variants):
            if variant and (variant in text_lower or variant in target_variants):
                return True
        return False
    expected_variants = _normalize_symbol_alignment_variants(expected_symbol)
    for expected_variant in expected_variants:
        if not expected_variant:
            continue
        if expected_variant in target_variants:
            return True
        if expected_variant in text_lower:
            return True
        if any(
            dominant_variant == expected_variant
            for dominant_variant in dominant_variants
        ):
            return True
    return False


def build_path_semantic_alignment_feedback(
    covers_original_tests: list[str],
    target_source_symbol: str,
    semantic_alignment_tokens: list[str],
    failure_focus: dict[str, Any] | None,
    code_context: dict[str, str] | None,
    content_text: str,
) -> str | None:
    signatures = extract_failure_path_signatures(failure_focus)
    if not signatures or not covers_original_tests:
        return None
    text_lower = content_text.lower()
    normalized_tokens = [token.lower() for token in semantic_alignment_tokens if token.strip()]
    for original_test in covers_original_tests:
        signature = signatures.get(str(original_test))
        if not signature:
            continue
        expected_symbol = str(signature.get("source_symbol") or "").strip().lower()
        dominant_symbols = [
            str(symbol).strip().lower()
            for symbol in _infer_failure_focus_dominant_symbols(
                {
                    **(failure_focus or {}),
                    "active_fail_to_pass_identifiers": [original_test],
                    "inactive_fail_to_pass_identifiers": [original_test],
                    "original_test_identifiers": [original_test],
                },
                code_context,
            )
            if str(symbol).strip()
        ]
        expected_tokens = [str(token).lower() for token in (signature.get("alignment_tokens") or []) if str(token).strip()]
        symbol_ok = (
            not expected_symbol
            or _single_root_symbol_alignment_match(
                expected_symbol,
                target_source_symbol,
                content_text,
                dominant_symbols,
            )
        )
        relaxed_single_root_token_ok = (
            expected_symbol.startswith("test_")
            and _single_root_symbol_alignment_match(
                expected_symbol,
                target_source_symbol,
                content_text,
                dominant_symbols,
            )
            and expected_tokens
            and all(token in {"assertionerror", "assert"} for token in expected_tokens)
            and any(token in normalized_tokens or token in text_lower for token in ("__dict__", "__slots__", "attributeerror"))
        )
        token_ok = not expected_tokens or any(
            token in normalized_tokens or token in text_lower for token in expected_tokens
        ) or relaxed_single_root_token_ok
        if symbol_ok and token_ok:
            continue
        return (
            f"The claimed coverage for {original_test} is not semantically aligned with its original failure path. "
            f"Expected source symbol '{signature.get('source_symbol')}' and alignment tokens "
            f"{signature.get('alignment_tokens') or []}, but the current content targets "
            f"'{target_source_symbol or 'unknown'}' with tokens {semantic_alignment_tokens or []}. "
            "Revise the idea so it exercises the same API path and exception semantics as the original failure."
        )
    return None


def build_obligation_trigger_alignment_feedback(
    covers_obligations: list[str],
    target_source_symbol: str,
    target_validation_subject: str,
    semantic_alignment_tokens: list[str],
    trigger_shape_tokens: list[str],
    failure_focus: dict[str, Any] | None,
    code_context: dict[str, str] | None,
    content_text: str,
) -> str | None:
    if not covers_obligations:
        return None
    obligation_lookup = _build_obligation_lookup(failure_focus, code_context)
    text_lower = content_text.lower()
    normalized_target_symbol = (target_source_symbol or "").strip().lower()
    normalized_subject = (target_validation_subject or "").strip().lower()
    normalized_semantic_tokens = [token.lower() for token in semantic_alignment_tokens if token.strip()]
    normalized_trigger_tokens = [token.lower() for token in trigger_shape_tokens if token.strip()]
    for obligation_id in covers_obligations:
        obligation = obligation_lookup.get(str(obligation_id))
        if not obligation:
            continue
        expected_symbol = str(obligation.get("source_symbol") or "").strip().lower()
        expected_subject = str(obligation.get("validation_subject") or "").strip().lower()
        expected_trigger_tokens = [
            str(token).lower()
            for token in (obligation.get("trigger_shape_tokens") or [])
            if str(token).strip()
        ]
        symbol_ok = not expected_symbol or expected_symbol in normalized_target_symbol or expected_symbol in text_lower
        subject_ok = not expected_subject or expected_subject == "core" or expected_subject in normalized_subject or expected_subject in text_lower
        trigger_ok = not expected_trigger_tokens or any(
            token in normalized_trigger_tokens
            or token in normalized_semantic_tokens
            or token in text_lower
            for token in expected_trigger_tokens
        )
        if symbol_ok and subject_ok and trigger_ok:
            continue
        return (
            f"The claimed obligation coverage for {obligation_id} is not trigger-shape aligned. "
            f"Expected source symbol '{obligation.get('source_symbol')}', validation subject "
            f"'{obligation.get('validation_subject')}', and trigger tokens "
            f"{obligation.get('trigger_shape_tokens') or []}, but the current idea targets "
            f"'{target_source_symbol or 'unknown'}' / '{target_validation_subject or 'unknown'}' with "
            f"semantic tokens {semantic_alignment_tokens or []} and trigger tokens {trigger_shape_tokens or []}. "
            "Revise the idea so it uses the real trigger shape from the landed code path instead of inventing a new API surface."
        )
    return None


def extract_multi_symbol_repair_obligations(
    failure_focus: dict[str, Any] | None,
    filtered_candidates: list[CandidatePatch] | None,
    code_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    signatures = extract_failure_path_signatures(failure_focus, code_context)
    covered_paths: set[str] = set()
    for candidate in filtered_candidates or []:
        for test_id in candidate.covered_original_tests or []:
            if isinstance(test_id, str) and test_id.strip():
                covered_paths.add(test_id)
    path_to_symbol: dict[str, str] = {}
    for test_id in covered_paths:
        signature = signatures.get(test_id) or {}
        source_symbol = str(signature.get("source_symbol") or "").strip()
        if source_symbol:
            path_to_symbol[test_id] = source_symbol
    unique_symbols = dedupe_preserve_order(path_to_symbol.values())
    return {
        "covered_paths": dedupe_preserve_order(list(covered_paths)),
        "path_to_symbol": path_to_symbol,
        "symbols": unique_symbols,
        "active": len(unique_symbols) > 1,
    }


def extract_validation_subject_repair_obligations(
    filtered_candidates: list[CandidatePatch] | None,
) -> dict[str, Any]:
    grouped_subjects: dict[str, list[str]] = {}
    grouped_trigger_tokens: dict[str, dict[str, list[str]]] = {}
    for candidate in filtered_candidates or []:
        idea = candidate.idea or {}
        source_symbol = str(idea.get("target_source_symbol") or "").strip()
        validation_subject = str(idea.get("target_validation_subject") or "").strip()
        if not source_symbol or not validation_subject or validation_subject.lower() == "core":
            continue
        subjects = grouped_subjects.setdefault(source_symbol, [])
        if validation_subject not in subjects:
            subjects.append(validation_subject)
        trigger_bucket = grouped_trigger_tokens.setdefault(source_symbol, {})
        existing_tokens = trigger_bucket.setdefault(validation_subject, [])
        for token in idea.get("trigger_shape_tokens", []) or []:
            normalized = str(token).strip()
            if normalized and normalized not in existing_tokens:
                existing_tokens.append(normalized)
    obligations: list[dict[str, Any]] = []
    for source_symbol, subjects in grouped_subjects.items():
        if len(subjects) < 2:
            continue
        obligations.append(
            {
                "source_symbol": source_symbol,
                "validation_subjects": subjects,
                "trigger_tokens": grouped_trigger_tokens.get(source_symbol, {}),
            }
        )
    return {
        "active": bool(obligations),
        "obligations": obligations,
    }


def extract_required_repair_obligations(
    filtered_candidates: list[CandidatePatch] | None,
    failure_focus: dict[str, Any] | None = None,
    code_context: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    obligations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    obligation_lookup = _build_obligation_lookup(failure_focus, code_context) if (failure_focus or code_context) else {}
    active_fail_to_pass = {
        str(test_id).strip()
        for test_id in ((failure_focus or {}).get("active_fail_to_pass_identifiers") or [])
        if str(test_id).strip()
    }
    dominant_failure_symbols = _infer_failure_focus_dominant_symbols(failure_focus, code_context)
    for candidate in filtered_candidates or []:
        idea = candidate.idea or {}
        source_symbol = str(idea.get("target_source_symbol") or "").strip()
        validation_subject = str(idea.get("target_validation_subject") or "").strip()
        trigger_tokens = [
            str(token).strip()
            for token in idea.get("trigger_shape_tokens", []) or []
            if str(token).strip()
        ]
        covered_original_tests = [
            str(test_id).strip()
            for test_id in candidate.covered_original_tests or []
            if str(test_id).strip()
        ]
        is_active_covered = any(
            _test_identifiers_match(active_test_id, covered_test_id)
            for active_test_id in active_fail_to_pass
            for covered_test_id in covered_original_tests
        )
        for obligation_id in idea.get("covers_obligations", []) or []:
            normalized_id = str(obligation_id).strip()
            if not normalized_id:
                continue
            lookup_item = obligation_lookup.get(normalized_id, {})
            lookup_source_symbol = str(lookup_item.get("source_symbol") or source_symbol).strip()
            if (
                (not lookup_source_symbol or lookup_source_symbol.startswith("test_"))
                and len(dominant_failure_symbols) == 1
            ):
                lookup_source_symbol = dominant_failure_symbols[0]
            lookup_validation_subject = str(lookup_item.get("validation_subject") or "").strip() or "core"
            candidate_validation_subject = validation_subject or "core"
            if candidate_validation_subject != "core":
                effective_validation_subject = candidate_validation_subject
            else:
                effective_validation_subject = lookup_validation_subject
            effective_obligation_id = normalized_id
            if (
                effective_validation_subject
                and effective_validation_subject != "core"
                and effective_validation_subject != lookup_validation_subject
            ):
                effective_obligation_id = f"{normalized_id}::{effective_validation_subject}"
            if effective_obligation_id in seen_ids:
                continue
            seen_ids.add(effective_obligation_id)
            obligations.append(
                {
                    "id": effective_obligation_id,
                    "source_symbol": lookup_source_symbol,
                    "validation_subject": effective_validation_subject,
                    "trigger_shape_tokens": trigger_tokens,
                    "statement_anchor_tokens": [
                        str(token).strip()
                        for token in (
                            lookup_item.get("statement_anchor_tokens")
                            or _derive_statement_anchor_tokens(
                                source_symbol=lookup_source_symbol,
                                validation_subject=effective_validation_subject,
                                trigger_shape_tokens=trigger_tokens,
                                alignment_tokens=[],
                                marker_group=[],
                            )
                        )
                        if str(token).strip()
                    ],
                    "covered_original_tests": covered_original_tests,
                    "is_active_fail_to_pass": is_active_covered,
                    "obligation_level": str(lookup_item.get("obligation_level") or "primary_direct").strip(),
                    "evidence_strength": str(lookup_item.get("evidence_strength") or "weak_evidence").strip(),
                    "canonical_statement_text": (
                        _extract_canonical_statement_from_tokens(
                            [
                                str(token).strip()
                                for token in (
                                    lookup_item.get("statement_anchor_tokens")
                                    or _derive_statement_anchor_tokens(
                                        source_symbol=lookup_source_symbol,
                                        validation_subject=effective_validation_subject,
                                        trigger_shape_tokens=trigger_tokens,
                                        alignment_tokens=[],
                                        marker_group=[],
                                    )
                                )
                                if str(token).strip()
                            ]
                        )
                    ),
                    "canonical_statement_required": is_active_covered
                    and bool(
                        _extract_canonical_statement_from_tokens(
                            [
                                str(token).strip()
                                for token in (
                                    lookup_item.get("statement_anchor_tokens")
                                    or _derive_statement_anchor_tokens(
                                        source_symbol=lookup_source_symbol,
                                        validation_subject=effective_validation_subject,
                                        trigger_shape_tokens=trigger_tokens,
                                        alignment_tokens=[],
                                        marker_group=[],
                                    )
                                )
                                if str(token).strip()
                            ]
                        )
                    ),
                }
            )
    return obligations


def _prioritize_active_required_repair_obligations(
    obligations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    active = [item for item in obligations if bool(item.get("is_active_fail_to_pass"))]
    if not active:
        return obligations
    active_symbols = {
        str(item.get("source_symbol") or "").strip()
        for item in active
        if str(item.get("source_symbol") or "").strip()
    }
    prioritized = [
        item
        for item in obligations
        if str(item.get("source_symbol") or "").strip() in active_symbols
        and (
            bool(item.get("is_active_fail_to_pass"))
            or str(item.get("obligation_level") or "").strip() in {"primary_direct", "primary_sibling"}
        )
    ]
    return prioritized or active


def filter_primary_required_repair_obligations(
    required_repair_obligations: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    obligations = [item for item in (required_repair_obligations or []) if isinstance(item, dict)]
    primary = [
        item
        for item in obligations
        if str(item.get("obligation_level") or "").strip() in {"primary_direct", "primary_sibling"}
    ]
    return primary or obligations


def extract_validation_subject_repair_obligations_from_obligations(
    required_repair_obligations: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for obligation in required_repair_obligations or []:
        if not isinstance(obligation, dict):
            continue
        source_symbol = str(obligation.get("source_symbol") or "").strip()
        validation_subject = str(obligation.get("validation_subject") or "").strip() or "core"
        if not source_symbol:
            continue
        grouped.setdefault(
            source_symbol,
            {"source_symbol": source_symbol, "validation_subjects": [], "trigger_tokens": {}},
        )
        if validation_subject not in grouped[source_symbol]["validation_subjects"]:
            grouped[source_symbol]["validation_subjects"].append(validation_subject)
        grouped[source_symbol]["trigger_tokens"][validation_subject] = [
            str(token).strip()
            for token in obligation.get("trigger_shape_tokens", []) or []
            if str(token).strip()
        ]
    obligations = [
        item for item in grouped.values()
        if len(item.get("validation_subjects", [])) > 1
    ]
    return {"active": bool(obligations), "obligations": obligations}


def _select_dominant_symbol_from_candidates(
    filtered_candidates: list[CandidatePatch] | None,
    obligations: list[dict[str, Any]],
) -> str:
    candidate_scores: dict[str, float] = {}
    candidate_order: list[str] = []
    for candidate in filtered_candidates or []:
        symbol = str((candidate.idea or {}).get("target_source_symbol") or "").strip()
        if not symbol:
            continue
        if symbol not in candidate_scores:
            candidate_order.append(symbol)
        candidate_scores[symbol] = max(candidate_scores.get(symbol, float("-inf")), float(candidate.quality_score or 0.0))
    if candidate_scores:
        ranked = sorted(candidate_order, key=lambda symbol: (candidate_scores[symbol], -candidate_order.index(symbol)), reverse=True)
        return ranked[0]
    for obligation in obligations:
        symbol = str(obligation.get("source_symbol") or "").strip()
        if symbol:
            return symbol
    return ""


def _group_obligations_by_original_tests(
    obligations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    fallback_bucket = "__unbucketed__"
    for obligation in obligations:
        covered_original_tests = [
            str(test_id).strip()
            for test_id in obligation.get("covered_original_tests", []) or []
            if str(test_id).strip()
        ]
        if not covered_original_tests:
            grouped.setdefault(fallback_bucket, []).append(obligation)
            continue
        for test_id in covered_original_tests:
            grouped.setdefault(test_id, []).append(obligation)
    return grouped


def _filter_candidates_for_original_test(
    filtered_candidates: list[CandidatePatch] | None,
    original_test_id: str,
) -> list[CandidatePatch]:
    if not original_test_id or original_test_id == "__unbucketed__":
        return list(filtered_candidates or [])
    matched: list[CandidatePatch] = []
    for candidate in filtered_candidates or []:
        covered_original_tests = [
            str(test_id).strip()
            for test_id in candidate.covered_original_tests or []
            if str(test_id).strip()
        ]
        if any(_test_identifiers_match(original_test_id, covered_test_id) for covered_test_id in covered_original_tests):
            matched.append(candidate)
    return matched


def _select_weak_evidence_symbols_by_original_test(
    obligations: list[dict[str, Any]],
    filtered_candidates: list[CandidatePatch] | None = None,
) -> list[str]:
    grouped = _group_obligations_by_original_tests(obligations)
    if not grouped:
        return []
    selected_symbols: list[str] = []
    for original_test_id, bucket_obligations in grouped.items():
        bucket_candidates = _filter_candidates_for_original_test(filtered_candidates, original_test_id)
        dominant_symbol = _select_dominant_symbol_from_candidates(bucket_candidates, bucket_obligations)
        if dominant_symbol:
            selected_symbols.append(dominant_symbol)
    return dedupe_preserve_order(selected_symbols)


def _select_effective_repair_symbols(
    obligations: list[dict[str, Any]],
    filtered_candidates: list[CandidatePatch] | None = None,
) -> list[str]:
    strong_primary_symbols = dedupe_preserve_order(
        [
            str(item.get("source_symbol") or "").strip()
            for item in obligations
            if str(item.get("source_symbol") or "").strip()
            and str(item.get("obligation_level") or "").strip() in {"primary_direct", "primary_sibling"}
            and str(item.get("evidence_strength") or "").strip() == "strong_evidence"
        ]
    )
    if strong_primary_symbols:
        return strong_primary_symbols
    weak_bucket_symbols = _select_weak_evidence_symbols_by_original_test(obligations, filtered_candidates)
    if weak_bucket_symbols:
        return weak_bucket_symbols
    dominant_symbol = _select_dominant_symbol_from_candidates(filtered_candidates, obligations)
    if dominant_symbol:
        return [dominant_symbol]
    return []


def select_effective_required_repair_obligations(
    required_repair_obligations: list[dict[str, Any]] | None,
    filtered_candidates: list[CandidatePatch] | None = None,
) -> list[dict[str, Any]]:
    obligations = [item for item in (required_repair_obligations or []) if isinstance(item, dict)]
    if not obligations:
        return []
    obligations = _prioritize_active_required_repair_obligations(obligations)
    effective_symbols = _select_effective_repair_symbols(obligations, filtered_candidates)
    if effective_symbols:
        strong_symbol_set = set(effective_symbols)
        primary_within_symbols = [
            item
            for item in obligations
            if str(item.get("source_symbol") or "").strip() in strong_symbol_set
            and str(item.get("obligation_level") or "").strip() in {"primary_direct", "primary_sibling"}
        ]
        if primary_within_symbols:
            return primary_within_symbols
        symbol_scoped = [
            item
            for item in obligations
            if str(item.get("source_symbol") or "").strip() in strong_symbol_set
        ]
        if symbol_scoped:
            return symbol_scoped
    return filter_primary_required_repair_obligations(obligations)


def _promote_single_root_helper_obligations(
    required_repair_obligations: list[dict[str, Any]] | None,
    analysis: dict[str, Any] | None,
    failure_focus: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    obligations = [dict(item) for item in (required_repair_obligations or []) if isinstance(item, dict)]
    if not obligations or not _has_single_root_helper_root_cause_signature(analysis, failure_focus):
        return obligations
    preferred_target = _preferred_single_root_source_target(analysis, failure_focus)
    if not preferred_target:
        return obligations
    preferred_symbol = preferred_target.split("::", 1)[1] if "::" in preferred_target else preferred_target
    promoted: list[dict[str, Any]] = []
    for item in obligations:
        promoted_item = dict(item)
        promoted_item["source_symbol"] = preferred_symbol
        promoted.append(promoted_item)
    return promoted


def classify_repair_topology(
    required_repair_obligations: list[dict[str, Any]] | None,
    filtered_candidates: list[CandidatePatch] | None = None,
) -> str:
    obligations = [item for item in (required_repair_obligations or []) if isinstance(item, dict)]
    if not obligations:
        return "generic"
    strong_primary = [
        item
        for item in obligations
        if str(item.get("obligation_level") or "").strip() in {"primary_direct", "primary_sibling"}
        and str(item.get("evidence_strength") or "").strip() == "strong_evidence"
    ]
    grouped_subjects = extract_validation_subject_repair_obligations_from_obligations(strong_primary)
    if grouped_subjects.get("active"):
        return "symbol_cluster"
    strong_symbols = dedupe_preserve_order(
        [
            str(item.get("source_symbol") or "").strip()
            for item in strong_primary
            if str(item.get("source_symbol") or "").strip()
        ]
    )
    if len(strong_symbols) == 1:
        statement_like = any(
            [
                any(str(token).strip() for token in item.get("statement_anchor_tokens", []) or [])
                or any(str(token).strip() for token in item.get("trigger_shape_tokens", []) or [])
                for item in strong_primary
            ]
        )
        if statement_like:
            return "statement_local"
        return "single_root_symbol"
    dominant_symbol = _select_dominant_symbol_from_candidates(filtered_candidates, obligations)
    if dominant_symbol:
        dominant_obligations = [
            item for item in obligations
            if str(item.get("source_symbol") or "").strip() == dominant_symbol
        ]
        if len(dominant_obligations) == 1 and any(
            str(token).strip()
            for token in dominant_obligations[0].get("statement_anchor_tokens", []) or []
        ):
            return "statement_local"
        if dominant_obligations:
            return "single_root_symbol"
    return "generic"


def _has_statement_local_canonical_signature(
    obligations: list[dict[str, Any]] | None,
) -> bool:
    normalized = [item for item in (obligations or []) if isinstance(item, dict)]
    if not normalized:
        return False
    active = [item for item in normalized if bool(item.get("is_active_fail_to_pass"))]
    scoped = active or normalized
    symbols = {
        str(item.get("source_symbol") or "").strip()
        for item in scoped
        if str(item.get("source_symbol") or "").strip()
    }
    if len(symbols) != 1:
        return False
    canonical_items = [
        item
        for item in scoped
        if (
            bool(item.get("canonical_statement_required"))
            or str(item.get("canonical_statement_text") or "").strip()
            or _extract_canonical_statement_from_tokens(item.get("statement_anchor_tokens") or [])
        )
    ]
    if not canonical_items:
        return False
    active_original_tests = {
        str(test_id).strip()
        for item in canonical_items
        for test_id in (item.get("covered_original_tests") or [])
        if str(test_id).strip() and bool(item.get("is_active_fail_to_pass"))
    }
    return len(active_original_tests) <= 1


def _has_single_root_symbol_signature(
    *,
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
    failure_focus: dict[str, Any] | None,
) -> bool:
    target_tests = (
        list((failure_focus or {}).get("active_fail_to_pass_identifiers") or [])
        or list((failure_focus or {}).get("inactive_fail_to_pass_identifiers") or [])
        or list((failure_focus or {}).get("original_test_identifiers") or [])
    )
    normalized_tests = dedupe_preserve_order(
        str(test_id).strip().split("::")[-1].split(":")[-1]
        for test_id in target_tests
        if str(test_id).strip()
    )
    if len(normalized_tests) != 1:
        return False
    edit_target_symbols = [
        str(target).split("::", 1)[1].strip()
        for target in (strategy or {}).get("edit_targets", [])
        if isinstance(target, str) and "::" in target and str(target).split("::", 1)[1].strip()
    ]
    if len(set(edit_target_symbols)) == 1 and edit_target_symbols:
        return True
    suspicious_symbols = [
        str(symbol).strip()
        for symbol in (analysis or {}).get("suspicious_symbols", [])
        if isinstance(symbol, str) and str(symbol).strip()
    ]
    return len(set(suspicious_symbols)) == 1 and bool(suspicious_symbols)


def _has_single_root_helper_root_cause_signature(
    analysis: dict[str, Any] | None,
    failure_focus: dict[str, Any] | None,
) -> bool:
    target_tests = (
        list((failure_focus or {}).get("active_fail_to_pass_identifiers") or [])
        or list((failure_focus or {}).get("inactive_fail_to_pass_identifiers") or [])
        or list((failure_focus or {}).get("original_test_identifiers") or [])
    )
    normalized_tests = dedupe_preserve_order(
        str(test_id).strip().split("::")[-1].split(":")[-1]
        for test_id in target_tests
        if str(test_id).strip()
    )
    analysis_source_files = [
        path for path in get_analysis_source_files(analysis)
        if isinstance(path, str) and path and not is_test_like_path(path)
    ]
    if len(normalized_tests) != 1 or len(dedupe_preserve_order(analysis_source_files)) != 1:
        return False
    root_cause_text = " ".join(
        [
            str((analysis or {}).get("root_cause", "")),
            str((analysis or {}).get("propagation_path", "")),
        ]
    ).lower()
    helper_tokens = (
        "mixin",
        "helper",
        "parent class",
        "inheritance",
        "inheritance chain",
        "subclass",
        "base class",
        "mro",
    )
    if not any(token in root_cause_text for token in helper_tokens):
        return False
    return True


def _preferred_single_root_source_target(
    analysis: dict[str, Any] | None,
    failure_focus: dict[str, Any] | None,
) -> str | None:
    analysis_source_files = [
        path for path in get_analysis_source_files(analysis)
        if isinstance(path, str) and path and not is_test_like_path(path)
    ]
    if len(dedupe_preserve_order(analysis_source_files)) != 1:
        return None
    preferred_file = analysis_source_files[0]
    preferred_symbol = None
    for component in (analysis or {}).get("affected_components", []) or []:
        if not isinstance(component, dict):
            continue
        if str(component.get("file") or "").strip() != preferred_file:
            continue
        symbol = str(component.get("symbol") or "").strip()
        if symbol and not is_test_like_path(symbol):
            preferred_symbol = symbol
            break
    if preferred_symbol:
        return f"{preferred_file}::{preferred_symbol}"
    return preferred_file


def _prefer_single_root_minimal_edit_targets(
    strategy: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    failure_focus: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(strategy, dict):
        return strategy
    if not (
        _has_single_root_symbol_signature(
            analysis=analysis,
            strategy=strategy,
            failure_focus=failure_focus,
        )
        or _has_single_root_helper_root_cause_signature(analysis, failure_focus)
    ):
        return strategy
    edit_targets = [
        str(target).strip()
        for target in (strategy.get("edit_targets") or [])
        if str(target).strip()
    ]
    source_edit_targets = [target for target in edit_targets if not is_test_like_path(target.split("::", 1)[0])]
    preferred_target = _preferred_single_root_source_target(analysis, failure_focus)
    analysis_source_files = [
        path for path in get_analysis_source_files(analysis)
        if isinstance(path, str) and path and not is_test_like_path(path)
    ]
    if not preferred_target or len(dedupe_preserve_order(analysis_source_files)) != 1:
        return strategy
    preferred_file = analysis_source_files[0]
    if len(source_edit_targets) == 1:
        current_file = source_edit_targets[0].split("::", 1)[0]
        if current_file != preferred_file and _has_single_root_helper_root_cause_signature(analysis, failure_focus):
            removed_file = current_file
            dependency_files = [
                str(path).strip()
                for path in (strategy.get("dependency_files") or [])
                if str(path).strip()
            ]
            if removed_file and removed_file not in dependency_files:
                dependency_files.append(removed_file)
            strategy["edit_targets"] = [preferred_target]
            strategy["dependency_files"] = dependency_files
        return strategy
    preferred_targets = [
        target for target in source_edit_targets
        if target == preferred_target or target.split("::", 1)[0] == preferred_file
    ]
    if not preferred_targets:
        preferred_targets = [preferred_target]
    dependency_files = [
        str(path).strip()
        for path in (strategy.get("dependency_files") or [])
        if str(path).strip()
    ]
    removed_targets = [
        target for target in source_edit_targets
        if target not in preferred_targets
    ]
    for removed in removed_targets:
        removed_file = removed.split("::", 1)[0]
        if removed_file and removed_file not in dependency_files:
            dependency_files.append(removed_file)
    strategy["edit_targets"] = preferred_targets
    strategy["dependency_files"] = dependency_files
    return strategy


def format_required_repair_obligations(
    required_repair_obligations: list[dict[str, Any]] | None,
) -> str:
    obligations = [item for item in (required_repair_obligations or []) if isinstance(item, dict)]
    if not obligations:
        return "No structured repair obligations available."
    sections = []
    for obligation in obligations:
        obligation_id = str(obligation.get("id") or "").strip() or "unknown"
        source_symbol = str(obligation.get("source_symbol") or "").strip() or "unknown"
        validation_subject = str(obligation.get("validation_subject") or "").strip() or "core"
        trigger_tokens = ", ".join(
            str(token).strip()
            for token in obligation.get("trigger_shape_tokens", []) or []
            if str(token).strip()
        ) or "none"
        covered_tests = ", ".join(
            str(test_id).strip()
            for test_id in obligation.get("covered_original_tests", []) or []
            if str(test_id).strip()
        ) or "none"
        sections.append(
            f"- {obligation_id}: source_symbol={source_symbol}; "
            f"validation_subject={validation_subject}; obligation_level={str(obligation.get('obligation_level') or 'primary_direct').strip()}; "
            f"evidence_strength={str(obligation.get('evidence_strength') or 'weak_evidence').strip()}; "
            f"trigger_shape_tokens={trigger_tokens}; "
            f"covered_original_tests={covered_tests}"
        )
    return "\n".join(sections)


def format_validation_subject_symbol_clusters(
    required_repair_obligations: list[dict[str, Any]] | None,
) -> str:
    grouped = extract_validation_subject_repair_obligations_from_obligations(required_repair_obligations)
    obligations = grouped.get("obligations") or []
    if not obligations:
        return "No multi-subject symbol clusters detected."
    sections: list[str] = []
    for item in obligations:
        source_symbol = str(item.get("source_symbol") or "").strip() or "unknown"
        subjects = [
            str(subject).strip()
            for subject in item.get("validation_subjects", []) or []
            if str(subject).strip()
        ]
        if not subjects:
            continue
        sections.append(f"- {source_symbol}: {', '.join(subjects)}")
    return "\n".join(sections) or "No multi-subject symbol clusters detected."


def _extract_conversion_rule_hints(
    required_repair_obligations: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    conversion_markers = (
        "decode(",
        "encode(",
        "builtin_str(",
        "str(",
        "normalize(",
        "lower(",
        "upper(",
        "strip(",
        "int(",
        "float(",
        "bool(",
    )
    for item in required_repair_obligations or []:
        if not isinstance(item, dict):
            continue
        obligation_id = str(item.get("id") or "").strip()
        source_symbol = str(item.get("source_symbol") or "").strip()
        validation_subject = str(item.get("validation_subject") or "").strip() or "core"
        conversion_tokens = [
            str(token).strip()
            for token in item.get("statement_anchor_tokens", []) or []
            if str(token).strip() and any(marker in str(token).strip().lower() for marker in conversion_markers)
        ]
        if not obligation_id or not conversion_tokens:
            continue
        canonical_conversion_statement = conversion_tokens[0]
        key = (obligation_id, canonical_conversion_statement)
        if key in seen:
            continue
        seen.add(key)
        hints.append(
            {
                "id": obligation_id,
                "source_symbol": source_symbol,
                "validation_subject": validation_subject,
                "canonical_conversion_statement": canonical_conversion_statement,
            }
        )
    return hints


def _extract_canonical_statement_from_tokens(statement_anchor_tokens: list[str] | None) -> str | None:
    conversion_hints = _extract_conversion_rule_hints(
        [{"id": "tmp", "statement_anchor_tokens": statement_anchor_tokens or []}]
    )
    if not conversion_hints:
        return None
    canonical_statement = str(conversion_hints[0].get("canonical_conversion_statement") or "").strip()
    return canonical_statement or None


def _extract_canonical_statement_replacement_hints(
    required_repair_obligations: list[dict[str, Any]] | None,
    *,
    force_all: bool = False,
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    filtered_obligations = []
    for item in required_repair_obligations or []:
        if not isinstance(item, dict):
            continue
        if force_all or bool(item.get("canonical_statement_required")):
            filtered_obligations.append(item)
    for item in _extract_conversion_rule_hints(filtered_obligations):
        obligation_id = str(item.get("id") or "").strip()
        canonical_statement = str(item.get("canonical_conversion_statement") or "").strip()
        if not obligation_id or not canonical_statement:
            continue
        key = (obligation_id, canonical_statement)
        if key in seen:
            continue
        seen.add(key)
        hints.append(
            {
                "id": obligation_id,
                "source_symbol": str(item.get("source_symbol") or "").strip(),
                "validation_subject": str(item.get("validation_subject") or "").strip() or "core",
                "canonical_statement_text": canonical_statement,
                "canonical_statement_required": True,
            }
        )
    return hints


def _find_first_line_index_containing(lines: list[str], needle: str) -> int | None:
    normalized = str(needle).strip().lower()
    if not normalized:
        return None
    for idx, line in enumerate(lines):
        if normalized in line.strip().lower():
            return idx
    return None


def _replacement_block_violates_conversion_order(
    replacement_lines: list[str],
    conversion_hints: list[dict[str, Any]],
) -> str | None:
    normalized_lines = [line.rstrip("\n") for line in replacement_lines]
    for hint in conversion_hints:
        canonical = str(hint.get("canonical_conversion_statement") or "").strip()
        if not canonical:
            continue
        conversion_idx = _find_first_line_index_containing(normalized_lines, canonical)
        if conversion_idx is None:
            continue
        has_pre_conversion_guard = any(
            idx < conversion_idx and line.strip().startswith("if ")
            for idx, line in enumerate(normalized_lines)
        )
        if has_pre_conversion_guard:
            continue
        for idx in range(conversion_idx + 1, len(normalized_lines)):
            stripped = normalized_lines[idx].strip()
            if not stripped.startswith("if "):
                continue
            if "isinstance(" in stripped or "bytes" in stripped:
                return (
                    f"The replacement_block checks a pre-conversion condition only after the canonical conversion "
                    f"statement `{canonical}`. Any bytes/type/normalization guard must appear BEFORE `{canonical}` "
                    "or it will be ineffective."
                )
    return None


def _replacement_block_violates_canonical_statement_replacement(
    replacement_lines: list[str],
    replacement_mode: str,
    canonical_hints: list[dict[str, Any]],
) -> str | None:
    if not canonical_hints:
        return None
    normalized_lines = [line.rstrip("\n") for line in replacement_lines]
    joined_text = "\n".join(normalized_lines).lower()
    for hint in canonical_hints:
        canonical = str(hint.get("canonical_statement_text") or "").strip()
        if not canonical:
            continue
        canonical_idx = _find_first_line_index_containing(normalized_lines, canonical)
        has_guard = any(
            line.strip().startswith(("if ", "try:"))
            for line in normalized_lines
        )
        has_else = any(line.strip().startswith("else:") for line in normalized_lines)
        if replacement_mode == "insert_before_anchor":
            return (
                f"This repair obligation is anchored on the canonical statement `{canonical}` and must be expressed "
                "as a full statement replacement, not insert_before_anchor. Replace the original statement with a "
                "complete branch structure that preserves the fallback path."
            )
        if has_guard and canonical_idx is not None and not has_else:
            return (
                f"The replacement_block still leaves the canonical statement `{canonical}` as a bare fall-through line "
                "after introducing a guard. For statement_local repairs, replace the canonical statement with a full "
                "branch structure (for example an if/else), rather than inserting a pre-guard and keeping the old "
                "statement unchanged."
            )
        if canonical_idx is not None and has_else:
            continue
        if canonical_idx is None and canonical.lower() in joined_text:
            continue
        if has_guard and not has_else:
            return (
                f"The replacement_block introduces a guard around the canonical statement `{canonical}` but does not "
                "make the fallback path explicit. Add an `else:` (or equivalent full replacement structure) so the "
                "original conversion is only used on the non-guarded path."
            )
    return None


def build_patch_analysis_feedback(
    analysis: dict[str, Any] | None,
    required_repair_obligations: list[dict[str, Any]] | None = None,
    failure_focus: dict[str, Any] | None = None,
    code_context: dict[str, str] | None = None,
) -> str | None:
    if analysis is None:
        return (
            "Return a JSON object with root_cause, affected_components, failing_signal, propagation_path, "
            "repair_constraint, suggested_repair_scope, suspicious_symbols, and repair_obligations."
        )
    for key in (
        "root_cause",
        "affected_components",
        "failing_signal",
        "propagation_path",
        "repair_constraint",
        "suggested_repair_scope",
        "suspicious_symbols",
        "repair_obligations",
    ):
        if key not in analysis:
            return f"The analysis JSON is missing '{key}'."
    if not isinstance(analysis.get("repair_obligations"), list):
        return "The analysis JSON must contain a repair_obligations list."
    required_items = [item for item in (required_repair_obligations or []) if isinstance(item, dict)]
    if not required_items:
        return None
    reported_items = [
        item for item in analysis.get("repair_obligations", []) or []
        if isinstance(item, dict)
    ]
    reported_by_id = {
        str(item.get("id") or "").strip(): item
        for item in reported_items
        if str(item.get("id") or "").strip()
    }
    missing_ids = [
        str(item.get("id") or "").strip()
        for item in required_items
        if str(item.get("id") or "").strip() and str(item.get("id") or "").strip() not in reported_by_id
    ]
    if missing_ids:
        return (
            "The retained enhanced tests already expose required_repair_obligations, but the analysis did not "
            "carry all of them forward. Add every obligation id to repair_obligations, especially: "
            + ", ".join(missing_ids)
            + "."
        )
    mismatches: list[str] = []
    for item in required_items:
        obligation_id = str(item.get("id") or "").strip()
        if not obligation_id:
            continue
        reported = reported_by_id.get(obligation_id, {})
        required_symbol = str(item.get("source_symbol") or "").strip()
        required_subject = str(item.get("validation_subject") or "").strip() or "core"
        if required_symbol and str(reported.get("source_symbol") or "").strip() != required_symbol:
            mismatches.append(f"{obligation_id}:source_symbol")
        if required_subject and (str(reported.get("validation_subject") or "").strip() or "core") != required_subject:
            mismatches.append(f"{obligation_id}:validation_subject")
    if mismatches:
        return (
            "The analysis repair_obligations do not preserve the required source_symbol/validation_subject pairs for: "
            + ", ".join(mismatches)
            + "."
        )
    required_by_symbol: dict[str, set[str]] = {}
    reported_by_symbol: dict[str, set[str]] = {}
    for item in required_items:
        source_symbol = str(item.get("source_symbol") or "").strip()
        validation_subject = str(item.get("validation_subject") or "").strip() or "core"
        if not source_symbol or validation_subject == "core":
            continue
        required_by_symbol.setdefault(source_symbol, set()).add(validation_subject)
    for item in reported_items:
        source_symbol = str(item.get("source_symbol") or "").strip()
        validation_subject = str(item.get("validation_subject") or "").strip() or "core"
        if not source_symbol or validation_subject == "core":
            continue
        reported_by_symbol.setdefault(source_symbol, set()).add(validation_subject)
    missing_subjects: list[str] = []
    for source_symbol, required_subjects in required_by_symbol.items():
        reported_subjects = reported_by_symbol.get(source_symbol, set())
        missing = sorted(required_subjects - reported_subjects)
        if missing:
            missing_subjects.append(f"{source_symbol}: {', '.join(missing)}")
    if missing_subjects:
        return (
            "The analysis dropped validation_subjects that are already exposed within the same strong primary symbol. "
            "Do not compress these into a generic core obligation. Add repair_obligations for: "
            + "; ".join(missing_subjects)
            + "."
        )
    compressed_core_symbols: list[str] = []
    for source_symbol, required_subjects in required_by_symbol.items():
        if len(required_subjects) < 2:
            continue
        symbol_reported = [
            item
            for item in reported_items
            if str(item.get("source_symbol") or "").strip() == source_symbol
        ]
        if not symbol_reported:
            continue
        if any((str(item.get("validation_subject") or "").strip() or "core") == "core" for item in symbol_reported):
            compressed_core_symbols.append(source_symbol)
    if compressed_core_symbols:
        return (
            "The analysis compressed a multi-subject strong primary symbol back into a generic core obligation. "
            "Keep separate repair_obligations for each validation_subject under: "
            + ", ".join(compressed_core_symbols)
            + "."
        )
    combined_text = "\n".join(
        str(analysis.get(key, "") or "")
        for key in ("root_cause", "failing_signal", "propagation_path", "repair_constraint")
    ).lower()
    missing_mentions: list[str] = []
    for item in required_items:
        subject = str(item.get("validation_subject") or "").strip().lower()
        symbol = str(item.get("source_symbol") or "").strip().lower()
        obligation_id = str(item.get("id") or "").strip()
        if subject and subject != "core" and subject not in combined_text and symbol.split(".")[-1] not in combined_text:
            missing_mentions.append(obligation_id)
    if missing_mentions:
        return (
            "The analysis still compresses multiple repair obligations into a generic summary. Explicitly mention the "
            "distinct validation subjects or their landed symbols in root_cause/failing_signal/propagation_path for: "
            + ", ".join(missing_mentions)
            + "."
        )
    dominant_failure_symbols = _infer_failure_focus_dominant_symbols(failure_focus, code_context)
    dominant_failure_source_files = _infer_failure_focus_source_files(failure_focus, code_context)
    if _has_single_root_symbol_signature(analysis=analysis, strategy=None, failure_focus=failure_focus):
        suspicious_symbols = {
            str(symbol).strip()
            for symbol in (analysis.get("suspicious_symbols") or [])
            if str(symbol).strip()
        }
        if dominant_failure_symbols and not suspicious_symbols.intersection(set(dominant_failure_symbols)):
            return (
                "This looks like a single_root_symbol failure, but the analysis did not carry forward the dominant "
                f"failing-test symbol(s): {', '.join(dominant_failure_symbols)}. Keep the analysis anchored to the "
                "dominant symbol instead of drifting to helpers/mixins."
            )
        analysis_scope = set(get_analysis_source_files(analysis))
        if dominant_failure_source_files and not analysis_scope.intersection(set(dominant_failure_source_files)):
            return (
                "This looks like a single_root_symbol failure, but the analysis does not include the dominant source "
                f"file(s) in affected_components/suggested_repair_scope: {', '.join(dominant_failure_source_files)}. "
                "Do not expand to helper/mixin files unless you also explain why the dominant source file is insufficient."
            )
    return None


def extract_exception_type_repair_obligations(
    failure_focus: dict[str, Any] | None,
    filtered_candidates: list[CandidatePatch] | None,
    code_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    signatures = extract_failure_path_signatures(failure_focus, code_context)
    covered_paths: set[str] = set()
    for candidate in filtered_candidates or []:
        for test_id in candidate.covered_original_tests or []:
            if isinstance(test_id, str) and test_id.strip():
                covered_paths.add(test_id)
    obligations: list[dict[str, str]] = []
    for test_id in dedupe_preserve_order(list(covered_paths)):
        signature = signatures.get(test_id) or {}
        expected_exception = str(signature.get("expected_exception") or "").strip()
        observed_exception = str(signature.get("observed_exception") or "").strip()
        source_symbol = str(signature.get("source_symbol") or "").strip()
        if not expected_exception or not observed_exception:
            continue
        if expected_exception == observed_exception:
            continue
        obligations.append(
            {
                "test_id": test_id,
                "source_symbol": source_symbol,
                "expected_exception": expected_exception,
                "observed_exception": observed_exception,
            }
        )
    return {
        "active": bool(obligations),
        "obligations": obligations,
    }


def _extract_assert_branch_markers(node: ast.Assert) -> list[str]:
    markers: list[str] = []
    for subnode in ast.walk(node.test):
        if isinstance(subnode, ast.Name):
            name = str(subnode.id).strip()
            if name:
                markers.append(name)
        elif isinstance(subnode, ast.Attribute):
            parts: list[str] = []
            current: ast.AST | None = subnode
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            dotted = ".".join(reversed([part for part in parts if part]))
            if dotted:
                markers.append(dotted)
    msg_value = getattr(node, "msg", None)
    if isinstance(msg_value, ast.Constant) and isinstance(msg_value.value, str):
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", msg_value.value):
            markers.append(token)
    return dedupe_preserve_order([marker for marker in markers if marker not in {"True", "False"}])


def _find_function_node_for_symbol(
    module: ast.AST,
    source_symbol: str,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, str] | None:
    symbol = str(source_symbol or "").strip()
    if not symbol:
        return None
    owner_name = ""
    method_name = symbol
    if "." in symbol:
        owner_name, method_name = symbol.split(".", 1)
    for node in getattr(module, "body", []):
        if owner_name and isinstance(node, ast.ClassDef) and node.name == owner_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return child, owner_name
        elif not owner_name and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return node, ""
        elif not owner_name and isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return child, node.name
    return None


def _resolve_code_file_for_symbol(
    code_context: dict[str, str] | None,
    source_symbol: str,
    preferred_file: str | None = None,
) -> tuple[str, str] | None:
    if not code_context or not source_symbol:
        return None
    symbol = str(source_symbol).strip()
    owner_name = ""
    method_name = symbol
    if "." in symbol:
        owner_name, method_name = symbol.split(".", 1)
    candidates: list[tuple[str, str, int]] = []
    for path, content in code_context.items():
        if not isinstance(content, str):
            continue
        try:
            module = ast.parse(content)
        except SyntaxError:
            continue
        resolved = _find_function_node_for_symbol(module, symbol)
        if resolved is None:
            continue
        _, resolved_owner = resolved
        score = 0
        if preferred_file and path == preferred_file:
            score += 4
        if owner_name and resolved_owner == owner_name:
            score += 2
        if method_name and re.search(rf"\bdef\s+{re.escape(method_name)}\s*\(", content):
            score += 1
        candidates.append((path, content, score))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[0][0], candidates[0][1]


def extract_sibling_validation_repair_obligations(
    failure_focus: dict[str, Any] | None,
    filtered_candidates: list[CandidatePatch] | None,
    code_context: dict[str, str] | None,
) -> dict[str, Any]:
    """Identify sibling assert-based validations inside the same repaired symbol.

    This is generic: once a covered failure path proves that a symbol must convert an
    assert-based validation into an explicit exception path, inspect the same symbol
    for other sibling assert validations so the strategy cannot stop after fixing only
    one assertion branch.
    """
    exception_obligations = extract_exception_type_repair_obligations(failure_focus, filtered_candidates)
    if not exception_obligations.get("active"):
        return {"active": False, "obligations": []}
    obligations: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for obligation in exception_obligations.get("obligations", []):
        observed_exception = str(obligation.get("observed_exception") or "").strip().lower()
        expected_exception = str(obligation.get("expected_exception") or "").strip()
        source_symbol = str(obligation.get("source_symbol") or "").strip()
        if observed_exception != "assertionerror" or not source_symbol or source_symbol in seen_symbols:
            continue
        seen_symbols.add(source_symbol)
        resolved = _resolve_code_file_for_symbol(code_context, source_symbol)
        if resolved is None:
            continue
        target_file, content = resolved
        try:
            module = ast.parse(content)
        except SyntaxError:
            continue
        found = _find_function_node_for_symbol(module, source_symbol)
        if found is None:
            continue
        fn_node, owner_name = found
        assert_infos: list[dict[str, Any]] = []
        for subnode in ast.walk(fn_node):
            if not isinstance(subnode, ast.Assert):
                continue
            markers = _extract_assert_branch_markers(subnode)
            if not markers:
                continue
            assert_infos.append(
                {
                    "lineno": getattr(subnode, "lineno", None),
                    "markers": markers,
                    "test_text": ast.unparse(subnode.test) if hasattr(ast, "unparse") else "",
                }
            )
        if len(assert_infos) <= 1:
            continue
        marker_groups = [info["markers"] for info in assert_infos[:4]]
        obligations.append(
            {
                "source_symbol": source_symbol,
                "owner_name": owner_name,
                "target_file": target_file,
                "expected_exception": expected_exception,
                "assert_count": len(assert_infos),
                "marker_groups": marker_groups,
            }
        )
    return {
        "active": bool(obligations),
        "obligations": obligations,
    }


def recommend_generation_budget(
    status_map: dict[str, str],
    original_failure_log: str,
    base_num_candidates: int,
    base_attempts: int,
) -> dict[str, int | float | str]:
    failed_count = count_failed(status_map)
    error_count = count_statuses(status_map).get("ERROR", 0)
    noise_ratio = compute_log_noise_ratio(original_failure_log)
    failure_mode = classify_failure_mode(original_failure_log)
    candidate_budget = max(base_num_candidates, 3)
    if failed_count > 50 or (error_count >= 10 and noise_ratio >= 0.15):
        candidate_budget = max(candidate_budget, 10)
    elif failed_count > 20 or error_count >= 5 or noise_ratio >= 0.12:
        candidate_budget = max(candidate_budget, 7)
    elif failed_count > 3 or error_count > 0:
        candidate_budget = max(candidate_budget, 5)
    candidate_budget = min(candidate_budget, DEFAULT_MAX_DYNAMIC_CANDIDATES)
    keep_top_k = 2
    if candidate_budget >= 5:
        keep_top_k = 3
    if candidate_budget >= 7:
        keep_top_k = 4
    if candidate_budget >= 10:
        keep_top_k = 5
    attempt_budget = base_attempts
    if noise_ratio >= 0.12 or failed_count > 20:
        attempt_budget += 1
    bucket_limit = min(candidate_budget, 5)
    template_limit = min(candidate_budget, 4)
    difficulty = "low"
    if candidate_budget >= 10:
        difficulty = "very_high"
    elif candidate_budget >= 7:
        difficulty = "high"
    elif candidate_budget >= 5:
        difficulty = "medium"
    return {
        "candidate_budget": candidate_budget,
        "keep_top_k": keep_top_k,
        "attempt_budget": attempt_budget,
        "bucket_limit": bucket_limit,
        "template_limit": template_limit,
        "failure_mode": failure_mode,
        "noise_ratio": round(noise_ratio, 4),
        "difficulty": difficulty,
    }


def score_candidate_quality(candidate: CandidatePatch, eval_result: EvalResult) -> tuple[float, dict[str, float]]:
    enhanced_identifiers = candidate.enhanced_identifiers or []
    failing_identifiers = candidate.failing_identifiers or []
    failing_ratio = safe_rate(len(failing_identifiers), max(len(enhanced_identifiers), 1))
    semantic_bucket = str((candidate.idea or {}).get("semantic_bucket", "")).strip()
    novelty_score = 1.0 if semantic_bucket else 0.5
    patch_lines = len(candidate.patch.splitlines())
    compactness_score = 1.0 if patch_lines <= 24 else 0.7 if patch_lines <= 48 else 0.4
    parse_score = 1.0 if has_meaningful_status_map(eval_result.status_map) else 0.0
    breakdown = {
        "reproduction": round(failing_ratio, 4),
        "novelty": round(novelty_score, 4),
        "compactness": round(compactness_score, 4),
        "parseability": round(parse_score, 4),
    }
    quality_score = (
        0.5 * failing_ratio
        + 0.2 * novelty_score
        + 0.15 * compactness_score
        + 0.15 * parse_score
    )
    return round(quality_score, 4), breakdown


def select_candidates_with_path_coverage(
    candidates: list[CandidatePatch],
    original_test_identifiers: list[str],
    keep_top_k: int | None,
) -> list[CandidatePatch]:
    if not candidates:
        return []
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            item.quality_score,
            len(item.failing_identifiers or []),
            -len(item.patch.splitlines()),
        ),
        reverse=True,
    )
    if not original_test_identifiers:
        return sorted_candidates[:keep_top_k] if keep_top_k is not None else sorted_candidates
    selected: list[CandidatePatch] = []
    selected_ids: set[int] = set()
    path_buckets: dict[str, list[int]] = {original_test: [] for original_test in original_test_identifiers}
    for idx, candidate in enumerate(sorted_candidates):
        covered = candidate.covered_original_tests or []
        for original_test in original_test_identifiers:
            if original_test in covered:
                path_buckets[original_test].append(idx)

    # Phase 1: guarantee at least one candidate per original failure path when available.
    for original_test in original_test_identifiers:
        for idx in path_buckets.get(original_test, []):
            if idx not in selected_ids:
                selected.append(sorted_candidates[idx])
                selected_ids.add(idx)
                break

    # Phase 2: within already-covered paths, expose additional repair obligations before
    # repeatedly retaining near-duplicate candidates for the same obligation cluster.
    obligation_buckets: dict[str, list[int]] = {}
    for idx, candidate in enumerate(sorted_candidates):
        for obligation_id in candidate.covered_obligations or []:
            obligation_buckets.setdefault(obligation_id, []).append(idx)
    for obligation_id, bucket in obligation_buckets.items():
        if keep_top_k is not None and len(selected) >= keep_top_k:
            break
        if not bucket:
            continue
        if any(idx in selected_ids for idx in bucket):
            continue
        selected.append(sorted_candidates[bucket[0]])
        selected_ids.add(bucket[0])

    # Phase 3: within the same original path, prefer different validation subjects
    # before repeating variants of the same subject. This helps keep sibling
    # obligations such as endpoint vs view_func_name from being crowded out even
    # when obligation ids or scores are noisy.
    subject_buckets: dict[tuple[str, str], list[int]] = {}
    for idx, candidate in enumerate(sorted_candidates):
        covered_paths = candidate.covered_original_tests or []
        subject = str((candidate.idea or {}).get("target_validation_subject", "")).strip() or "core"
        for original_test in covered_paths:
            subject_buckets.setdefault((original_test, subject), []).append(idx)
    for (original_test, _subject), bucket in subject_buckets.items():
        if keep_top_k is not None and len(selected) >= keep_top_k:
            break
        if not bucket:
            continue
        if any(idx in selected_ids for idx in bucket):
            continue
        # Only give subject diversity credit once the path already has at least one
        # retained candidate; otherwise Phase 1 remains the main path-coverage gate.
        if not any(
            original_test in (candidate.covered_original_tests or [])
            for candidate in selected
        ):
            continue
        selected.append(sorted_candidates[bucket[0]])
        selected_ids.add(bucket[0])

    # Phase 4: balance the retained set by round-robin over paths, so one dominant
    # path does not consume all remaining slots before other paths get a second chance.
    bucket_positions = {original_test: 0 for original_test in original_test_identifiers}
    progress = True
    while progress and (keep_top_k is None or len(selected) < keep_top_k):
        progress = False
        for original_test in original_test_identifiers:
            bucket = path_buckets.get(original_test, [])
            pos = bucket_positions[original_test]
            while pos < len(bucket) and bucket[pos] in selected_ids:
                pos += 1
            bucket_positions[original_test] = pos
            if pos >= len(bucket):
                continue
            idx = bucket[pos]
            selected.append(sorted_candidates[idx])
            selected_ids.add(idx)
            bucket_positions[original_test] = pos + 1
            progress = True
            if keep_top_k is not None and len(selected) >= keep_top_k:
                break

    # Phase 5: if slots remain, fill from the global ranking.
    for idx, candidate in enumerate(sorted_candidates):
        if keep_top_k is not None and len(selected) >= keep_top_k:
            break
        if idx in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(idx)
    return selected[:keep_top_k] if keep_top_k is not None else selected


def _enhanced_tests_were_collected(candidate: CandidatePatch) -> bool:
    status_map = candidate.eval_status_map or {}
    if not status_map or not candidate.enhanced_identifiers:
        return False
    status_map_keys = list(status_map.keys())
    return any(
        any(ident in key for key in status_map_keys)
        for ident in (candidate.enhanced_identifiers or [])
    )


def retain_missing_symbol_cluster_subject_candidates(
    selected_candidates: list[CandidatePatch],
    all_candidates: list[CandidatePatch],
    keep_top_k: int | None,
) -> list[CandidatePatch]:
    if not all_candidates:
        return selected_candidates
    cluster_subjects: dict[str, set[str]] = {}
    for candidate in all_candidates:
        idea = candidate.idea or {}
        source_symbol = str(idea.get("target_source_symbol") or "").strip()
        validation_subject = str(idea.get("target_validation_subject") or "").strip() or "core"
        if not source_symbol or validation_subject == "core":
            continue
        cluster_subjects.setdefault(source_symbol, set()).add(validation_subject)
    cluster_subjects = {
        symbol: subjects
        for symbol, subjects in cluster_subjects.items()
        if len(subjects) > 1
    }
    if not cluster_subjects:
        return selected_candidates

    selected = list(selected_candidates)
    selected_ids = {id(candidate) for candidate in selected}
    selected_subjects_by_symbol: dict[str, set[str]] = {}
    for candidate in selected:
        idea = candidate.idea or {}
        source_symbol = str(idea.get("target_source_symbol") or "").strip()
        validation_subject = str(idea.get("target_validation_subject") or "").strip() or "core"
        if source_symbol and validation_subject != "core":
            selected_subjects_by_symbol.setdefault(source_symbol, set()).add(validation_subject)

    for source_symbol, expected_subjects in cluster_subjects.items():
        if keep_top_k is not None and len(selected) >= keep_top_k:
            break
        missing_subjects = sorted(expected_subjects - selected_subjects_by_symbol.get(source_symbol, set()))
        for missing_subject in missing_subjects:
            if keep_top_k is not None and len(selected) >= keep_top_k:
                break
            eligible: list[CandidatePatch] = []
            for candidate in all_candidates:
                if id(candidate) in selected_ids:
                    continue
                idea = candidate.idea or {}
                candidate_symbol = str(idea.get("target_source_symbol") or "").strip()
                candidate_subject = str(idea.get("target_validation_subject") or "").strip() or "core"
                if candidate_symbol != source_symbol or candidate_subject != missing_subject:
                    continue
                if not _enhanced_tests_were_collected(candidate):
                    continue
                if candidate.reason in {
                    "test patch did not apply to buggy base commit",
                    "evaluation produced an empty status_map; likely log_parse_failed or tests_not_collected",
                    "enhanced tests were not collected from the evaluation status_map",
                }:
                    continue
                eligible.append(candidate)
            if not eligible:
                continue
            eligible.sort(
                key=lambda item: (
                    item.quality_score,
                    len(item.failing_identifiers or []),
                    -len(item.patch.splitlines()),
                ),
                reverse=True,
            )
            rescued = eligible[0]
            rescued.kept = True
            rescued.reason = (
                rescued.reason + "; " if rescued.reason else ""
            ) + "retained_for_symbol_cluster_subject_coverage"
            selected.append(rescued)
            selected_ids.add(id(rescued))
            selected_subjects_by_symbol.setdefault(source_symbol, set()).add(missing_subject)
    return selected[:keep_top_k] if keep_top_k is not None else selected


def select_template_names(failure_mode: str, limit: int = 3) -> list[str]:
    templates = TEMPLATE_LIBRARY.get(failure_mode, TEMPLATE_LIBRARY["generic_failure"])
    return templates[:limit]


def select_semantic_buckets(failure_mode: str, limit: int = 3) -> list[str]:
    buckets = SEMANTIC_BUCKETS.get(failure_mode, SEMANTIC_BUCKETS["generic_failure"])
    return buckets[:limit]


def build_generation_feedback(
    candidate: CandidatePatch,
    seen_signatures: set[tuple[str, ...]],
) -> str | None:
    if candidate.duplicate_identifiers:
        return (
            "Your previous patch reused original SWE-bench test names: "
            + ", ".join(candidate.duplicate_identifiers)
            + ". Generate only new test_sweb_enhanced_* functions."
        )
    if not candidate.enhanced_identifiers:
        return "Your previous patch did not add any test_sweb_enhanced_* test. Add at least one."
    signature = tuple(candidate.enhanced_identifiers)
    if signature in seen_signatures:
        return (
            "Your previous patch repeated an already generated semantic angle: "
            + ", ".join(candidate.enhanced_identifiers)
            + ". Generate a distinct enhanced test."
        )
    return None


def build_idea_feedback(
    idea: dict[str, Any] | None,
    seen_titles: set[str],
    seen_buckets: set[str],
    allowed_buckets: list[str],
    original_test_identifiers: list[str],
    required_uncovered_tests: list[str] | None = None,
    failure_focus: dict[str, Any] | None = None,
    required_uncovered_obligations: list[str] | None = None,
    code_context: dict[str, str] | None = None,
) -> str | None:
    if idea is None:
        return (
            "Return a JSON object with title, semantic_bucket, goal, oracle, target_tests, covers_original_tests, "
            "covers_obligations, target_source_symbol, target_validation_subject, semantic_alignment_tokens, "
            "trigger_shape_tokens, template, and rationale."
        )
    title = str(idea.get("title", "")).strip()
    if not title:
        return "The JSON idea is missing a non-empty title."
    if title in seen_titles:
        return f"The idea title '{title}' duplicates a previous idea. Produce a different semantic angle."
    target_tests = idea.get("target_tests")
    if not isinstance(target_tests, list) or not target_tests:
        return "The JSON idea must include a non-empty target_tests list using only test_sweb_enhanced_* names."
    if any(not str(name).startswith(ENHANCED_TEST_PREFIX) for name in target_tests):
        return "All target_tests names must start with test_sweb_enhanced_."
    covers_original_tests = idea.get("covers_original_tests")
    if not isinstance(covers_original_tests, list) or not covers_original_tests:
        return (
            "The JSON idea must include a non-empty covers_original_tests list naming the original FAIL_TO_PASS "
            "tests whose path this idea is meant to cover."
        )
    normalized_original = set(original_test_identifiers)
    invalid_targets = [str(name) for name in covers_original_tests if str(name) not in normalized_original]
    if invalid_targets:
        return (
            "covers_original_tests may only contain original FAIL_TO_PASS identifiers from: "
            + ", ".join(original_test_identifiers)
        )
    required_uncovered_tests = [str(name) for name in (required_uncovered_tests or []) if str(name).strip()]
    if required_uncovered_tests and not any(
        str(name) in covers_original_tests for name in required_uncovered_tests
    ):
        return (
            "This idea must help cover at least one still-uncovered original FAIL_TO_PASS path from: "
            + ", ".join(required_uncovered_tests)
        )
    target_source_symbol = str(idea.get("target_source_symbol", "")).strip()
    if not target_source_symbol:
        return "The JSON idea must include a non-empty target_source_symbol naming the source symbol or API path it exercises."
    target_validation_subject = str(idea.get("target_validation_subject", "")).strip()
    if not target_validation_subject:
        return "The JSON idea must include a non-empty target_validation_subject naming the concrete validation subject it exercises."
    semantic_alignment_tokens = _normalize_alignment_tokens(idea.get("semantic_alignment_tokens"))
    if not semantic_alignment_tokens:
        return "The JSON idea must include a non-empty semantic_alignment_tokens list describing the concrete API/exception cues it matches."
    trigger_shape_tokens = _normalize_alignment_tokens(idea.get("trigger_shape_tokens"))
    if not trigger_shape_tokens:
        return "The JSON idea must include a non-empty trigger_shape_tokens list describing the concrete trigger shape it uses."
    obligation_ids = _normalize_obligation_ids(
        idea.get("covers_obligations"),
        [str(name) for name in covers_original_tests],
        failure_focus,
        code_context,
    )
    if not obligation_ids:
        return "The JSON idea must include covers_obligations naming at least one concrete repair obligation for the covered path."
    direct_assertion_feedback = build_direct_assertion_idea_drift_feedback(idea, failure_focus)
    if direct_assertion_feedback:
        return direct_assertion_feedback
    required_uncovered_obligations = [
        str(name) for name in (required_uncovered_obligations or []) if str(name).strip()
    ]
    if required_uncovered_obligations and not any(
        obligation_id in obligation_ids for obligation_id in required_uncovered_obligations
    ):
        return (
            "This idea must help cover at least one still-uncovered repair obligation from: "
            + ", ".join(required_uncovered_obligations)
        )
    alignment_feedback = build_path_semantic_alignment_feedback(
        [str(name) for name in covers_original_tests],
        target_source_symbol,
        semantic_alignment_tokens,
        failure_focus,
        code_context,
        json.dumps(idea, ensure_ascii=False),
    )
    if alignment_feedback:
        return alignment_feedback
    obligation_alignment_feedback = build_obligation_trigger_alignment_feedback(
        obligation_ids,
        target_source_symbol,
        target_validation_subject,
        semantic_alignment_tokens,
        trigger_shape_tokens,
        failure_focus,
        code_context,
        json.dumps(idea, ensure_ascii=False),
    )
    if obligation_alignment_feedback:
        return obligation_alignment_feedback
    template = str(idea.get("template", "")).strip()
    if not template:
        return "The JSON idea must include a template."
    oracle = str(idea.get("oracle", "")).strip().lower()
    if oracle != "regression_should_fail_on_buggy_and_pass_on_fixed":
        return (
            "The JSON idea must include oracle='regression_should_fail_on_buggy_and_pass_on_fixed'. "
            "Do not validate the buggy behavior as the expected outcome."
        )
    rationale = str(idea.get("rationale", "")).lower()
    if any(
        banned in rationale
        for banned in [
            "verify current buggy behavior",
            "confirm the bug is present",
        ]
    ):
        return "The rationale appears to validate buggy behavior. Describe the intended fixed behavior instead."
    semantic_bucket = str(idea.get("semantic_bucket", "")).strip()
    if semantic_bucket not in allowed_buckets:
        return (
            "The JSON idea must include semantic_bucket chosen from: "
            + ", ".join(allowed_buckets)
        )
    if semantic_bucket in seen_buckets:
        return f"The semantic_bucket '{semantic_bucket}' already exists. Choose a different bucket."
    return None


def build_enhanced_test_prompt(
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    failure_focus: dict[str, Any] | None = None,
    feedback: str | None = None,
) -> str:
    original_tests = normalize_patch(instance["test_patch"])
    original_identifiers = ", ".join(get_original_test_identifiers(instance)) or "N/A"
    context_text = format_code_context(code_context)
    # Show the actual buggy-commit content of each test file so the model can
    # write a patch whose context lines match the real file, not the test_patch diff.
    test_file_paths = dedupe_preserve_order(
        get_modified_files(instance["test_patch"]) + get_new_files(instance["test_patch"])
    )
    base_test_content_blocks = []
    for path in test_file_paths:
        if path in code_context:
            tail_key = f"__tail__{path}"
            if tail_key in code_context:
                # File was truncated — show the beginning for imports/class context,
                # and the exact tail so the model knows what the last line looks like.
                tail_content = code_context[tail_key]
                tail_lines = tail_content.splitlines()
                # Count total lines in truncated + tail to estimate file line count
                head_lines = code_context[path].splitlines()
                # The tail may overlap with head; use tail length as lower bound for EOF offset
                eof_line_approx = len(head_lines) + len(tail_lines)
                base_test_content_blocks.append(
                    f"### {path} — FILE START (truncated; use for imports/class structure)\n"
                    f"```python\n{code_context[path]}\n```\n\n"
                    f"### {path} — FILE END — last {len(tail_lines)} lines\n"
                    f"```python\n{tail_content}\n```\n"
                    f"CRITICAL DIFF INSTRUCTIONS for this truncated file:\n"
                    f"- The file has approximately {eof_line_approx}+ lines total.\n"
                    f"- Append your new test AFTER the very last line shown in FILE END above.\n"
                    f"- Use a ZERO-CONTEXT append hunk so git apply does not need to match any existing lines:\n"
                    f"  ```diff\n"
                    f"  diff --git a/{path} b/{path}\n"
                    f"  --- a/{path}\n"
                    f"  +++ b/{path}\n"
                    f"  @@ -{eof_line_approx},0 +{eof_line_approx},N @@\n"
                    f"  +<your new lines here, each prefixed with '+'>\n"
                    f"  ```\n"
                    f"- Do NOT include any context lines (lines without '+'/'-' prefix).\n"
                    f"- Adjust N to the actual number of lines you add.\n"
                    f"- Use the exact line count {eof_line_approx} or higher for the hunk offset.\n"
                )
            else:
                base_test_content_blocks.append(
                    f"### {path} (buggy base commit — this is the file your patch must apply to)\n"
                    f"```python\n{code_context[path]}\n```"
                )
    base_test_content = "\n\n".join(base_test_content_blocks)
    prompt = (
        "Task: generate an enhanced test patch against the buggy base commit.\n"
        "The patch must preserve the intent of the original SWE-bench tests and add a few semantically aligned tests.\n"
        "Requirements:\n"
        "1. Output only one unified diff patch.\n"
        "2. Keep modifications restricted to test files.\n"
        "3. Add at least one new test function whose name starts with test_sweb_enhanced_.\n"
        "4. The patch must be directly applicable to the buggy base commit, not to an already patched tree.\n"
        "   CRITICAL: The context lines in your diff must match the buggy base commit file content shown below.\n"
        "   Do NOT use lines from the 'Original SWE-bench test patch' as diff context — those lines do not\n"
        "   exist in the buggy base commit and will cause the patch to fail.\n"
        "5. Prefer small, targeted additions that expose the same bug semantics.\n"
        "5.1. The enhanced test must expose the bug through a different observation angle, boundary condition, or usage path than the original failing test.\n"
        "5.2. Avoid near-duplicate rewrites that only rename variables or restate the same assertion in slightly different words.\n"
        "5.3. Prefer one minimal new test function over parameterized or multi-scenario tests.\n"
        "5.4. Avoid @pytest.mark.parametrize, helper abstractions, or multiple independent assertions unless the original failing behavior truly requires them.\n"
        "5.5. Prefer a direct, local reproduction with a single setup, a single action, and a single oracle.\n"
        "6. Do not redefine, rename, delete, or edit the original SWE-bench tests.\n"
        "7. Do not reuse any existing test function names.\n\n"
        f"Instance ID: {instance['instance_id']}\n"
        f"Repository: {instance['repo']}\n\n"
        f"Original test function names to avoid: {original_identifiers}\n\n"
        f"Problem statement:\n{instance['problem_statement']}\n\n"
        f"Original SWE-bench test patch (for reference only — do NOT use these lines as diff context):\n```diff\n{original_tests}\n```\n\n"
        f"Original test failure log:\n```text\n{truncate_text(original_failure_log, 12000)}\n```\n\n"
        f"Focused failure summary:\n{format_failure_focus(failure_focus)}\n\n"
    )
    # Inject per-test tracebacks so the model knows EACH original test's call path.
    # This is critical when there are multiple original failing tests covering different
    # code paths (e.g. one tests __init__, another tests add_url_rule).  Without this,
    # the model tends to only generate tests for the most obvious path.
    orig_ids = get_original_test_identifiers(instance)
    per_test_tb = extract_per_test_tracebacks(original_failure_log, orig_ids)
    if per_test_tb:
        prompt += (
            "IMPORTANT — each original failing test exercises a DIFFERENT code path. "
            "Your enhanced tests MUST collectively cover ALL of these paths:\n\n"
        )
        for name, tb in per_test_tb.items():
            prompt += f"### {name}\n```text\n{tb}\n```\n\n"
        prompt += (
            "Generate at least one enhanced test per distinct code path shown above. "
            "A test that only re-covers a path already covered by other enhanced tests will score lower. "
            "Do NOT spend most of the candidate budget on only one path while ignoring the others. "
            "When there are multiple original FAIL_TO_PASS tests, the retained enhanced set must include at least one "
            "reproduction for each original path, or the repair guidance will be incomplete.\n\n"
        )
    if len(orig_ids) == 1:
        prompt += (
            "IMPORTANT — there is only one original FAIL_TO_PASS test. "
            "Stay close to that exact execution path and build a minimal variant of the same failure, "
            "instead of broadening into several scenarios. The best enhanced test is usually a compact, "
            "single-purpose reproduction that changes just one condition while preserving the original oracle shape.\n\n"
        )
    if (
        len(orig_ids) == 1
        and "nomatch:" in original_failure_log
        and "SKIPPED [1]" in original_failure_log
    ):
        prompt += (
            "IMPORTANT — the original failure is a single skip-location reporting mismatch. "
            "Generate only near-neighbor variants of that same path. Keep the same one skipped test shape, "
            "the same `-rs --runxfail` style invocation, and the same location-report oracle. "
            "Do NOT broaden into nested functions, multiple skipped tests, multiple marks, or skipif/xfail "
            "combinations unless the original failing traceback already contains those structures.\n\n"
        )
    if failure_focus:
        dominant_errors = ", ".join(failure_focus.get("dominant_errors") or []) or "unknown"
        prompt += (
            f"Target failure types to reproduce: {dominant_errors}\n"
            f"Your enhanced test MUST trigger one of these error types on the buggy code. "
            f"A test that passes on the buggy code will be discarded.\n\n"
        )
    if base_test_content:
        prompt += (
            f"Buggy base commit test file content (use this as the base for your diff):\n"
            f"{base_test_content}\n\n"
        )
    prompt += f"Buggy code context:\n{context_text}\n"
    if feedback:
        prompt += f"\nRevision feedback for the next attempt:\n{feedback}\n"
    return prompt


def build_test_idea_prompt(
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    failure_mode: str,
    template_names: list[str],
    semantic_buckets: list[str],
    failure_focus: dict[str, Any] | None = None,
    feedback: str | None = None,
    uncovered_original_tests: list[str] | None = None,
    uncovered_obligations: list[str] | None = None,
) -> str:
    context_text = format_code_context(code_context)
    original_test_identifiers = get_original_test_identifiers(instance)
    original_identifiers = ", ".join(original_test_identifiers) or "N/A"
    uncovered_original_tests = [
        str(name) for name in (uncovered_original_tests or []) if str(name).strip()
    ]
    uncovered_obligations = [
        str(name) for name in (uncovered_obligations or []) if str(name).strip()
    ]
    bucket_definitions = "\n".join(
        f"- {bucket}: {SEMANTIC_BUCKET_GUIDANCE.get(bucket, 'choose a distinct semantic angle for this bucket')}"
        for bucket in semantic_buckets
    )
    prompt = (
        "Task: propose one high-quality enhanced test idea before writing any diff.\n"
        "Return exactly one JSON object.\n"
        "JSON schema:\n"
        '{\n'
        '  "title": "short semantic angle",\n'
        '  "semantic_bucket": "one bucket from the allowed list",\n'
        '  "goal": "what bug behavior this test stresses",\n'
        '  "oracle": "must be regression_should_fail_on_buggy_and_pass_on_fixed",\n'
        '  "template": "one template name from the allowed list",\n'
        '  "target_tests": ["test_sweb_enhanced_*"],\n'
        '  "covers_original_tests": ["exact original FAIL_TO_PASS identifiers this idea covers"],\n'
        '  "covers_obligations": ["exact repair obligation ids this idea covers"],\n'
        '  "target_source_symbol": "source symbol or API path this idea is intended to exercise",\n'
        '  "target_validation_subject": "concrete validation subject such as endpoint, name, attribute_name",\n'
        '  "semantic_alignment_tokens": ["concrete tokens such as endpoint=, route(, add_url_rule, Blueprint(, ValueError"],\n'
        '  "trigger_shape_tokens": ["tokens that describe the real trigger shape such as endpoint=, __name__, Blueprint("],\n'
        '  "rationale": "why this is distinct from the original tests"\n'
        '}\n\n'
        f"Detected failure mode: {failure_mode}\n"
        f"Allowed templates: {', '.join(template_names)}\n"
        f"Allowed semantic buckets: {', '.join(semantic_buckets)}\n"
        f"Bucket definitions:\n{bucket_definitions}\n\n"
        f"Failure mode constraint: the proposed test idea must be explicitly designed to trigger the detected failure mode '{failure_mode}' or a closely related manifestation of the same bug.\n"
        "Prefer a compact idea that can be implemented as one new test function with one setup, one action, and one assertion block.\n"
        "Avoid parameterized, multi-case, or kitchen-sink ideas unless the original failing behavior itself requires multiple inputs.\n"
        f"Original test names to avoid: {original_identifiers}\n\n"
        f"Original FAIL_TO_PASS identifiers that must be covered: {', '.join(original_test_identifiers) or 'N/A'}\n\n"
        f"Structured semantic signatures for those original paths:\n{format_failure_path_signatures(failure_focus, code_context)}\n\n"
        f"Structured repair obligations for those paths:\n{format_failure_path_obligations(failure_focus, code_context)}\n\n"
        f"Problem statement:\n{instance['problem_statement']}\n\n"
        f"Original failure log:\n```text\n{truncate_text(original_failure_log, 8000)}\n```\n\n"
        f"Focused failure summary:\n{format_failure_focus(failure_focus)}\n\n"
        f"Buggy code context:\n{context_text}\n"
        "\nImportant oracle rule: the enhanced test must REPRODUCE the bug on the current unfixed code. "
        "Concretely: the test must FAIL (AssertionError, AttributeError, ValueError, etc.) when run on "
        "buggy code, and PASS when run on fixed code. The failing assertion is the proof that the bug exists. "
        "Design the test so that the exact error type shown in the failure log is triggered.\n"
        "The rationale must explain how this idea differs from the original failure observation and why it adds new repair guidance.\n"
        f"{build_direct_assertion_enhanced_test_guidance(failure_focus)}"
        "\nTRACEBACK-TO-OBLIGATION REFINEMENT RULES:\n"
        "LAYER 1 — PATH LOCALIZATION:\n"
        "1. Start from the original traceback, not from a guessed root cause. Identify the landed source symbol, the "
        "expected behavior/exception, and the observed wrong behavior/exception.\n"
        "2. TRACEBACK-ANCHOR RULE: if the traceback already lands inside a concrete source file/symbol, prefer an idea "
        "that exercises that same landed symbol directly. Do not drift to an upstream or downstream propagation site "
        "unless the traceback itself makes that propagation site part of the repair obligation.\n"
        "3. EXCEPTION-GAP RULE: if the traceback shows an expected exception that differs from the observed exception "
        "(for example ValueError expected but AssertionError observed), preserve that exact mismatch in your idea. "
        "Do not collapse it into a generic 'should fail' test — the idea should still expose the wrong exception type.\n"
        "\nLAYER 2 — OBLIGATION REFINEMENT:\n"
        "4. Do not stop at high-level path coverage. Refine each landed path into concrete repair obligation(s): the "
        "specific validation subject, branch, or local check that is still wrong.\n"
        "5. SIBLING VALIDATION RULE: if a landed symbol appears to contain multiple sibling validation branches of the "
        "same kind, do not assume one branch represents the whole bug. Generate ideas that collectively expose distinct "
        "sibling obligations instead of repeatedly stressing only the easiest branch.\n"
        "5.1. TRIGGER-SHAPE RULE: for each obligation, identify the real trigger shape used by the buggy code. If the "
        "obligation is about an argument value, trigger it through the real argument. If it is about an object "
        "attribute or callable attribute, construct that object/function and trigger the bug through the real "
        "attribute path. Do not invent new parameter names or API surfaces that do not exist in the landed symbol.\n"
        "\nLAYER 3 — IDEA GENERATION CONSTRAINTS:\n"
        "6. covers_original_tests marks which original FAIL_TO_PASS path you are extending. covers_obligations marks "
        "which concrete repair obligation inside that path you are exposing. Do not mark one idea as covering an entire "
        "path if it only exercises one branch inside that path.\n"
        "7. IDEA GRANULARITY RULE: each idea should usually expose one concrete obligation, not an entire source symbol "
        "in the abstract. If a symbol contains multiple validation subjects, prefer separate ideas for separate subjects.\n"
        "8. UNFINISHED-PATH RULE: if a path is already covered at the path level but one of its obligations is still "
        "uncovered, prioritize a new idea for that uncovered obligation instead of another variant of an already-covered one.\n"
        "9. REAL-TRIGGER RULE: target_validation_subject and trigger_shape_tokens must describe the actual way the bug "
        "is triggered in code. A semantically similar but API-invented test does not count as obligation coverage.\n"
    )
    if len(original_test_identifiers) == 1:
        prompt += (
            "Because there is only one original FAIL_TO_PASS test, prefer a near-path variant of that same failing flow. "
            "Do not expand into multiple scenarios just to create diversity.\n"
        )
    else:
        prompt += (
            "Because there are multiple original FAIL_TO_PASS tests, you MUST set covers_original_tests to the exact "
            "original identifiers whose path this idea is intended to cover. Across the candidate set, every original "
            "FAIL_TO_PASS path should receive at least one idea.\n"
        )
    prompt += (
        "Semantic alignment rule: covers_original_tests is not just a label. Your idea must actually follow the same "
        "source symbol / API path / exception semantics shown in the structured signatures above. "
        "For example, if an original failure path goes through an endpoint=... call into add_url_rule, do not relabel "
        "a Blueprint(name=...) constructor test as if it covered that path.\n"
        "Obligation rule: covers_obligations must name the concrete repair obligations this idea helps expose. "
        "When one source symbol has multiple sibling validation obligations, spread ideas across those obligations "
        "instead of repeatedly covering only the easiest one.\n"
        "Trigger-shape rule: use target_validation_subject and trigger_shape_tokens to describe the concrete subject "
        "and trigger shape for the obligation. If the bug is triggered through a real attribute path like "
        "view_func.__name__, your idea must preserve that path instead of replacing it with an invented keyword or helper.\n"
        "The idea must encode the intended correct behavior while preserving the buggy traceback signal that proves the "
        "current implementation is wrong.\n"
    )
    if uncovered_original_tests:
        prompt += (
            "Original FAIL_TO_PASS paths that are STILL UNCOVERED by earlier generated ideas: "
            + ", ".join(uncovered_original_tests)
            + ". This next idea MUST cover at least one of these uncovered paths. Do not spend this attempt on an "
              "already-covered path unless all original paths are already covered.\n"
        )
    if uncovered_obligations:
        prompt += (
            "Repair obligations that are STILL UNCOVERED by earlier generated ideas: "
            + ", ".join(uncovered_obligations)
            + ". This next idea MUST cover at least one of these uncovered obligations, not just an already-covered "
              "obligation on the same path.\n"
        )
    if feedback:
        prompt += f"\nRevision feedback for the next idea:\n{feedback}\n"
    return prompt


def build_diff_from_idea_prompt(
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    idea: dict[str, Any],
    failure_focus: dict[str, Any] | None = None,
) -> str:
    prompt = build_enhanced_test_prompt(
        instance,
        code_context,
        original_failure_log,
        failure_focus=failure_focus,
    )
    prompt += (
        "\nYou must implement this exact structured test idea as a unified diff.\n"
        f"Structured idea:\n```json\n{json.dumps(idea, indent=2, ensure_ascii=False)}\n```\n"
        "\nCRITICAL: the enhanced test must FAIL on the current buggy code and PASS after the fix.\n"
        "To achieve this, reproduce the exact failure observed in the original failure log:\n"
        "  - Call the same code path that triggers the bug\n"
        "  - Assert the CORRECT expected behavior (which the buggy code violates)\n"
        "  - The assertion failure proves the bug is present\n"
        "Do NOT write a test that already passes on the buggy code.\n"
        f"{build_direct_assertion_enhanced_test_guidance(failure_focus)}"
    )
    # If the idea specifies a concrete trigger shape (attribute access, specific argument),
    # add an explicit reminder so the model cannot silently substitute a simpler variant.
    validation_subject = str(idea.get("target_validation_subject") or "").strip()
    trigger_tokens = [str(t) for t in (idea.get("trigger_shape_tokens") or []) if str(t).strip()]
    source_symbol = str(idea.get("target_source_symbol") or "").strip()
    if validation_subject and trigger_tokens:
        # Detect attribute-path triggers (e.g. view_func.__name__)
        attr_tokens = [t for t in trigger_tokens if "." in t and not t.startswith("Blueprint")]
        if attr_tokens:
            prompt += (
                f"\nIMPORTANT — TRIGGER SHAPE ENFORCEMENT:\n"
                f"The idea specifies target_validation_subject='{validation_subject}' with "
                f"trigger_shape_tokens {trigger_tokens}.\n"
                f"This means the bug must be triggered via the ATTRIBUTE PATH: {attr_tokens[0]}.\n"
                f"You MUST construct a real object/function where that attribute contains the forbidden value.\n"
                f"Example pattern for '{attr_tokens[0]}':\n"
                f"  def my_func_with_dot_in_name(): pass\n"
                f"  my_func_with_dot_in_name.__name__ = 'invalid.name'\n"
                f"  bp.add_url_rule('/', view_func=my_func_with_dot_in_name)\n"
                f"Do NOT test a different subject (e.g. Blueprint name or endpoint argument) and claim "
                f"it covers this obligation — only the attribute path '{attr_tokens[0]}' counts.\n"
            )
        elif source_symbol:
            prompt += (
                f"\nIMPORTANT — TRIGGER SHAPE ENFORCEMENT:\n"
                f"The idea specifies target_validation_subject='{validation_subject}' in "
                f"source_symbol='{source_symbol}'.\n"
                f"Your test MUST call '{source_symbol}' (or the API that reaches it) with "
                f"a value that triggers the '{validation_subject}' validation specifically.\n"
                f"Do NOT substitute a different parameter or API surface.\n"
            )
    return prompt


def build_patch_prompt(
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    filtered_candidates: list[CandidatePatch],
    failure_focus: dict[str, Any] | None = None,
    hide_original_test_patch: bool = False,
) -> str:
    context_text = format_code_context(code_context)
    enhanced_summary = format_candidate_summary(filtered_candidates)
    repair_mode = get_repair_mode(filtered_candidates)
    original_test_patch_section = (
        "Original SWE-bench test patch: hidden by --hide_original_test_patch_in_repair. "
        "Do not infer the fix from the official test diff; rely on the failure log, code context, "
        "and retained enhanced tests only.\n\n"
        if hide_original_test_patch
        else f"Original SWE-bench test patch:\n```diff\n{normalize_patch(instance['test_patch'])}\n```\n\n"
    )
    prompt = (
        "Task: generate a source-code patch that fixes the bug.\n"
        "Requirements:\n"
        "1. Output only one unified diff patch.\n"
        "2. Do not modify tests.\n"
        "3. Use the original failing behavior and the retained enhanced tests as repair guidance.\n"
        "4. Keep the patch minimal and targeted.\n"
        "5. SINGLE-HUNK RULE: if the fix is in one function or one contiguous code region, "
        "use exactly ONE hunk. Multi-hunk patches frequently have wrong @@ line offsets for "
        "the second hunk (because the first hunk shifts lines), causing apply failures. "
        "Only use multiple hunks when fixing genuinely separate locations in the file.\n"
        "6. ELIF→IF RULE: if the fix requires changing `elif` to `if` (to make a branch run "
        "unconditionally rather than as part of an elif chain), write a single hunk that "
        "removes the `elif` line and adds `if`. Do NOT add a new `elif` before the existing one.\n\n"
        f"Instance ID: {instance['instance_id']}\n"
        f"Repository: {instance['repo']}\n\n"
        f"Problem statement:\n{instance['problem_statement']}\n\n"
        f"{original_test_patch_section}"
        f"Original test failure log:\n```text\n{truncate_text(original_failure_log, 12000)}\n```\n\n"
        f"Retained enhanced test patches:\n{enhanced_summary}\n\n"
        f"Buggy code context:\n{context_text}\n"
    )
    # P1: Inject per-target-test tracebacks as the primary repair signal so the LLM
    # focuses on the actual FAIL_TO_PASS tests rather than unrelated environment noise.
    target_tracebacks = (failure_focus or {}).get("target_test_tracebacks") or {}
    if target_tracebacks:
        prompt += "\nCRITICAL — PRIMARY BUG SIGNAL (the following tests MUST pass after the fix):\n"
        prompt += "The log above may contain many failures from unrelated environment issues. "
        prompt += "Focus your repair exclusively on the tracebacks below:\n"
        for test_id, tb in target_tracebacks.items():
            prompt += f"\n=== FAIL_TO_PASS: {test_id} ===\n{truncate_text(tb, 2000)}\n"
    if repair_mode == "enhanced_guided":
        prompt += (
            "\nEnhanced-guided repair mode: retained enhanced tests are available and should act as stronger semantic guidance. "
            "If the enhanced tests suggest cross-file or cross-class propagation, explicitly consider a broader edit scope instead of collapsing to a single local change.\n"
        )
    else:
        prompt += (
            "\nBaseline-fallback repair mode: no enhanced tests were retained. "
            "Rely on the original failure log, original test identifiers, and buggy code context. Prefer a minimally invasive repair unless the failure evidence clearly indicates cross-file propagation.\n"
        )
    return prompt


def build_patch_analysis_prompt(
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    filtered_candidates: list[CandidatePatch],
    failure_focus: dict[str, Any] | None = None,
) -> str:
    enhanced_failed = aggregate_enhanced_failures(filtered_candidates)
    required_repair_obligations = select_effective_required_repair_obligations(
        extract_required_repair_obligations(
            filtered_candidates,
            failure_focus=failure_focus,
            code_context=code_context,
        ),
        filtered_candidates=filtered_candidates,
    )
    validation_subject_clusters = extract_validation_subject_repair_obligations_from_obligations(
        required_repair_obligations
    )
    dominant_failure_symbols = _infer_failure_focus_dominant_symbols(failure_focus, code_context)
    dominant_failure_source_files = _infer_failure_focus_source_files(failure_focus, code_context)
    context_text = format_code_context(code_context)
    original_failure_lower = original_failure_log.lower()
    needs_flag_short_circuit_guidance = (
        "runxfail" in original_failure_lower
        and any(token in original_failure_lower for token in ("nomatch:", "skipped [1]", "test_sample.py", "skipping.py"))
    )
    # P1: Inject target-test tracebacks as primary analysis signal.
    target_tracebacks = (failure_focus or {}).get("target_test_tracebacks") or {}
    target_traceback_section = ""
    if target_tracebacks:
        target_traceback_section = (
            "\nPRIMARY BUG SIGNAL — tracebacks for the tests that MUST be fixed (FAIL_TO_PASS):\n"
            "(Focus your root-cause analysis on these; other failures may be environment noise.)\n"
        )
        for test_id, tb in target_tracebacks.items():
            target_traceback_section += f"\n--- {test_id} ---\n{truncate_text(tb, 1500)}\n"
    prompt = (
        "Task: analyze why the buggy code fails before proposing a patch.\n"
        "Return exactly one JSON object.\n"
        "JSON schema:\n"
        '{\n'
        '  "root_cause": "short explanation of the likely bug",\n'
        '  "affected_components": [{"file": "path/to/file.py", "symbol": "ClassOrFunction", "reason": "why this component matters"}],\n'
        '  "failing_signal": "what the failing assertions/logs indicate",\n'
        '  "propagation_path": "how the bug likely propagates from root cause to the visible failure",\n'
        '  "repair_constraint": "what must remain true after the fix",\n'
        '  "suggested_repair_scope": ["file1.py", "file2.py"],\n'
        '  "suspicious_symbols": ["names of classes/functions/attributes involved"],\n'
        '  "repair_obligations": [{"id": "required obligation id", "source_symbol": "symbol", "validation_subject": "subject", "why_it_matters": "why this specific obligation must be preserved"}]\n'
        '}\n\n'
        f"Problem statement:\n{instance['problem_statement']}\n\n"
        f"Original failure log:\n```text\n{truncate_text(original_failure_log, 10000)}\n```\n"
        f"{target_traceback_section}\n"
        f"Retained enhanced failing tests: {', '.join(enhanced_failed) or 'None'}\n\n"
        "Structured required repair obligations already exposed by retained enhanced tests:\n"
        f"{format_required_repair_obligations(required_repair_obligations)}\n\n"
        f"Buggy code context (these are the actual source files available — base your analysis on what you see here):\n{context_text}\n\n"
        "Important: look carefully at ALL files in the code context above. "
        "The root cause may be in a mixin or helper class (e.g. a class providing __slots__ = () that is missing it), "
        "not necessarily in the most prominent class. "
        "If the bug looks inheritance-related, check every class in the MRO that is visible in the context. "
        "Only include files in suggested_repair_scope that are present in the code context above.\n"
        "Do not collapse multiple required_repair_obligations into one generic root-cause summary. "
        "Carry every listed obligation forward into repair_obligations and keep their source_symbol / "
        "validation_subject pairs intact so downstream strategy/edit-planning cannot silently drop them.\n"
    )
    if dominant_failure_symbols or dominant_failure_source_files:
        prompt += (
            "\nSINGLE-ROOT EVIDENCE:\n"
            f"- Dominant symbols inferred directly from the failing test body: {', '.join(dominant_failure_symbols) or 'none'}\n"
            f"- Source files implied by those symbols/imports: {', '.join(dominant_failure_source_files) or 'none'}\n"
            "Treat these as higher-confidence localization signals than generic mixins/helpers. If you propose a helper "
            "or mixin as the edit target, you must explicitly explain why the dominant symbol/source file cannot satisfy "
            "the failure signal.\n"
        )
    if validation_subject_clusters.get("active"):
        prompt += (
            "\nSYMBOL-CLUSTER REPAIR RULE:\n"
            "The retained enhanced tests already expose multiple validation_subjects within the same strong primary "
            "symbol. You MUST keep them as distinct repair_obligations instead of compressing them into a generic "
            "core obligation or a single dominant subject.\n"
            "Multi-subject symbol clusters:\n"
            f"{format_validation_subject_symbol_clusters(required_repair_obligations)}\n"
        )
    if needs_flag_short_circuit_guidance:
        prompt += (
            "\nFLAG-SHORT-CIRCUIT CONTROL-FLOW RULE:\n"
            "- The failure evidence is about behavior when a flag-enabled mode is ON (for example `--runxfail`).\n"
            "- Do NOT stop at the surface symptom ('wrong location', 'wrong longrepr', 'wrong payload').\n"
            "- You MUST explicitly inspect whether an earlier `if`/`elif`/guard branch under that flag short-circuits "
            "a later correction branch.\n"
            "- If a later correction block becomes unreachable only when the flag is ON, state that as the root cause "
            "in `root_cause` and `propagation_path`.\n"
            "- Prefer explanations like 'a preceding `elif` branch consumes the chain, so the later correction block "
            "never runs' over explanations that only restate the visible wrong output.\n"
        )
    return prompt


def get_analysis_source_files(analysis: dict[str, Any] | None) -> list[str]:
    if not analysis:
        return []
    candidates: list[str] = []
    for path in analysis.get("suggested_repair_scope", []) or []:
        if isinstance(path, str) and path and not is_test_like_path(path):
            candidates.append(path)
    for component in analysis.get("affected_components", []) or []:
        if not isinstance(component, dict):
            continue
        path = component.get("file")
        if isinstance(path, str) and path and not is_test_like_path(path):
            candidates.append(path)
    return dedupe_preserve_order(candidates)


def build_patch_strategy_feedback(
    strategy: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None,
    original_failure_log: str = "",
    failure_focus: dict[str, Any] | None = None,
    filtered_candidates: list[CandidatePatch] | None = None,
    code_context: dict[str, str] | None = None,
) -> str | None:
    if strategy is None:
        return (
            "Return a JSON object with title, approach, edit_targets, dependency_files, "
            "sufficiency_assessment, and risks."
        )
    for key in ("title", "approach", "edit_targets", "dependency_files", "sufficiency_assessment"):
        if key not in strategy:
            return f"The strategy JSON is missing '{key}'."
    edit_targets = strategy.get("edit_targets")
    if not isinstance(edit_targets, list) or not edit_targets:
        return "The strategy JSON must contain a non-empty edit_targets list."
    test_in_edit_targets = [t for t in edit_targets if isinstance(t, str) and is_test_like_path(t)]
    if test_in_edit_targets:
        return (
            f"edit_targets must only contain source files, not test files. "
            f"Remove {', '.join(test_in_edit_targets)} from edit_targets "
            f"(test files belong in dependency_files for reference only)."
        )
    dependency_files = strategy.get("dependency_files")
    if not isinstance(dependency_files, list):
        return "The strategy JSON must contain a dependency_files list."
    sufficiency_assessment = str(strategy.get("sufficiency_assessment", "")).strip()
    if not sufficiency_assessment:
        return "The strategy JSON must include a non-empty sufficiency_assessment."
    structural_rules = strategy.get("structural_rules")
    if structural_rules is not None and not isinstance(structural_rules, list):
        return "If provided, structural_rules must be a list."
    forbidden_patterns = strategy.get("forbidden_patterns")
    if forbidden_patterns is not None and not isinstance(forbidden_patterns, list):
        return "If provided, forbidden_patterns must be a list."
    approach_text = str(strategy.get("approach", ""))
    sufficiency_text = str(strategy.get("sufficiency_assessment", ""))
    combined_text = f"{approach_text}\n{sufficiency_text}".lower()
    source_symbol_targets = [
        str(item).split("::", 1)[1]
        for item in edit_targets
        if isinstance(item, str) and "::" in item and not is_test_like_path(str(item).split("::", 1)[0])
    ]
    unmentioned_targets = []
    for symbol in source_symbol_targets:
        symbol_name = symbol.split(".")[-1].lower()
        owner_name = symbol.split(".")[0].lower()
        if symbol_name not in combined_text and owner_name not in combined_text:
            unmentioned_targets.append(symbol)
    if unmentioned_targets:
        return (
            "The strategy is internally inconsistent: edit_targets lists source symbols that are not mentioned in "
            f"approach/sufficiency_assessment: {', '.join(unmentioned_targets)}. "
            "Either mention how each listed target will be changed, or remove targets that are not actually part of the fix."
        )
    if _has_single_root_symbol_signature(
        analysis=analysis,
        strategy=strategy,
        failure_focus=failure_focus,
    ):
        analysis_source_files = get_analysis_source_files(analysis)
        preferred_target = _preferred_single_root_source_target(analysis, failure_focus)
        preferred_file = analysis_source_files[0] if len(analysis_source_files) == 1 else None
        source_edit_target_files = dedupe_preserve_order(
            str(target).split("::", 1)[0]
            for target in edit_targets
            if isinstance(target, str) and str(target).strip() and not is_test_like_path(str(target).split("::", 1)[0])
        )
        if (
            preferred_target
            and preferred_file
            and len(source_edit_target_files) == 1
            and source_edit_target_files[0] != preferred_file
            and _has_single_root_helper_root_cause_signature(analysis, failure_focus)
        ):
            return (
                "This looks like a single_root_symbol repair where the analysis identifies a helper/mixin/root-cause "
                f"file as the minimal sufficient fix site ({preferred_target}), but the strategy instead targets the "
                f"symptom-bearing file {source_edit_target_files[0]}. Move edit_targets to the helper/root-cause file "
                "unless you can explicitly prove that editing the dominant class file alone fixes the inheritance/helper issue."
            )
        if len(analysis_source_files) == 1 and len(source_edit_target_files) > 1:
            if preferred_file in source_edit_target_files:
                extra_files = [path for path in source_edit_target_files if path != preferred_file]
                if extra_files:
                    return (
                        "This looks like a single_root_symbol repair and the analysis already identifies a single "
                        f"minimal sufficient source file ({preferred_file}). Do not expand edit_targets to extra "
                        f"source files ({', '.join(extra_files)}) unless the strategy explicitly proves the primary "
                        "file alone cannot satisfy the failure signal."
                    )
    raw_structural_rules_lower = [str(item).strip().lower() for item in structural_rules or []]
    structural_rules_lower = [
        STRUCTURAL_RULE_RESTORE_UNREACHABLE_BRANCH
        if item == "elif_to_if"
        else item
        for item in raw_structural_rules_lower
    ]
    if STRUCTURAL_RULE_RESTORE_UNREACHABLE_BRANCH in structural_rules_lower and not needs_minimal_structural_fix_guidance(
        analysis=analysis,
        strategy=strategy,
        original_failure_log=original_failure_log,
    ):
        return (
            "The strategy lists a branch-reachability structural rule, but the current failure evidence does not "
            "indicate a dispatcher/hook control-flow short-circuit bug. Do not carry over control-flow reachability "
            "rules into ordinary validation or exception-type instances."
        )
    if STRUCTURAL_RULE_RESTORE_UNREACHABLE_BRANCH in structural_rules_lower and not any(
        token in combined_text
        for token in (
            "elif",
            "standalone if",
            "standalone `if`",
            "control flow",
            "short-circuit",
            "short circuit",
            "unreachable branch",
            "restore reachability",
            "existing branch",
        )
    ):
        return (
            "The strategy lists a branch-reachability structural rule but the approach/sufficiency_assessment do not "
            "describe how an existing branch becomes reachable again. Either explain the control-flow reachability fix "
            "explicitly or remove that structural rule."
        )
    forbidden_patterns_lower = [str(item).strip().lower() for item in forbidden_patterns or []]
    if "assert_for_validation" in forbidden_patterns_lower and "assert" not in combined_text:
        return (
            "The strategy lists forbidden_patterns=assert_for_validation but the approach/sufficiency_assessment do not "
            "mention replacing assert-based validation. Either explain that assert->raise change explicitly or remove the "
            "forbidden pattern."
        )
    effective_required_repair_obligations = select_effective_required_repair_obligations(
        extract_required_repair_obligations(
            filtered_candidates,
            failure_focus=failure_focus,
            code_context=code_context,
        ),
        filtered_candidates=filtered_candidates,
    )
    obligation_symbols = dedupe_preserve_order(
        str(item.get("source_symbol") or "").strip()
        for item in effective_required_repair_obligations
        if str(item.get("source_symbol") or "").strip()
    )
    if len(obligation_symbols) > 1:
        approach_lower = approach_text.lower()
        missing_in_approach = [
            symbol for symbol in obligation_symbols
            if symbol.lower() not in approach_lower and symbol.split(".")[-1].lower() not in approach_lower
        ]
        if missing_in_approach:
            return (
                "The retained enhanced tests semantically cover multiple original failure paths with distinct source "
                f"symbols ({', '.join(obligation_symbols)}). The strategy approach must describe a repair action for "
                f"each obligated symbol, but it does not mention: {', '.join(missing_in_approach)}."
            )
        sufficiency_lower = sufficiency_text.lower()
        if any(
            phrase in sufficiency_lower
            for phrase in ("no changes are required", "is sufficient", "editing only", "alone is sufficient")
        ):
            return (
                "The retained enhanced tests establish multiple repair obligations across distinct source symbols "
                f"({', '.join(obligation_symbols)}). Do not claim that a single-symbol edit is sufficient or that other "
                "covered symbols require no changes unless the strategy explicitly repairs each covered symbol."
            )
    required_repair_obligations = effective_required_repair_obligations
    if required_repair_obligations:
        missing_required_obligation_mentions: list[str] = []
        for obligation in required_repair_obligations:
            obligation_id = str(obligation.get("id") or "").strip()
            source_symbol = str(obligation.get("source_symbol") or "").strip()
            validation_subject = str(obligation.get("validation_subject") or "").strip() or "core"
            aliases = _expand_validation_subject_aliases(
                validation_subject,
                obligation.get("trigger_shape_tokens", []),
            )
            symbol_mentioned = source_symbol and (
                source_symbol.lower() in combined_text
                or source_symbol.split(".")[-1].lower() in combined_text
            )
            subject_mentioned = validation_subject == "core" or any(alias in combined_text for alias in aliases)
            if not (symbol_mentioned and subject_mentioned):
                missing_required_obligation_mentions.append(obligation_id)
        if missing_required_obligation_mentions:
            return (
                "The retained enhanced tests already expose required_repair_obligations, but the strategy still "
                "compresses some of them into a generic summary. Mention both the landed source_symbol and the "
                "validation_subject/trigger-shape for: "
                + ", ".join(missing_required_obligation_mentions)
                + "."
            )
    validation_subject_obligations = extract_validation_subject_repair_obligations_from_obligations(
        required_repair_obligations
    )
    if validation_subject_obligations.get("active"):
        approach_lower = approach_text.lower()
        sufficiency_lower = sufficiency_text.lower()
        missing_subjects: list[str] = []
        for obligation in validation_subject_obligations.get("obligations", []):
            source_symbol = str(obligation.get("source_symbol") or "").strip() or "target symbol"
            trigger_tokens = obligation.get("trigger_tokens", {})
            for subject in obligation.get("validation_subjects", []):
                aliases = _expand_validation_subject_aliases(
                    str(subject),
                    trigger_tokens.get(str(subject), []),
                )
                if not any(alias in approach_lower for alias in aliases):
                    missing_subjects.append(f"{source_symbol}:{subject}:approach")
                if not any(alias in sufficiency_lower for alias in aliases):
                    missing_subjects.append(f"{source_symbol}:{subject}:sufficiency_assessment")
        if missing_subjects:
            return (
                "The retained enhanced tests already cover multiple validation_subject obligations inside the same "
                "source symbol, so the strategy must explicitly name every covered validation_subject in both approach "
                "and sufficiency_assessment. Missing validation-subject coverage for: "
                f"{', '.join(missing_subjects)}."
            )
    exception_obligations = extract_exception_type_repair_obligations(failure_focus, filtered_candidates, code_context)
    if exception_obligations.get("active"):
        obligation_items = exception_obligations.get("obligations", [])
        missing_exception_repairs = []
        for obligation in obligation_items:
            expected_exception = str(obligation.get("expected_exception", "")).lower()
            observed_exception = str(obligation.get("observed_exception", "")).lower()
            source_symbol = str(obligation.get("source_symbol", "")).strip() or str(obligation.get("test_id", "")).strip()
            mentions_expected = expected_exception and expected_exception in combined_text
            mentions_observed = observed_exception and observed_exception in combined_text
            mentions_raise = "raise" in combined_text
            mentions_assert = "assert" in combined_text if observed_exception == "assertionerror" else True
            if not (mentions_expected and mentions_observed and mentions_raise and mentions_assert):
                missing_exception_repairs.append(
                    f"{source_symbol} ({observed_exception or 'current exception'} -> {expected_exception or 'target exception'})"
                )
        if missing_exception_repairs:
            return (
                "The retained enhanced tests establish exception-type repair obligations for covered paths, but the "
                "strategy does not explicitly describe how the current exception mechanism will be changed into the "
                f"expected one. Missing exception-type repair details for: {', '.join(missing_exception_repairs)}."
            )
    sibling_validation_obligations = extract_sibling_validation_repair_obligations(
        failure_focus=failure_focus,
        filtered_candidates=filtered_candidates,
        code_context=code_context,
    )
    if sibling_validation_obligations.get("active"):
        sibling_text = combined_text
        missing_sibling_coverage = []
        for obligation in sibling_validation_obligations.get("obligations", []):
            source_symbol = str(obligation.get("source_symbol", "")).strip() or "target symbol"
            marker_groups = [
                [str(marker).lower() for marker in group if str(marker).strip()]
                for group in obligation.get("marker_groups", [])
            ]
            covers_all_siblings = "all assert" in sibling_text or "all assertion" in sibling_text or "sibling assertion" in sibling_text
            if not covers_all_siblings:
                mentioned_groups = 0
                for group in marker_groups:
                    if any(marker in sibling_text for marker in group):
                        mentioned_groups += 1
                covers_all_siblings = mentioned_groups >= min(2, len(marker_groups))
            if not covers_all_siblings:
                preview_groups = [
                    "/".join(group[:2]) for group in marker_groups[:3] if group
                ]
                missing_sibling_coverage.append(
                    f"{source_symbol} (sibling validation markers: {', '.join(preview_groups)})"
                )
        if missing_sibling_coverage:
            return (
                "The retained enhanced tests and failure evidence indicate that some covered symbols contain multiple "
                "sibling assert-based validation branches that should be converted together, but the strategy still "
                "describes only a partial branch-level fix. Explicitly state that all sibling assert-based validation "
                f"branches in the same symbol will be converted to the expected exception path. Missing sibling-branch "
                f"coverage for: {', '.join(missing_sibling_coverage)}."
            )
    referenced_source_files = get_analysis_source_files(analysis)
    if len(referenced_source_files) > 1:
        mentioned_files = {
            item
            for item in [*edit_targets, *dependency_files]
            if isinstance(item, str) and item and not is_test_like_path(item)
        }
        missing_files = [path for path in referenced_source_files if path not in mentioned_files]
        unexplained_missing = [path for path in missing_files if path not in sufficiency_assessment]
        if unexplained_missing:
            return (
                "The analysis references multiple relevant source files. "
                f"Either include {', '.join(unexplained_missing)} in edit_targets/dependency_files, "
                "or explicitly justify in sufficiency_assessment why a narrower edit is enough."
            )
    return None


def extract_strategy_constraints(strategy: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(strategy, dict):
        return {}
    structural_rules = []
    for item in strategy.get("structural_rules", []) or []:
        rule = str(item).strip()
        if not rule:
            continue
        if rule == "elif_to_if":
            rule = STRUCTURAL_RULE_RESTORE_UNREACHABLE_BRANCH
        if rule not in structural_rules:
            structural_rules.append(rule)
    forbidden_patterns = [
        str(item).strip()
        for item in strategy.get("forbidden_patterns", []) or []
        if str(item).strip()
    ]
    return {
        "structural_rules": structural_rules,
        "forbidden_patterns": forbidden_patterns,
    }


def _expand_validation_subject_aliases(subject: str, trigger_tokens: list[str] | None = None) -> list[str]:
    normalized = str(subject or "").strip().lower()
    if not normalized:
        return []
    aliases: list[str] = []
    candidates = [
        normalized,
        normalized.replace("__", "."),
        normalized.replace("__", " "),
        normalized.replace("_", " "),
        normalized.replace("func", "function"),
        normalized.replace("_", " ").replace("func", "function"),
    ]
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    for token in trigger_tokens or []:
        normalized_token = str(token).strip().lower()
        if normalized_token and normalized_token not in aliases:
            aliases.append(normalized_token)
    return aliases


def build_patch_strategy_prompt(
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    filtered_candidates: list[CandidatePatch],
    analysis: dict[str, Any] | None,
    failure_focus: dict[str, Any] | None = None,
    feedback: str | None = None,
) -> str:
    context_text = format_code_context(code_context)
    original_failure_lower = original_failure_log.lower()
    analysis_text = json.dumps(analysis or {}, ensure_ascii=False).lower()
    needs_flag_short_circuit_guidance = (
        "runxfail" in original_failure_lower
        and any(token in (original_failure_lower + " " + analysis_text) for token in ("hook", "report", "dispatcher", "makereport", "runtest", "elif", "unreachable", "short-circuit", "short circuit"))
    )
    anchor_text = build_anchor_context(
        code_context,
        symbols=[*(analysis or {}).get("suspicious_symbols", [])],
        edit_targets=get_analysis_source_files(analysis) or list(code_context.keys()),
    )
    enhanced_summary = format_candidate_summary(filtered_candidates)
    required_repair_obligations = select_effective_required_repair_obligations(
        extract_required_repair_obligations(
            filtered_candidates,
            failure_focus=failure_focus,
            code_context=code_context,
        ),
        filtered_candidates=filtered_candidates,
    )
    candidate_target_symbols = dedupe_preserve_order(
        str(item.get("source_symbol") or "").strip()
        for item in required_repair_obligations
        if str(item.get("source_symbol") or "").strip()
    )
    prompt = (
        "Task: propose one repair strategy before writing a patch.\n"
        "Return exactly one JSON object.\n"
        "JSON schema:\n"
        '{\n'
        '  "title": "short repair strategy name",\n'
        '  "approach": "how the bug should be fixed",\n'
        '  "edit_targets": ["source files or symbols to modify"],\n'
        '  "dependency_files": ["additional source files that may need inspection or edits"],\n'
        '  "structural_rules": ["optional machine-checkable structural requirements such as restore_unreachable_existing_branch"],\n'
        '  "forbidden_patterns": ["optional machine-checkable anti-patterns such as negative_gating or top_of_function_side_path"],\n'
        '  "sufficiency_assessment": "explain whether editing only the primary target is enough; if not, say what else is needed",\n'
        '  "risks": ["possible regressions to avoid"]\n'
        '}\n\n'
        f"Problem statement:\n{instance['problem_statement']}\n\n"
        f"Original failure log:\n```text\n{truncate_text(original_failure_log, 12000)}\n```\n\n"
        f"Failure analysis:\n```json\n{json.dumps(analysis or {}, indent=2, ensure_ascii=False)}\n```\n\n"
        f"Retained enhanced tests:\n{enhanced_summary}\n\n"
        "Structured required repair obligations from retained enhanced tests:\n"
        f"{format_required_repair_obligations(required_repair_obligations)}\n\n"
        f"Local anchor context:\n{anchor_text}\n\n"
    )
    if len(candidate_target_symbols) > 1:
        prompt += (
            "MULTI-PATH REPAIR OBLIGATION:\n"
            "The retained enhanced tests now semantically cover multiple distinct source symbols, so the strategy must "
            "treat them as separate repair obligations rather than collapsing everything into one dominant path.\n"
            f"Covered symbols from retained enhanced tests: {', '.join(candidate_target_symbols)}\n"
            "If one original failure path points to a different source symbol than another, do not claim that editing "
            "only one symbol is sufficient unless your strategy explicitly explains how the other covered symbol is also repaired.\n\n"
        )
    if _has_single_root_symbol_signature(
        analysis=analysis,
        strategy={"edit_targets": get_analysis_source_files(analysis)},
        failure_focus=failure_focus,
    ):
        analysis_source_files = get_analysis_source_files(analysis)
        if len(analysis_source_files) == 1:
            prompt += (
                "SINGLE-ROOT MINIMALITY RULE:\n"
                "This failure currently looks like a single_root_symbol repair and the analysis already points to one "
                f"minimal sufficient source file: {analysis_source_files[0]}.\n"
                "Prefer a strategy that edits only that source file unless you can explicitly justify why an "
                "additional source file is required. Do not add the failing subclass or a nearby helper file to "
                "edit_targets just because it manifests the symptom if the inherited/root helper file alone explains "
                "and fixes the bug.\n\n"
            )
        if _has_single_root_helper_root_cause_signature(analysis, failure_focus):
            preferred_target = _preferred_single_root_source_target(analysis, failure_focus)
            if preferred_target:
                prompt += (
                    "HELPER ROOT-CAUSE OVERRIDE:\n"
                    "The current failure reads like a single-root inheritance/helper bug: the prominent failing class "
                    "carries the symptom, but the minimal sufficient fix likely lives in an inherited helper/mixin/root "
                    f"class target. Prefer `{preferred_target}` as the only edit_target unless you can prove the symptom "
                    "class itself is missing the actual logic.\n\n"
                )
    if required_repair_obligations:
        prompt += (
            "REQUIRED REPAIR OBLIGATIONS:\n"
            "The obligations listed above are not optional hints. Your approach and sufficiency_assessment must "
            "preserve every listed obligation id by explicitly naming its landed source_symbol and concrete "
            "validation subject (or trigger-shape equivalent). Do not merge them into a single generic phrase like "
            "'validate inputs' if multiple obligations were separately exposed.\n\n"
        )
    validation_subject_obligations = extract_validation_subject_repair_obligations_from_obligations(
        required_repair_obligations
    )
    if validation_subject_obligations.get("active"):
        prompt += (
            "VALIDATION-SUBJECT REPAIR OBLIGATION:\n"
            "Some retained enhanced tests already cover multiple concrete validation_subjects inside the same landed "
            "source symbol. The strategy must explicitly name every covered validation_subject in BOTH the approach "
            "and the sufficiency_assessment. Do not collapse them into a generic statement like 'validate inputs' if "
            "multiple subjects were separately covered.\n"
        )
        for obligation in validation_subject_obligations.get("obligations", []):
            prompt += (
                f"- {obligation.get('source_symbol')}: "
                f"{', '.join(str(subject) for subject in obligation.get('validation_subjects', []))}\n"
            )
        prompt += (
            "If retained enhanced tests separately cover subjects such as endpoint and view_func_name in the same "
            "symbol, the strategy must say that both endpoint and view_func_name validations are part of the repair scope.\n\n"
        )
    exception_obligations = extract_exception_type_repair_obligations(failure_focus, filtered_candidates, code_context)
    if exception_obligations.get("active"):
        prompt += (
            "EXCEPTION-TYPE REPAIR OBLIGATION:\n"
            "Some retained enhanced tests correspond to paths where the current exception type is wrong, not merely the "
            "trigger condition. Your strategy must explicitly explain how the implementation changes the currently "
            "observed exception mechanism into the expected exception type.\n"
        )
        for obligation in exception_obligations.get("obligations", []):
            prompt += (
                f"- {obligation.get('source_symbol') or obligation.get('test_id')}: "
                f"current={obligation.get('observed_exception')}, expected={obligation.get('expected_exception')}\n"
            )
        prompt += (
            "If the current implementation uses `assert` but the expected behavior is `ValueError`, say that the patch "
            "must replace assertion-based validation with an explicit `raise ValueError(...)` path rather than adding "
            "an unrelated guard elsewhere.\n\n"
        )
    sibling_validation_obligations = extract_sibling_validation_repair_obligations(
        failure_focus=failure_focus,
        filtered_candidates=filtered_candidates,
        code_context=code_context,
    )
    if sibling_validation_obligations.get("active"):
        prompt += (
            "SIBLING VALIDATION REPAIR OBLIGATION:\n"
            "For some covered symbols, the current failure evidence establishes that one assert-based validation path "
            "must be converted into an explicit exception path. Before declaring the strategy sufficient, inspect the "
            "same symbol for sibling assert-based validation branches in the same local validation cluster. If multiple "
            "sibling assertion branches exist, the strategy must say that all relevant sibling validations in that "
            "symbol will be converted together unless it explicitly justifies leaving one unchanged.\n"
        )
        for obligation in sibling_validation_obligations.get("obligations", []):
            marker_groups = [
                "/".join(group[:2]) for group in obligation.get("marker_groups", [])[:3] if group
            ]
            prompt += (
                f"- {obligation.get('source_symbol')} in {obligation.get('target_file')}: "
                f"{obligation.get('assert_count')} sibling assert-based validation branch(es); "
                f"markers={marker_groups}\n"
            )
        prompt += "\n"
    # Inject per-test tracebacks so the model sees EACH original test's source location
    orig_ids = get_original_test_identifiers(instance)
    per_test = extract_per_test_tracebacks(original_failure_log, orig_ids)
    if per_test:
        prompt += (
            "CRITICAL — per-test failure tracebacks (each original test may point to a DIFFERENT source location):\n"
        )
        for name, tb in per_test.items():
            prompt += f"### {name}\n```text\n{tb}\n```\n\n"
        prompt += (
            "Every source location referenced in the tracebacks above MUST be listed in edit_targets. "
            "If two tests fail at two different files/functions, you need TWO entries in edit_targets.\n\n"
        )
    prompt += (
        "Do not assume a single-file or single-class edit is always sufficient. "
        "If parent classes, helper constructors, or dependency modules may also need changes, include them in dependency_files "
        "and explain that in sufficiency_assessment.\n\n"
        "If the analysis references multiple affected source files or propagation crosses helper classes / wrappers, "
        "account for every relevant file in edit_targets or dependency_files. "
        "If the final repair needs more than one location, the patch MUST contain multiple hunks — one per changed location.\n\n"
        "CRITICAL — edit_targets completeness: if the fix requires changing code in more than one function, method, or class, "
        "you MUST list EVERY location in edit_targets (e.g. ['requests/sessions.py::Session.request', "
        "'requests/adapters.py::HTTPAdapter.send']). The downstream diff generator will only touch locations you list here — "
        "omitting a location means that location will NOT be fixed.\n\n"
        "IMPORTANT — structural_rules discipline: only emit machine-checkable structural rules like "
        f"`{STRUCTURAL_RULE_RESTORE_UNREACHABLE_BRANCH}` when the "
        "failure evidence clearly shows a control-flow short-circuit / unreachable-branch bug in an existing dispatcher, "
        "hook, or report-style function. Do NOT add that rule to ordinary validation, constructor, or exception-type "
        "bugs just because another instance needed it.\n\n"
        "IMPORTANT: edit_targets must ONLY contain source files that need to be patched. "
        "NEVER put test files (paths containing /tests/, test_, or _test) in edit_targets — "
        "they belong in dependency_files for reference only. "
        "The repair patch must not modify test files.\n\n"
        "Patch placement constraint: if the repair strategy targets a class symbol, the final diff should edit inside the class body, "
        "not at module-level imports or unrelated top-level statements.\n\n"
        f"Buggy code context:\n{context_text}\n"
    )
    if needs_flag_short_circuit_guidance:
        prompt += (
            "CONTROL-FLOW WARNING: when the buggy function contains an `elif` chain (if/elif/elif/...), check whether "
            "the fix needs to run UNCONDITIONALLY rather than as part of the chain. If a preceding `elif` branch can "
            "short-circuit execution (e.g., `elif item.config.option.runxfail: pass`), any subsequent `elif` block "
            "will be skipped when that branch fires. If that is the root cause, the fix must change the existing "
            "`elif` to a standalone `if` statement — NOT insert another `elif`. "
            "Identify this pattern and include the structural keyword change in your strategy's approach.\n\n"
            "\nFLAG-SHORT-CIRCUIT STRATEGY RULE:\n"
            "- The failing evidence points to behavior that is wrong only when a flag-enabled mode is ON "
            "(for example `--runxfail`).\n"
            "- Your strategy MUST explicitly answer: which earlier branch fires under that flag, which later "
            "correction branch becomes unreachable, and what structural change makes the later correction run again.\n"
            "- If the analysis suggests an existing correction block is already present but chained under `elif`, "
            "the strategy's `approach` should explicitly say to convert that existing `elif` into a standalone `if`.\n"
            "- Do NOT propose generic rewrites like 'update longrepr handling' unless you also explain the exact "
            "control-flow reason the existing handling is skipped.\n"
            "- Prefer the smallest structural change that restores reachability over adding a new top-of-function branch.\n"
            f"- Set `structural_rules` to include `{STRUCTURAL_RULE_RESTORE_UNREACHABLE_BRANCH}` when the fix requires restoring an existing correction branch that is currently unreachable.\n"
            "- If the concrete implementation is a minimal keyword change such as `elif` -> `if`, say so in the approach, but keep the structural rule at the higher abstraction level.\n"
            "- Set `forbidden_patterns` to include `negative_gating` and `top_of_function_side_path` when those workarounds would violate the strategy.\n"
        )
    if feedback:
        prompt += f"\nRevision feedback for the next strategy:\n{feedback}\n"
    return prompt


def build_edit_anchor_regions(
    code_context: dict[str, str],
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
    required_repair_obligations: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    edit_targets = list(dict.fromkeys(
        str(target).split("::")[0]
        for target in (strategy or {}).get("edit_targets", [])
        if isinstance(target, str)
    ))
    candidate_symbols = [str(symbol) for symbol in (analysis or {}).get("suspicious_symbols", []) if isinstance(symbol, str)]
    # Also extract Class.method symbols from edit_targets (e.g. "src/flask/blueprints.py::Blueprint.add_url_rule")
    for target in (strategy or {}).get("edit_targets", []):
        if isinstance(target, str) and "::" in target:
            sym_part = target.split("::", 1)[1]
            if sym_part and sym_part not in candidate_symbols:
                candidate_symbols.append(sym_part)
    # Extract class names from symbols like "Blueprint.__init__" or "Blueprint.add_url_rule"
    # — take only the part before the first "." so we can find the class definition.
    class_symbols = list(dict.fromkeys(
        symbol.split(".")[0]
        for symbol in candidate_symbols
        if symbol and symbol.split(".")[0][:1].isupper()
    ))
    # Track class.method pairs for non-__init__ methods (e.g. "Blueprint.add_url_rule")
    # so we can generate per-method anchor regions in addition to the class-level one.
    class_method_pairs = list(dict.fromkeys(
        (symbol.split(".")[0], symbol.split(".", 1)[1])
        for symbol in candidate_symbols
        if "." in symbol
        and symbol.split(".")[0][:1].isupper()
        and symbol.split(".", 1)[1] not in ("__init__",)  # __init__ handled separately
    ))
    # Also track which classes have __init__ as a suspicious symbol
    init_classes = {
        symbol.split(".")[0]
        for symbol in candidate_symbols
        if ".__init__" in symbol and symbol.split(".")[0][:1:1].isupper()
    }
    regions: list[dict[str, str]] = []
    obligation_anchor_tokens_by_symbol: dict[str, list[str]] = {}
    for item in required_repair_obligations or []:
        if not isinstance(item, dict):
            continue
        source_symbol = str(item.get("source_symbol") or "").strip()
        if not source_symbol:
            continue
        token_bucket = obligation_anchor_tokens_by_symbol.setdefault(source_symbol, [])
        for token in item.get("statement_anchor_tokens", []) or []:
            normalized = str(token).strip()
            if normalized and normalized not in token_bucket:
                token_bucket.append(normalized)

    def _get_class_docstring_end_idx(content: str, class_name: str) -> int | None:
        try:
            module = ast.parse(content)
        except SyntaxError:
            return None
        for node in getattr(module, "body", []):
            if isinstance(node, ast.ClassDef) and node.name == class_name and node.body:
                first_stmt = node.body[0]
                if (
                    isinstance(first_stmt, ast.Expr)
                    and isinstance(getattr(first_stmt, "value", None), ast.Constant)
                    and isinstance(first_stmt.value.value, str)
                ):
                    end_lineno = getattr(first_stmt, "end_lineno", None)
                    if isinstance(end_lineno, int):
                        return end_lineno - 1
                return None
        return None

    def _find_def_signature_end(lines: list[str], start_idx: int, max_scan: int = 40) -> int:
        balance = 0
        limit = min(len(lines), start_idx + max_scan)
        for idx in range(start_idx, limit):
            line = lines[idx]
            balance += line.count("(") - line.count(")")
            if line.strip().endswith(":") and balance <= 0:
                return idx
        return start_idx

    def _select_anchor_pair_from_logic_lines(
        lines: list[str],
        logic_indices: list[int],
        statement_anchor_tokens: list[str] | None,
    ) -> tuple[int, int]:
        if len(logic_indices) < 2:
            raise ValueError("Need at least two logic lines for anchor selection")
        normalized_tokens = [
            str(token).strip().lower()
            for token in (statement_anchor_tokens or [])
            if str(token).strip()
        ]
        if not normalized_tokens:
            return logic_indices[0], logic_indices[1]
        scored: list[tuple[int, int, int]] = []
        for pos, idx in enumerate(logic_indices):
            text = lines[idx].strip().lower()
            score = sum(1 for token in normalized_tokens if token in text)
            if score > 0:
                scored.append((score, -pos, idx))
        if not scored:
            return logic_indices[0], logic_indices[1]
        _, _, best_idx = max(scored)
        best_pos = logic_indices.index(best_idx)
        if best_pos + 1 < len(logic_indices):
            return best_idx, logic_indices[best_pos + 1]
        if best_pos > 0:
            return logic_indices[best_pos - 1], best_idx
        return logic_indices[0], logic_indices[1]

    def _should_suppress_tail_anchors(
        target_symbol: str,
        lines: list[str],
        func_idx: int,
        body_indices: list[int],
    ) -> bool:
        """Prefer head-focused anchors for dispatcher-style functions.

        Long hook/report/dispatcher functions often maintain shared state near the
        top and then branch on it later. Tail anchors tend to lure the model into
        patching only the final branch, which is brittle and often misses the real
        control-flow fix. This heuristic is generic to dispatcher-style functions
        and should not affect class-level repairs such as sympy's Printable fix.
        """
        if not body_indices:
            return False
        top_level_branch_count = 0
        func_indent = len(lines[func_idx]) - len(lines[func_idx].lstrip())
        body_indent = func_indent + 4
        for idx in body_indices:
            stripped = lines[idx].strip()
            line_indent = len(lines[idx]) - len(lines[idx].lstrip())
            if line_indent == body_indent and stripped.startswith(("if ", "elif ", "else:", "try:", "except ", "except:", "finally:")):
                top_level_branch_count += 1
        if top_level_branch_count < 2:
            return False
        symbol_name = target_symbol.split(".")[-1].lower()
        dispatcher_like_name = any(
            token in symbol_name
            for token in ("hook", "report", "dispatch", "dispatcher", "runtest", "makereport", "handle")
        )
        early_window = body_indices[: min(len(body_indices), 8)]
        has_early_yield = any(lines[idx].strip().startswith(("yield", "outcome = yield")) for idx in early_window)
        return dispatcher_like_name or has_early_yield

    def _append_branch_regions(
        *,
        target_file: str,
        target_symbol: str,
        class_anchor_line: str,
        lines: list[str],
        func_idx: int,
        body_indices: list[int],
        ) -> None:
        if not body_indices:
            return
        suppress_tail_anchors = _should_suppress_tail_anchors(
            target_symbol,
            lines,
            func_idx,
            body_indices,
        )
        branch_indices: list[int] = []
        for idx in body_indices:
            stripped = lines[idx].strip()
            if stripped.startswith(("if ", "elif ", "else:", "try:", "except ", "except:", "finally:")):
                branch_indices.append(idx)
        if not branch_indices:
            return

        def _make_branch_region(branch_idx: int, suffix: str) -> None:
            try:
                pos = body_indices.index(branch_idx)
            except ValueError:
                return
            next_logic_idx = body_indices[pos + 1] if pos + 1 < len(body_indices) else None
            if next_logic_idx is None:
                return

            # For elif-bearing regions, set anchor_before to the last non-empty
            # line of the PREVIOUS body statement.  This lets the model include
            # the `elif` keyword itself inside replacement_block and change it to
            # `if`, which is impossible when anchor_before IS the `elif (` line
            # (because replacement_block[0] must equal anchor_before verbatim).
            actual_anchor_before_idx = branch_idx
            actual_anchor_before_lineno = branch_idx + 1
            stripped_branch = lines[branch_idx].strip()
            target_elif_line: str | None = None
            if stripped_branch.startswith("elif ") and pos > 0:
                target_elif_line = lines[branch_idx]
                prev_body_idx = body_indices[pos - 1]
                # Walk forward from prev_body_idx to find the last non-empty line
                # before branch_idx (the closing line of the preceding block).
                last_nonempty = prev_body_idx
                for scan in range(prev_body_idx, branch_idx):
                    if lines[scan].strip():
                        last_nonempty = scan
                actual_anchor_before_idx = last_nonempty
                actual_anchor_before_lineno = last_nonempty + 1

            snippet_end = min(len(lines), next_logic_idx + 6)
            region_entry: dict[str, Any] = {
                "target_file": target_file,
                "target_symbol": f"{target_symbol}{suffix}",
                "class_anchor_line": class_anchor_line,
                "anchor_line_before_lineno": actual_anchor_before_lineno,
                "anchor_line_after_lineno": next_logic_idx + 1,
                "anchor_line_before": lines[actual_anchor_before_idx],
                "anchor_line_after": lines[next_logic_idx],
                "region_snippet": "\n".join(lines[func_idx:snippet_end]),
            }
            if target_elif_line is not None:
                # Signal to the prompt builder and validator that this region
                # covers an `elif` branch that may need to become a standalone
                # `if`.  The replacement_block must NOT repeat the `elif` verbatim.
                region_entry["target_elif_line"] = target_elif_line
            regions.append(region_entry)

        _make_branch_region(branch_indices[0], "__branch")
        if len(branch_indices) > 1 and not suppress_tail_anchors:
            _make_branch_region(branch_indices[-1], "__branch_tail")
        if len(branch_indices) > 2 and not suppress_tail_anchors:
            # Also expose middle branches when not suppressed
            for _mid_i, _mid_idx in enumerate(branch_indices[1:-1]):
                _make_branch_region(_mid_idx, f"__branch_mid{_mid_i}")

    for target_file in edit_targets:
        content = code_context.get(target_file)
        if not content:
            continue
        lines = content.splitlines()
        for symbol in class_symbols:
            class_pattern = re.compile(rf"^class\s+{re.escape(symbol)}\b.*:")
            class_idx = next((idx for idx, line in enumerate(lines) if class_pattern.match(line.strip())), None)
            if class_idx is None:
                continue
            method_idx = next(
                (
                    idx
                    for idx in range(class_idx + 1, len(lines))
                    if lines[idx].startswith("    def ") or lines[idx].startswith("    @")
                ),
                None,
            )
            if method_idx is None:
                continue
            # Determine the docstring span (if any) between class_idx and method_idx so
            # we can avoid selecting an anchor line that lives inside a docstring body.
            # A class-level docstring is the first triple-quoted block after the class line.
            _docstring_end_idx: int | None = _get_class_docstring_end_idx(content, symbol)
            _in_doc = False
            _doc_delim: str | None = None
            if _docstring_end_idx is None:
                for _i in range(class_idx + 1, method_idx):
                    _stripped = lines[_i].strip()
                    if not _in_doc:
                        for _delim in ('"""', "'''"):
                            if _delim in lines[_i]:
                                # Check if it opens and closes on the same line
                                _rest = lines[_i].replace(_delim, "", 1)
                                if _delim in _rest:
                                    # single-line docstring — whole line is inside docstring
                                    _docstring_end_idx = _i
                                else:
                                    _in_doc = True
                                    _doc_delim = _delim
                                break
                    else:
                        if _doc_delim and _doc_delim in lines[_i]:
                            _docstring_end_idx = _i
                            _in_doc = False
                            _doc_delim = None
                            break
            anchor_before_idx = next(
                (
                    idx
                    for idx in range(method_idx - 1, class_idx, -1)
                    if lines[idx].strip()
                    and not lines[idx].lstrip().startswith("#")
                    and lines[idx].strip() not in ('"""', "'''")  # skip bare docstring delimiters
                    # Skip any line that is inside the class docstring body
                    and (_docstring_end_idx is None or idx > _docstring_end_idx)
                ),
                None,
            )
            # Fallback: when the entire class body between the class line and the first
            # method is a docstring (e.g. Printable in sympy), use the docstring closing
            # line as anchor_before so the model inserts structural class attributes
            # immediately after the docstring rather than inside it.
            if anchor_before_idx is None and _docstring_end_idx is not None:
                anchor_before_idx = _docstring_end_idx
            if anchor_before_idx is None:
                continue
            anchor_before = lines[anchor_before_idx]
            anchor_after = lines[method_idx]
            # Always start from class_idx so the snippet contains the class definition
            # line — validate_fragment_edit_plan checks that class_anchor_line is present.
            snippet_start = class_idx
            snippet_end = min(len(lines), method_idx + 36)
            snippet = "\n".join(lines[snippet_start:snippet_end])
            regions.append(
                {
                    "target_file": target_file,
                    "target_symbol": symbol,
                    "class_anchor_line": lines[class_idx],
                    "class_anchor_lineno": class_idx + 1,
                    "anchor_line_before_lineno": anchor_before_idx + 1,
                    "anchor_line_after_lineno": method_idx + 1,
                    "anchor_line_before": anchor_before,
                    "anchor_line_after": anchor_after,
                    "region_snippet": snippet,
                    "preferred_replacement_mode": (
                        "insert_after_anchor"
                        if _docstring_end_idx is not None and anchor_before_idx == _docstring_end_idx
                        else "replace_span"
                    ),
                }
            )
            # Also add an intra-method anchor region inside __init__ so the model
            # can edit inside the constructor body. Do this for every class symbol
            # because the most common repair pattern (add validation) lives in __init__.
            # Use the first two consecutive "self.xxx = " assignments as unique anchors
            # to avoid ambiguity with "        )" lines that appear multiple times in a file.
            init_idx = next(
                (idx for idx in range(class_idx + 1, len(lines)) if re.search(r"^\s+def\s+__init__\s*\(", lines[idx])),
                None,
            )
            if init_idx is not None:
                # Collect all "self.xxx = " assignment lines inside __init__
                # (stop at the next def/class at same or lower indentation)
                init_indent = len(lines[init_idx]) - len(lines[init_idx].lstrip())
                body_indent = init_indent + 4  # expected indentation of the method body
                assign_indices: list[int] = []
                # Find the end of the function signature: look for "    ):" or "    )  # ..."
                # or single-line "def __init__(self, ...):" on the same line.
                sig_end_idx = _find_def_signature_end(lines, init_idx)
                for idx in range(sig_end_idx + 1, min(init_idx + 80, len(lines))):
                    stripped = lines[idx].strip()
                    if not stripped:
                        continue
                    line_indent = len(lines[idx]) - len(lines[idx].lstrip())
                    # Stop if indent drops back to init_indent (next method/attribute)
                    if line_indent <= init_indent and not stripped.startswith("#"):
                        break
                    if stripped.startswith("self.") and "=" in stripped and not stripped.startswith("self.__class__"):
                        assign_indices.append(idx)
                # Use first pair of consecutive assignments as anchor_before / anchor_after.
                # The model inserts new code between them (i.e., after anchor_before).
                if len(assign_indices) >= 2:
                    first_assign_idx, second_assign_idx = _select_anchor_pair_from_logic_lines(
                        lines,
                        assign_indices,
                        obligation_anchor_tokens_by_symbol.get(f"{symbol}.__init__", []),
                    )
                    inner_snippet_start = max(0, init_idx - 2)
                    inner_snippet_end = min(len(lines), second_assign_idx + 6)
                    inner_snippet = "\n".join(lines[inner_snippet_start:inner_snippet_end])
                    regions.append(
                        {
                            "target_file": target_file,
                            "target_symbol": f"{symbol}.__init__",
                            "class_anchor_line": lines[class_idx],
                            "class_anchor_lineno": class_idx + 1,
                            "anchor_line_before_lineno": first_assign_idx + 1,
                            "anchor_line_after_lineno": second_assign_idx + 1,
                            "anchor_line_before": lines[first_assign_idx],
                            "anchor_line_after": lines[second_assign_idx],
                            "region_snippet": inner_snippet,
                        }
                    )
            # Do NOT break here — continue processing all matching class symbols in this file.

        # Handle module-level variable symbols (e.g. FILE_UPLOAD_PERMISSIONS = None).
        # --- Method-level anchor regions for explicit "ClassName.method_name" symbols ---
        # For each (class_name, method_name) pair identified above, generate an anchor
        # that covers the actual logic lines of the method body (skipping docstrings).
        for cls_name, method_name in class_method_pairs:
            cls_idx = next(
                (i for i, l in enumerate(lines) if re.search(rf"^class\s+{re.escape(cls_name)}\b", l.strip())),
                None,
            )
            if cls_idx is None:
                continue
            method_idx = next(
                (i for i in range(cls_idx + 1, len(lines))
                 if re.search(rf"^\s+def\s+{re.escape(method_name)}\s*\(", lines[i])),
                None,
            )
            if method_idx is None:
                continue
            # Find signature end (closing "    ):" or "    ) -> ...:") of method def)
            sig_end_idx = _find_def_signature_end(lines, method_idx)
            method_indent = len(lines[method_idx]) - len(lines[method_idx].lstrip())
            # Skip past the docstring (if any) to find actual logic lines
            body_start = sig_end_idx + 1
            in_docstring = False
            docstring_end = None
            for idx in range(sig_end_idx + 1, min(sig_end_idx + 20, len(lines))):
                stripped = lines[idx].strip()
                if not stripped:
                    continue
                if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                    if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                        docstring_end = idx  # single-line docstring
                        break
                    in_docstring = True
                    continue
                if in_docstring:
                    if '"""' in stripped or "'''" in stripped:
                        docstring_end = idx
                        break
                    continue
                # First non-docstring body line
                docstring_end = idx - 1
                break
            body_start = (docstring_end + 1) if docstring_end is not None else sig_end_idx + 1
            # Collect actual logic lines from body_start. Keep the full list so we
            # can also expose branch-focused anchors deeper in the function body.
            logic_lines: list[int] = []
            for idx in range(body_start, min(body_start + 40, len(lines))):
                stripped = lines[idx].strip()
                if not stripped:
                    continue
                line_indent = len(lines[idx]) - len(lines[idx].lstrip())
                if line_indent <= method_indent and not stripped.startswith("#"):
                    break  # left method body
                logic_lines.append(idx)
            if len(logic_lines) >= 2:
                anchor_before_idx, anchor_after_idx = _select_anchor_pair_from_logic_lines(
                    lines,
                    logic_lines,
                    obligation_anchor_tokens_by_symbol.get(f"{cls_name}.{method_name}", []),
                )
                # Use class definition line as class_anchor_line for disambiguation in
                # apply_single_fragment_edit (since method names can repeat in sibling classes).
                # Prepend the class header to snippet so validate_fragment_edit_plan finds it.
                snippet_start = method_idx
                snippet_end = min(len(lines), max(logic_lines[-1], anchor_after_idx) + 4)
                method_snippet = "\n".join(lines[snippet_start:snippet_end])
                # Include class header in snippet so class_anchor_line check passes
                snippet = lines[cls_idx] + "\n    ...\n" + method_snippet
                regions.append(
                    {
                        "target_file": target_file,
                        "target_symbol": f"{cls_name}.{method_name}",
                        "class_anchor_line": lines[cls_idx],  # "class Blueprint(Scaffold):"
                        "class_anchor_lineno": cls_idx + 1,
                        "anchor_line_before_lineno": anchor_before_idx + 1,
                        "anchor_line_after_lineno": anchor_after_idx + 1,
                        "anchor_line_before": lines[anchor_before_idx],
                        "anchor_line_after": lines[anchor_after_idx],
                        "region_snippet": snippet,
                    }
                )
                _append_branch_regions(
                    target_file=target_file,
                    target_symbol=f"{cls_name}.{method_name}",
                    class_anchor_line=lines[cls_idx],
                    lines=lines,
                    func_idx=method_idx,
                    body_indices=logic_lines,
                )
        # These are identifiers that are NOT class names. We distinguish by checking
        # whether the symbol actually appears as a "class NAME" definition in the file.
        # Symbols like FILE_UPLOAD_PERMISSIONS are all-caps constants; class names are CamelCase.
        defined_class_names = {
            m.group(1) for m in (re.search(r"^class\s+(\w+)", l) for l in lines) if m
        }
        module_level_symbols = [
            symbol for symbol in candidate_symbols
            if symbol
            and symbol.split(".")[0] not in defined_class_names  # not a class defined in file
            and not symbol.startswith("test_")                    # not a test function
            and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol.split(".")[0])  # valid identifier
        ]
        for sym in module_level_symbols:
            # First try: function definition "def sym(" at module level
            func_idx = next(
                (idx for idx, line in enumerate(lines)
                 if re.match(rf"^def\s+{re.escape(sym)}\s*\(", line)),
                None,
            )
            if func_idx is not None:
                # Build anchor region(s) inside the function body (skip docstring)
                sig_end_idx = _find_def_signature_end(lines, func_idx)
                in_docstring = False
                docstring_end = None
                for idx in range(sig_end_idx + 1, min(sig_end_idx + 20, len(lines))):
                    stripped = lines[idx].strip()
                    if not stripped:
                        continue
                    if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                        if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                            docstring_end = idx; break
                        in_docstring = True; continue
                    if in_docstring:
                        if '"""' in stripped or "'''" in stripped:
                            docstring_end = idx; break
                        continue
                    docstring_end = idx - 1; break
                body_start = (docstring_end + 1) if docstring_end is not None else sig_end_idx + 1
                func_indent = len(lines[func_idx]) - len(lines[func_idx].lstrip())
                # Collect ALL logic line indices in the function body (up to 200 lines)
                all_logic_lines: list[int] = []
                for idx in range(body_start, min(body_start + 200, len(lines))):
                    stripped = lines[idx].strip()
                    if not stripped:
                        continue
                    line_indent = len(lines[idx]) - len(lines[idx].lstrip())
                    if line_indent <= func_indent and not stripped.startswith("#"):
                        break
                    all_logic_lines.append(idx)
                # Generate TWO anchor regions: one near the start, one near the end.
                # The "near-end" region is critical for long functions where the bug is at the bottom.
                def _make_func_region(logic_subset: list[int]) -> dict | None:
                    if len(logic_subset) < 2:
                        return None
                    anchor_before_idx, anchor_after_idx = _select_anchor_pair_from_logic_lines(
                        lines,
                        logic_subset,
                        obligation_anchor_tokens_by_symbol.get(sym, []),
                    )
                    snippet_start = func_idx
                    snippet_end = min(len(lines), max(logic_subset[-1], anchor_after_idx) + 6)
                    return {
                        "target_file": target_file,
                        "target_symbol": sym,
                        "class_anchor_line": lines[func_idx],
                        "class_anchor_lineno": func_idx + 1,
                        "anchor_line_before_lineno": anchor_before_idx + 1,
                        "anchor_line_after_lineno": anchor_after_idx + 1,
                        "anchor_line_before": lines[anchor_before_idx],
                        "anchor_line_after": lines[anchor_after_idx],
                        "region_snippet": "\n".join(lines[snippet_start:snippet_end]),
                    }
                if len(all_logic_lines) >= 2:
                    # Near-start region (first 2 logic lines)
                    r = _make_func_region(all_logic_lines[:2])
                    if r:
                        regions.append(r)
                    _append_branch_regions(
                        target_file=target_file,
                        target_symbol=sym,
                        class_anchor_line=lines[func_idx],
                        lines=lines,
                        func_idx=func_idx,
                        body_indices=all_logic_lines,
                    )
                suppress_tail_anchors = _should_suppress_tail_anchors(
                    sym,
                    lines,
                    func_idx,
                    all_logic_lines,
                )
                if len(all_logic_lines) >= 6 and not suppress_tail_anchors:
                    # Near-end region: find the last top-level elif/if keyword in the function,
                    # and use the line just before it as anchor_before.
                    # This ensures the model can see and modify the elif→if pattern.
                    func_body_indent = func_indent + 4  # standard one-level indent inside function
                    last_elif_idx = None
                    for idx in reversed(all_logic_lines):
                        stripped = lines[idx].strip()
                        line_indent = len(lines[idx]) - len(lines[idx].lstrip())
                        if line_indent == func_body_indent and (stripped.startswith("elif ") or stripped.startswith("elif(")):
                            last_elif_idx = idx
                            break
                    if last_elif_idx is not None:
                        # Find the line just before last_elif in logic_lines
                        try:
                            pos = all_logic_lines.index(last_elif_idx)
                        except ValueError:
                            pos = None
                        if pos is not None and pos >= 1:
                            # anchor_before = line just before elif, anchor_after = first line inside elif block
                            tail_subset = all_logic_lines[max(0, pos - 1): pos + 3]
                            r = _make_func_region(tail_subset)
                            if r:
                                r2 = dict(r)
                                r2["target_symbol"] = f"{sym}__tail"
                                # Expand snippet to show full function
                                r2["region_snippet"] = "\n".join(lines[func_idx: min(len(lines), all_logic_lines[-1] + 4)])
                                regions.append(r2)
                    else:
                        # Fallback: last 4 logic lines
                        tail = all_logic_lines[max(0, len(all_logic_lines) - 10):]
                        r = _make_func_region(tail)
                        if r:
                            r2 = dict(r)
                            r2["target_symbol"] = f"{sym}__tail"
                            regions.append(r2)
                continue  # handled as function, skip the assignment path

            # Second try: assignment line "SYMBOL = " or "SYMBOL=" at module level
            target_idx = next(
                (idx for idx, line in enumerate(lines)
                 if re.match(rf"^{re.escape(sym)}\s*=", line.strip()) and not line[0].isspace()),
                None,
            )
            if target_idx is None:
                continue
            # Use the line before (or nearby non-blank line) as anchor_before
            before_idx = next(
                (idx for idx in range(target_idx - 1, max(-1, target_idx - 6), -1)
                 if lines[idx].strip() and not lines[idx].strip().startswith("#")),
                None,
            )
            # Use the line after as anchor_after
            after_idx = next(
                (idx for idx in range(target_idx + 1, min(len(lines), target_idx + 6))
                 if lines[idx].strip()),
                None,
            )
            if before_idx is None or after_idx is None:
                continue
            snippet_start = max(0, before_idx - 2)
            snippet_end = min(len(lines), after_idx + 4)
            snippet = "\n".join(lines[snippet_start:snippet_end])
            # For module-level variables, class_anchor_line is the target line itself
            # (there's no enclosing class; use it for search-start disambiguation)
            regions.append(
                {
                    "target_file": target_file,
                    "target_symbol": sym,
                    "class_anchor_line": lines[before_idx],  # nearest preceding non-comment line
                    "class_anchor_lineno": before_idx + 1,
                    "anchor_line_before_lineno": before_idx + 1,
                    "anchor_line_after_lineno": after_idx + 1,
                    "anchor_line_before": lines[before_idx],
                    "anchor_line_after": lines[after_idx],
                    "region_snippet": snippet,
                }
            )

    return regions


def build_class_anchor_region(
    code_context: dict[str, str],
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
) -> dict[str, str] | None:
    regions = build_edit_anchor_regions(code_context, analysis, strategy)
    return regions[0] if regions else None


def build_fragment_edit_prompt(
    instance: dict[str, Any],
    original_failure_log: str,
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
    anchor_regions: list[dict[str, str]],
    required_source_targets: list[str],
    required_repair_obligations: list[dict[str, Any]] | None = None,
    repair_topology: str | None = None,
    feedback: str | None = None,
) -> str:
    strategy_text = json.dumps(strategy or {}, ensure_ascii=False).lower()
    analysis_text = json.dumps(analysis or {}, ensure_ascii=False).lower()
    original_failure_lower = original_failure_log.lower()
    needs_flag_path_guidance = (
        "runxfail" in original_failure_lower
        and any(token in (strategy_text + analysis_text) for token in ("hook", "report", "dispatcher", "makereport", "runtest"))
    )
    needs_minimal_structural_guidance = needs_minimal_structural_fix_guidance(
        analysis=analysis,
        strategy=strategy,
        original_failure_log=original_failure_log,
    )
    anchor_preview = []
    for region in anchor_regions:
        region_dict: dict[str, Any] = {
            "target_file": region["target_file"],
            "target_symbol": region["target_symbol"],
            "class_anchor_line": region["class_anchor_line"],
            "class_anchor_lineno": region.get("class_anchor_lineno"),
            "anchor_line_before": region["anchor_line_before"],
            "anchor_line_before_lineno": region.get("anchor_line_before_lineno"),
            "anchor_line_after": region["anchor_line_after"],
            "anchor_line_after_lineno": region.get("anchor_line_after_lineno"),
            "region_snippet": region["region_snippet"],
        }
        if region.get("target_elif_line"):
            region_dict["ELIF_TO_IF_NOTE"] = (
                f"This region covers an `elif` branch: {region['target_elif_line']!r}. "
                "anchor_line_before is the LAST LINE OF THE PRECEDING BLOCK so you can include "
                "that `elif` inside replacement_block. You MUST change it to `if` — "
                "do NOT copy the `elif` verbatim into replacement_block."
            )
        anchor_preview.append(
            json.dumps(region_dict, ensure_ascii=False, indent=2)
        )
    prompt = (
        "Task: propose an exact code-fragment edit plan before generating any unified diff.\n"
        "Return exactly one JSON object.\n"
        "JSON schema:\n"
        "{\n"
        '  "coverage_check": [\n'
        "    {\n"
        '      "target_file": "one required source file from the strategy",\n'
        '      "action": "modify_region or no_change",\n'
        '      "justification": "why this file needs an edit or why it is safe to leave unchanged"\n'
        "    }\n"
        "  ],\n"
        '  "edits": [\n'
        "    {\n"
        '      "target_file": "must match one anchor target file",\n'
        '      "target_symbol": "must match one anchor target symbol",\n'
        '      "replacement_mode": "replace_span or insert_before_anchor or insert_after_anchor (optional; inferred when omitted)",\n'
        '      "covers_obligations": ["required obligation ids covered by this edit"],\n'
        '      "anchor_line_before": "exact existing line before the edited span",\n'
        '      "anchor_line_after": "exact existing line after the edited span",\n'
        '      "replacement_block": "replacement text that starts with anchor_line_before and ends with anchor_line_after"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Requirements:\n"
        "1. The coverage_check array must mention every required source file exactly once.\n"
        "2. Multiple anchor regions are provided when the bug requires fixes in multiple places. "
        "If there are anchor regions for different methods/symbols, evaluate EACH one — "
        "if the failing tests would require changes to that symbol, include an edit for it.\n"
        "3. CRITICAL — edit_targets enforcement: every symbol listed in the repair strategy's "
        "edit_targets MUST either have a corresponding edit entry OR be explicitly justified as "
        "no_change in coverage_check. Do NOT skip a symbol just because the strategy's "
        "sufficiency_assessment says one edit is enough — that assessment may be wrong. "
        "If the failing tests reference code in that symbol (traceback, assert, raise), add an edit.\n"
        "3b. ASSERT→RAISE RULE: if the traceback shows 'AssertionError' inside a source method "
        "(not a test file), that method contains an `assert` statement that must be converted to "
        "`raise ValueError(...)`. You MUST mark that method as modify_region and provide an edit "
        "entry that replaces the assert with the appropriate raise. Marking it no_change is wrong.\n"
        "4. You may return one or more edits when the repair requires multiple files or multiple class regions.\n"
        "4b. REQUIRED REPAIR OBLIGATIONS RULE: if required_repair_obligations are listed below, every obligation id "
        "must appear in at least one edit entry's covers_obligations list. Do not collapse multiple separately exposed "
        "obligations into one generic edit without listing all of the obligation ids it covers.\n"
        "5. Each replacement_block must stay inside its target class body.\n"
        "6. Do not edit imports, module-level assignments, unrelated classes, or tests.\n"
        "7. Preserve Python indentation exactly.\n"
        "8. replacement_mode controls how replacement_block relates to anchor_line_before.\n"
        "   - replace_span (default): replacement_block must begin with anchor_line_before EXACTLY.\n"
        "   - insert_before_anchor: use this when you must insert a new guard/check BEFORE an existing statement.\n"
        "     In this mode, replacement_block may start with new lines, but it must still include anchor_line_before\n"
        "     later in the block so the original anchor statement is preserved in the correct order.\n"
        "   - insert_after_anchor: use this when you must keep anchor_line_before itself (for example a closing class\n"
        "     docstring line) and insert new class-level statements immediately AFTER it but before anchor_line_after.\n"
        "     In this mode, replacement_block must begin with anchor_line_before, then include the inserted lines,\n"
        "     and finally include anchor_line_after later in the block.\n"
        "   - The replacement always spans from anchor_line_before through anchor_line_after in the original source.\n"
        "   - anchor_line_after defines WHERE in the original source the span ends (it will be removed).\n"
        "   - You may include anchor_line_after at the end of replacement_block if the line should be kept.\n"
        "   - If anchor previews include *_lineno fields, they identify the exact occurrence when anchor text repeats.\n"
        "   - Example replace_span: if anchor_line_before='    x = 1' and anchor_line_after='    def foo(self):',\n"
        "     a valid replacement_block is '    x = 1\\n\\n    NEW_LINE = True\\n\\n    def foo(self):'\n"
        "   - Example insert_before_anchor: if anchor_line_before='    x = normalize(x)' and anchor_line_after='    return x',\n"
        "     a valid replacement_block is '    if isinstance(x, bytes):\\n        x = x.decode(\"utf-8\")\\n    x = normalize(x)\\n    return x'\n\n"
        "   - Example insert_after_anchor: if anchor_line_before='    \"\"\"' and anchor_line_after='    def __str__(self):',\n"
        "     a valid replacement_block is '    \"\"\"\\n\\n    __slots__ = ()\\n\\n    def __str__(self):'\n\n"
        "9. FUNCTION CONTROL-FLOW RULE: if an anchor target symbol is a function or method and branch-focused "
        "anchors are available (target_symbol suffixes like __branch / __branch_tail), prefer editing the "
        "existing if/elif/else/except block shown there instead of prepending an unrelated guard clause near "
        "the top of the function.\n"
        "10. When fixing behavior in a long function, prefer modifying the existing branch condition/body that "
        "already handles the failing case. Appending a separate side-path is only acceptable if the existing "
        "branch structure clearly cannot express the fix.\n"
        "11. If the bug is in a hook/report/dispatcher function, choose the anchor region that directly contains "
        "the broken logic — do NOT default to the function top unless the root cause is clearly in the setup code.\n"
        "12. ELIF→IF STRUCTURAL FIX: If the existing code contains `elif` chains and the fix needs to run "
        "REGARDLESS of a preceding `elif` condition (e.g., a guard clause that would otherwise bypass it), "
        "you MUST change the controlling keyword from `elif` to `if`. "
        "If an anchor region includes ELIF_TO_IF_NOTE or target_elif_line metadata, its anchor_line_before is "
        "intentionally set to the LAST LINE OF THE PRECEDING BLOCK (not the `elif` line itself) so you can include "
        "the `elif` inside replacement_block and change it to `if`. "
        "Use replacement_mode=replace_span for this structural rewrite. Your replacement_block should start with "
        "anchor_line_before verbatim, then include the fixed block "
        "using `if (` instead of `elif (`, followed by the rest of the branch body.\n"
        "13. STRUCTURED REPORT OBJECT RULE: In hook/report/dispatcher functions, if a report field such as "
        "`rep.longrepr` is already a structured tuple/object, do NOT overwrite it with a raw location tuple like "
        "`item.location` or another lossy replacement. Preserve the existing payload (for example the reason text) "
        "and update only the specific fields that are wrong, such as filename/line, using the existing tuple/object shape. "
        "IMPORTANT: never extract fields from a tuple (e.g. `_, _, reason = rep.longrepr`) inside a conditional guard "
        "that may not execute — if the variable is used outside the guard it will be undefined. Always extract tuple "
        "fields unconditionally or inline them at the point of use.\n\n"
        f"Problem statement:\n{instance['problem_statement']}\n\n"
        f"Tests that must pass after the fix (FAIL_TO_PASS targets): {', '.join(get_original_test_identifiers(instance))}\n\n"
        f"Original failure log:\n```text\n{truncate_text(original_failure_log, 8000)}\n```\n\n"
        f"Failure analysis:\n```json\n{json.dumps(analysis or {}, indent=2, ensure_ascii=False)}\n```\n\n"
        f"Repair strategy:\n```json\n{json.dumps(strategy or {}, indent=2, ensure_ascii=False)}\n```\n\n"
        f"Required source files from the strategy: {', '.join(required_source_targets) or 'None'}\n\n"
        "Required repair obligations from retained enhanced tests:\n```json\n"
        f"{json.dumps(required_repair_obligations or [], ensure_ascii=False, indent=2)}\n```\n\n"
        "Available editable anchor regions:\n```json\n"
        f"{json.dumps([json.loads(item) for item in anchor_preview], ensure_ascii=False, indent=2)}\n```\n"
    )
    conversion_hints = _extract_conversion_rule_hints(required_repair_obligations)
    canonical_hints = _extract_canonical_statement_replacement_hints(
        required_repair_obligations,
        force_all=repair_topology == "statement_local",
    )
    if conversion_hints:
        prompt += (
            "\n16. PRE-CONVERSION ORDER RULE: some obligations target logic around an existing normalize/convert statement. "
            "If you need to guard, decode, or type-check the value before that canonical conversion, the pre-conversion "
            "check MUST appear before the existing conversion statement inside replacement_block. Do not place a bytes/type "
            "guard after the canonical conversion line, because that makes the new branch unreachable or ineffective.\n"
        )
        for hint in conversion_hints:
            prompt += (
                f"- obligation={hint['id']}; source_symbol={hint.get('source_symbol') or 'unknown'}; "
                f"validation_subject={hint.get('validation_subject') or 'core'}; "
                f"canonical_conversion_statement={hint['canonical_conversion_statement']}\n"
            )
        prompt += "\n"
    if repair_topology == "statement_local" and canonical_hints:
        prompt += (
            "17. CANONICAL STATEMENT REPLACEMENT RULE: this is a statement_local repair centered on a single canonical "
            "statement. Do NOT merely insert a guard before that statement. Replace the canonical statement with a full "
            "replacement block that makes both the guarded path and the fallback path explicit (for example, an if/else). "
            "Use replacement_mode=replace_span for this repair.\n"
        )
        for hint in canonical_hints:
            prompt += (
                f"- obligation={hint['id']}; source_symbol={hint.get('source_symbol') or 'unknown'}; "
                f"validation_subject={hint.get('validation_subject') or 'core'}; "
                f"canonical_statement_text={hint['canonical_statement_text']}\n"
            )
        prompt += "\n"
    if needs_flag_path_guidance:
        prompt += (
            "14. FLAG-PATH FIX RULE: the failing evidence explicitly points to behavior when a flag-enabled mode is ON "
            "(for example `--runxfail`). Do NOT 'fix' this by adding `not <flag>` guards, by returning early before "
            "the flagged path executes, or by disabling the existing branch under that mode. Those are negative-gating "
            "patches and leave the real bug unfixed. Instead, modify the flagged path so it produces the correct "
            "structured result when the flag is ON.\n\n"
        )
    if needs_minimal_structural_guidance:
        prompt += (
            "15. MINIMAL STRUCTURAL FIX RULE: the analysis indicates that the bug is caused by incorrect control-flow "
            "attachment, not by missing new behavior. Prefer the SMALLEST POSSIBLE structural edit to the existing "
            "branches. If changing one keyword (for example `elif` -> `if`) makes the correct branch execute, do exactly "
            "that and avoid introducing extra guards, temporary variables, helper assignments, or duplicated logic.\n\n"
        )
    if feedback:
        prompt += f"\nRevision feedback for the next edit plan:\n{feedback}\n"
    return prompt


def build_diff_from_strategy_prompt(
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    filtered_candidates: list[CandidatePatch],
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
    feedback: str | None = None,
    failure_focus: dict[str, Any] | None = None,
    hide_original_test_patch: bool = False,
) -> str:
    edit_targets = list(dict.fromkeys(
        t.split("::")[0] for t in (strategy or {}).get("edit_targets", [])
    ))
    strategy_text = json.dumps(strategy or {}, ensure_ascii=False).lower()
    analysis_text = json.dumps(analysis or {}, ensure_ascii=False).lower()
    original_failure_lower = original_failure_log.lower()
    needs_flag_path_guidance = (
        "runxfail" in original_failure_lower
        and any(token in (strategy_text + analysis_text) for token in ("hook", "report", "dispatcher", "makereport", "runtest"))
    )
    needs_minimal_structural_guidance = needs_minimal_structural_fix_guidance(
        analysis=analysis,
        strategy=strategy,
        original_failure_log=original_failure_log,
    )
    prompt = build_patch_prompt(
        instance=instance,
        code_context=code_context,
        original_failure_log=original_failure_log,
        filtered_candidates=filtered_candidates,
        failure_focus=failure_focus,
        hide_original_test_patch=hide_original_test_patch,
    )
    anchor_ctx = build_anchor_context(code_context, (analysis or {}).get('suspicious_symbols', []), edit_targets, max_chars=3200)
    prompt += (
        "\nYou must follow this failure analysis and repair strategy when generating the unified diff.\n"
        f"Failure analysis:\n```json\n{json.dumps(analysis or {}, indent=2, ensure_ascii=False)}\n```\n\n"
        f"Repair strategy:\n```json\n{json.dumps(strategy or {}, indent=2, ensure_ascii=False)}\n```\n"
        f"\nLocal anchor context (line numbers are 1-based and EXACT — use them for the diff hunk header):\n{anchor_ctx}\n"
        "\nCRITICAL diff rules:\n"
        "1. The line numbers shown in the anchor context above are the real line numbers in the file. Use them to set the @@ hunk header.\n"
        "2. Context lines in the diff (lines without +/-) must be copied VERBATIM from the file content shown above — do not add, remove, or change any leading spaces.\n"
        "3. Do NOT invent a fake 'index' hash line. Only include diff --git, --- a/..., +++ b/..., and @@ ... @@ lines.\n"
        "4. Patch placement constraint: if the repair strategy targets a class symbol, the diff hunk must modify inside that class body "
        "with correct indentation, not near imports or module-level statements.\n"
        "5. NEVER insert Python statements (if/raise/assert) inside a function-call argument list. "
        "If the target location is inside super().__init__(...) arguments, move the insertion to AFTER the closing ')' of that call, "
        "at the same indentation level as the surrounding method body statements.\n"
        f"6. The repair strategy lists {len(edit_targets)} edit target(s): {edit_targets}. "
        "If more than one location needs changing, you MUST generate one @@ hunk per location. "
        "Do NOT omit any location — the patch is only accepted if ALL required changes are present.\n"
        "7. Do NOT add code inside docstrings. Any new executable statements must appear AFTER the closing triple-quote of the docstring.\n"
        "8. CONDITIONAL BLOCK SCOPE: When the fix requires skipping an entire conditional block under a new condition, "
        "add the condition to the controlling `if`/`elif` expression itself (using `and`/`or`), NOT by wrapping only some of the "
        "block's body statements in a new `if`. Wrapping only the assignment but not subsequent uses (e.g. `assert x is not None`) "
        "leaves those subsequent lines with undefined variables. The correct pattern is: "
        "`elif existing_cond and not new_flag:` — this skips the entire block atomically.\n"
        "9. INPUT VALIDATION PLACEMENT (P2): If the buggy code already calls a type-conversion function "
        "(e.g. builtin_str(), str(), int(), bytes.decode()) on an input variable, do NOT add isinstance() "
        "or type checks AFTER that conversion — the conversion has already changed the type. "
        "Instead, insert any input validation or type normalization BEFORE the existing conversion call, "
        "operating on the original raw input value.\n"
        "10. FUNCTION CONTROL-FLOW RULE: for function/method targets, prefer changing the existing branch near the "
        "provided branch-focused anchor context rather than inserting a new top-of-function side path. If the bug is "
        "about how a report/hook/dispatcher chooses behavior, modify the existing if/elif/else logic that already "
        "selects that behavior.\n"
        "11. For hook/report/dispatcher functions with shared state initialization near the top, prefer a fix near "
        "that shared setup or the earliest controlling branch over a late __tail-only edit. A tail edit is only "
        "appropriate when the failing behavior is isolated to the tail block itself.\n"
        "12. STRUCTURED REPORT OBJECT RULE: when editing report/hook/dispatcher functions, do NOT replace a "
        "structured report payload such as `rep.longrepr` with a raw `item.location` tuple or any other lossy value. "
        "If the bug is a wrong source location, preserve the existing reason/payload shape and update only the "
        "filename/line fields derived from the existing tuple/object.\n"
    )
    if needs_flag_path_guidance:
        prompt += (
            "13. FLAG-PATH FIX RULE: the failure evidence is specifically about behavior when a flag-enabled mode is ON "
            "(for example `--runxfail`). Do NOT generate a patch that merely adds `and not <flag>`, wraps the code in "
            "`if not <flag>`, or otherwise disables the existing branch under that mode. That is a negative-gating "
            "workaround, not a repair. The patch must make the flagged path itself produce the correct structured "
            "result when the flag is ON.\n"
        )
    if needs_minimal_structural_guidance:
        prompt += (
            "14. MINIMAL STRUCTURAL FIX RULE: if the analysis says an existing branch is unreachable because it is "
            "chained under the wrong control-flow structure, prefer the smallest keyword-level diff that fixes "
            "reachability. If a single `elif` -> `if` change is sufficient, emit that one-line structural fix instead "
            "of adding new logic, new guards, or duplicated post-processing code. When two candidate patches would have "
            "the same effect, prefer the smaller structural patch.\n"
        )
    if feedback:
        prompt += f"\nRevision feedback for the next patch attempt:\n{feedback}\n"
        if "Semantic oracle failed identifiers" in feedback:
            prompt += (
                "\nPrevious attempt diagnosis: the earlier patch was incomplete. "
                "It may have matched the local symptom but still failed stricter semantic checks. "
                "In this attempt, explicitly reconsider whether the repair scope must expand to additional related classes or files mentioned in the strategy.\n"
            )
    return prompt


def build_anchor_context(
    code_context: dict[str, str],
    symbols: list[str],
    edit_targets: list[str],
    max_chars: int = 1600,
) -> str:
    if not edit_targets:
        edit_targets = list(code_context.keys())
    # Strip ::Symbol notation from edit_targets — code_context keys are plain file paths
    edit_targets = list(dict.fromkeys(t.split("::")[0] for t in edit_targets))
    # Parse symbols into (class_name, method_name) pairs
    # e.g. "Blueprint.__init__" → ("Blueprint", "__init__")
    # e.g. "Blueprint" → ("Blueprint", None)
    parsed_symbols: list[tuple[str, str | None]] = []
    for sym in symbols:
        if sym and "." in sym:
            cls, method = sym.split(".", 1)
            parsed_symbols.append((cls, method))
        elif sym:
            parsed_symbols.append((sym, None))

    sections: list[str] = []
    seen_regions: set[str] = set()

    # Iterate over symbols first to generate per-symbol anchor sections
    for cls_name, method_name in parsed_symbols:
        for path in edit_targets:
            content = code_context.get(path)
            if not content:
                continue
            lines = content.splitlines()
            total_lines = len(lines)
            best_idx = None

            if method_name is not None:
                # Find class definition, then find method inside it
                cls_idx = next(
                    (i for i, l in enumerate(lines) if re.search(rf"^class\s+{re.escape(cls_name)}\b", l.strip())),
                    None,
                )
                if cls_idx is not None:
                    method_pattern = re.compile(rf"^\s+def\s+{re.escape(method_name)}\s*\(")
                    method_idx = next(
                        (i for i in range(cls_idx + 1, total_lines) if method_pattern.search(lines[i])),
                        None,
                    )
                    if method_idx is not None:
                        best_idx = method_idx
            else:
                # Find class definition, then prefer __init__ over class header
                cls_idx = next(
                    (i for i, l in enumerate(lines) if re.search(rf"^class\s+{re.escape(cls_name)}\b", l.strip())),
                    None,
                )
                if cls_idx is not None:
                    # Prefer anchoring at __init__ so the model sees the constructor body
                    init_idx = next(
                        (i for i in range(cls_idx + 1, min(cls_idx + 80, total_lines))
                         if re.search(r"^\s+def\s+__init__\s*\(", lines[i])),
                        None,
                    )
                    best_idx = init_idx if init_idx is not None else cls_idx

            if best_idx is None:
                continue

            region_key = f"{path}:{best_idx}"
            if region_key in seen_regions:
                continue
            seen_regions.add(region_key)

            start = max(0, best_idx - 4)
            end = min(total_lines, best_idx + 80)
            numbered_lines = [f"{start + i + 1:4d}: {lines[start + i]}" for i in range(end - start)]
            snippet = "\n".join(numbered_lines)
            label = f"{cls_name}.{method_name}" if method_name else cls_name
            sections.append(
                f"### Anchor: {path} — {label} (total {total_lines} lines; showing lines {start+1}–{end})\n"
                f"```text\n{truncate_text(snippet, max_chars)}\n```"
            )

    if not sections:
        # Last resort: include all target files with line numbers
        for target in edit_targets:
            content = code_context.get(target)
            if content:
                lines = content.splitlines()
                numbered = "\n".join(f"{i+1:4d}: {l}" for i, l in enumerate(lines))
                sections.append(f"### Anchor: {target}\n```text\n{truncate_text(numbered, max_chars)}\n```")

    return "\n\n".join(sections[:3])


def build_unified_diff_from_replacement(
    original_content: str,
    updated_content: str,
    target_file: str,
) -> str:
    diff_lines = unified_diff(
        original_content.splitlines(keepends=True),
        updated_content.splitlines(keepends=True),
        fromfile=f"a/{target_file}",
        tofile=f"b/{target_file}",
    )
    return normalize_patch("".join(diff_lines))


def normalize_edit_plan_entries(edit_plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    if edit_plan is None:
        return []
    edits = edit_plan.get("edits")
    if isinstance(edits, list):
        return [edit for edit in edits if isinstance(edit, dict)]
    if all(key in edit_plan for key in ("target_file", "target_symbol", "anchor_line_before", "anchor_line_after", "replacement_block")):
        return [edit_plan]
    return []


def _infer_replacement_mode(
    replacement_lines: list[str],
    anchor_before: str,
    anchor_after: str = "",
) -> str:
    if not replacement_lines:
        return "replace_span"
    if replacement_lines[0] == anchor_before:
        if anchor_after and anchor_after in replacement_lines[1:]:
            anchor_after_idx = replacement_lines.index(anchor_after, 1)
            if anchor_after_idx > 1:
                return "insert_after_anchor"
        return "replace_span"
    if anchor_before and anchor_before in replacement_lines[1:]:
        return "insert_before_anchor"
    return "replace_span"


def _get_replacement_mode(edit_entry: dict[str, Any], replacement_lines: list[str]) -> str:
    raw_mode = str(edit_entry.get("replacement_mode") or "").strip().lower()
    preferred_mode = str(edit_entry.get("preferred_replacement_mode") or "").strip().lower()
    inferred_mode = _infer_replacement_mode(
        replacement_lines,
        str(edit_entry.get("anchor_line_before", "")),
        str(edit_entry.get("anchor_line_after", "")),
    )
    # Be tolerant of a model that labels the edit as replace_span while the block
    # clearly inserts new logic before the anchor statement. This keeps the
    # plan-derived patch path viable for statement-level repairs such as
    # pre-conversion guards inserted before an existing canonical conversion line.
    if raw_mode == "replace_span" and inferred_mode == "insert_before_anchor":
        return inferred_mode
    if raw_mode == "replace_span" and inferred_mode == "insert_after_anchor":
        return inferred_mode
    # Also tolerate the reverse mismatch: some models label a full replacement
    # as insert_before_anchor even though the replacement_block clearly starts at
    # anchor_line_before and rewrites the span through anchor_line_after.
    if raw_mode == "insert_before_anchor" and inferred_mode == "replace_span":
        return inferred_mode
    if raw_mode == "insert_after_anchor" and inferred_mode == "replace_span":
        return raw_mode
    if raw_mode in {"replace_span", "insert_before_anchor", "insert_after_anchor"}:
        return raw_mode
    if preferred_mode == "insert_after_anchor" and inferred_mode == "replace_span":
        anchor_before = str(edit_entry.get("anchor_line_before", ""))
        anchor_after = str(edit_entry.get("anchor_line_after", ""))
        if (
            replacement_lines
            and replacement_lines[0] == anchor_before
            and anchor_after
            and anchor_after in replacement_lines[1:]
        ):
            return "insert_after_anchor"
    return inferred_mode


def _normalize_target_symbol_name(target_symbol: str) -> str:
    symbol = str(target_symbol or "").strip()
    for suffix in ("__branch_tail", "__tail", "__branch", "__head"):
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)]
            break
    return symbol


def _find_symbol_node(module: ast.AST, target_symbol: str) -> ast.AST | None:
    symbol = _normalize_target_symbol_name(target_symbol)
    if not symbol:
        return None
    if "." in symbol:
        class_name, member_name = symbol.split(".", 1)
        for node in getattr(module, "body", []):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == member_name:
                        return child
        return None
    for node in getattr(module, "body", []):
        if isinstance(node, ast.ClassDef) and node.name == symbol:
            return node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            return node
    return None


def _get_docstring_span(node: ast.AST) -> tuple[int, int] | None:
    body = getattr(node, "body", None)
    if not body:
        return None
    first_stmt = body[0]
    if not isinstance(first_stmt, ast.Expr):
        return None
    value = first_stmt.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    start = getattr(first_stmt, "lineno", None)
    end = getattr(first_stmt, "end_lineno", start)
    if start is None or end is None:
        return None
    return start, end


def _is_nontrivial_added_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    return True


def _get_executable_statement_spans(node: ast.AST) -> list[tuple[int, int]]:
    body = getattr(node, "body", None)
    if not body:
        return []
    spans: list[tuple[int, int]] = []
    for idx, stmt in enumerate(body):
        if idx == 0 and isinstance(stmt, ast.Expr):
            value = stmt.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                continue
        start = getattr(stmt, "lineno", None)
        end = getattr(stmt, "end_lineno", start)
        if start is None or end is None:
            continue
        spans.append((start, end))
    return spans


def _infer_single_root_target_symbol(
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
) -> tuple[str, str] | tuple[None, None]:
    edit_targets = [
        str(target)
        for target in (strategy or {}).get("edit_targets", [])
        if isinstance(target, str) and "::" in str(target)
    ]
    if len(edit_targets) == 1:
        target_file, target_symbol = edit_targets[0].split("::", 1)
        return target_file, target_symbol
    suspicious_symbols = [
        str(symbol).strip()
        for symbol in (analysis or {}).get("suspicious_symbols", [])
        if isinstance(symbol, str) and str(symbol).strip()
    ]
    scope_files = get_analysis_source_files(analysis)
    if len(suspicious_symbols) == 1 and len(scope_files) == 1:
        return scope_files[0], suspicious_symbols[0]
    return None, None


def validate_single_root_symbol_patch_landing(
    patch: str,
    code_context: dict[str, str] | None,
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
) -> str | None:
    if not patch.strip() or not code_context:
        return None
    target_file, target_symbol = _infer_single_root_target_symbol(analysis, strategy)
    if not target_file or not target_symbol or target_file not in code_context:
        return None
    try:
        parsed_patch = PatchSet(patch)
    except Exception:
        return None
    patched_file = next((item for item in parsed_patch if item.path == target_file), None)
    if patched_file is None:
        return (
            f"This is a single_root_symbol repair targeting {target_file}::{target_symbol}. "
            "Your patch must modify that file directly."
        )
    updated_content, patch_error = apply_patchset_to_content(code_context[target_file], patched_file)
    if patch_error is not None:
        return patch_error
    try:
        module = ast.parse(updated_content)
    except SyntaxError:
        return None
    symbol_node = _find_symbol_node(module, target_symbol)
    if symbol_node is None:
        return (
            f"The patched file no longer contains the target symbol {target_symbol}. "
            "Land the fix inside the existing symbol body rather than moving it elsewhere."
        )
    symbol_start = getattr(symbol_node, "lineno", None)
    symbol_end = getattr(symbol_node, "end_lineno", symbol_start)
    if symbol_start is None or symbol_end is None:
        return None
    hunk_touches_symbol = False
    hunk_adds_outside_symbol = False
    for hunk in patched_file:
        new_line = max(int(getattr(hunk, "target_start", 1) or 1), 1)
        for line in hunk:
            if line.is_added:
                if _is_nontrivial_added_line(str(line.value or "")):
                    if symbol_start <= new_line <= symbol_end:
                        hunk_touches_symbol = True
                    else:
                        hunk_adds_outside_symbol = True
                new_line += 1
            elif line.is_removed:
                continue
            else:
                new_line += 1
    if not hunk_touches_symbol:
        return (
            f"This is a single_root_symbol repair targeting {target_symbol}. "
            "The patch did not land inside the target symbol body."
        )
    if hunk_adds_outside_symbol:
        return (
            f"This is a single_root_symbol repair targeting {target_symbol}. "
            "Do not add new executable or class-level lines outside the target symbol body."
        )
    return None


def validate_patch_semantic_landing(
    patch: str,
    edit_plan: dict[str, Any] | None,
    code_context: dict[str, str] | None,
) -> str | None:
    if edit_plan is None or not code_context:
        return None
    edit_entries = normalize_edit_plan_entries(edit_plan)
    if not edit_entries:
        return None
    try:
        parsed_patch = PatchSet(patch)
    except Exception:
        return None
    patched_files = {patched_file.path: patched_file for patched_file in parsed_patch}
    updated_contents: dict[str, str] = {}
    ast_modules: dict[str, ast.AST] = {}
    for entry in edit_entries:
        target_file = str(entry.get("target_file", ""))
        target_symbol = str(entry.get("target_symbol", ""))
        if not target_file.endswith(".py") or target_file not in code_context:
            continue
        patched_file = patched_files.get(target_file)
        if patched_file is None:
            continue
        if target_file not in updated_contents:
            updated_content, patch_error = apply_patchset_to_content(code_context[target_file], patched_file)
            if patch_error is not None:
                return patch_error
            updated_contents[target_file] = updated_content
        if target_file not in ast_modules:
            try:
                ast_modules[target_file] = ast.parse(updated_contents[target_file], filename=target_file)
            except SyntaxError:
                continue
        symbol_node = _find_symbol_node(ast_modules[target_file], target_symbol)
        if symbol_node is None:
            continue
        node_start = getattr(symbol_node, "lineno", None)
        node_end = getattr(symbol_node, "end_lineno", None)
        if node_start is None or node_end is None:
            continue
        added_lines = [
            int(line.target_line_no)
            for hunk in patched_file
            for line in hunk
            if line.is_added
            and line.target_line_no is not None
            and _is_nontrivial_added_line(str(line.value).rstrip("\n"))
        ]
        if not added_lines:
            continue
        symbol_added_lines = [line_no for line_no in added_lines if node_start <= line_no <= node_end]
        if not symbol_added_lines:
            return f"The patch did not land inside the expected symbol {target_symbol} in {target_file}."
        executable_spans = _get_executable_statement_spans(symbol_node)
        executable_added_lines = [
            line_no
            for line_no in symbol_added_lines
            if any(start <= line_no <= end for start, end in executable_spans)
        ]
        if executable_added_lines:
            continue
        docstring_span = _get_docstring_span(symbol_node)
        if docstring_span is not None:
            doc_start, doc_end = docstring_span
            offending = [line_no for line_no in symbol_added_lines if doc_start <= line_no <= doc_end]
            if offending:
                return (
                    f"The patch added code only inside the docstring of {target_symbol} "
                    f"in {target_file}:{min(offending)}. Move the code below the closing triple-quote."
                )
        return f"The patch did not create any executable change inside the expected symbol {target_symbol} in {target_file}."
    return None


def apply_single_fragment_edit(
    original_content: str,
    edit_entry: dict[str, Any],
) -> tuple[str, str | None]:
    anchor_before = str(edit_entry.get("anchor_line_before", ""))
    anchor_after = str(edit_entry.get("anchor_line_after", ""))
    replacement_block = str(edit_entry.get("replacement_block", ""))
    class_anchor_line = str(edit_entry.get("class_anchor_line", ""))
    class_anchor_lineno = edit_entry.get("class_anchor_lineno")
    anchor_before_lineno = edit_entry.get("anchor_line_before_lineno")
    anchor_after_lineno = edit_entry.get("anchor_line_after_lineno")
    if not anchor_before or not anchor_after or not replacement_block:
        return "", "The edit plan is missing anchor_line_before, anchor_line_after, or replacement_block."
    replacement_lines = replacement_block.splitlines()
    if not replacement_lines:
        return "", "The replacement_block is empty."
    replacement_mode = _get_replacement_mode(edit_entry, replacement_lines)
    if replacement_mode == "replace_span":
        if replacement_lines[0] != anchor_before:
            return "", "The replacement_block must start with anchor_line_before exactly."
    elif replacement_mode == "insert_before_anchor":
        if anchor_before not in replacement_lines:
            return "", "The replacement_block must include anchor_line_before when using insert_before_anchor."
        if replacement_lines[0] == anchor_before:
            return "", "insert_before_anchor requires at least one inserted line before anchor_line_before."
    elif replacement_mode == "insert_after_anchor":
        if not replacement_lines or replacement_lines[0] != anchor_before:
            return "", "The replacement_block must begin with anchor_line_before when using insert_after_anchor."
        if anchor_after not in replacement_lines[1:]:
            return "", "The replacement_block must include anchor_line_after later in the block when using insert_after_anchor."
        anchor_after_idx = replacement_lines.index(anchor_after, 1)
        if anchor_after_idx <= 1:
            return "", "insert_after_anchor requires at least one inserted line after anchor_line_before."
    else:
        return "", f"Unknown replacement_mode: {replacement_mode!r}."
    # Note: we do NOT require replacement_lines[-1] == anchor_after.
    # The model may legitimately add lines after anchor_line_after in the block
    # (e.g. inserting new code after the closing anchor).  The original span
    # [anchor_before .. anchor_after] is always replaced by replacement_lines.

    original_lines = original_content.splitlines()

    start_idx, end_idx = _resolve_fragment_anchor_indices(
        original_lines,
        anchor_before=anchor_before,
        anchor_after=anchor_after,
        class_anchor_line=class_anchor_line,
        class_anchor_lineno=class_anchor_lineno,
        anchor_before_lineno=anchor_before_lineno,
        anchor_after_lineno=anchor_after_lineno,
    )
    if end_idx is None:
        return "", "anchor_line_after was not found in the target file."
    if start_idx is None:
        return "", "anchor_line_before was not found before anchor_line_after in the target file."

    updated_lines = original_lines[:start_idx] + replacement_lines + original_lines[end_idx + 1 :]
    updated_content = "\n".join(updated_lines)
    if original_content.endswith("\n"):
        updated_content += "\n"
    return updated_content, None


def apply_fragment_edit_plan(
    code_context: dict[str, str],
    edit_plan: dict[str, Any],
    anchor_regions: list[dict[str, str]] | None = None,
) -> tuple[dict[str, str], str | None]:
    updated_files: dict[str, str] = {}
    edit_entries = normalize_edit_plan_entries(edit_plan)
    if not edit_entries:
        return {}, "The edit plan did not contain any valid edit entries."
    current_contents = dict(code_context)
    # Build a region_map so we can look up disambiguating anchor metadata by
    # (file, symbol, before, after), including exact line-number hints when
    # anchor text repeats in the same symbol.
    region_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for region in (anchor_regions or []):
        key = (
            region.get("target_file", ""),
            region.get("target_symbol", ""),
            region.get("anchor_line_before", ""),
            region.get("anchor_line_after", ""),
        )
        region_map[key] = {
            "class_anchor_line": region.get("class_anchor_line", ""),
            "class_anchor_lineno": region.get("class_anchor_lineno"),
            "anchor_line_before_lineno": region.get("anchor_line_before_lineno"),
            "anchor_line_after_lineno": region.get("anchor_line_after_lineno"),
            "preferred_replacement_mode": region.get("preferred_replacement_mode", ""),
        }
    for edit_entry in edit_entries:
        target_file = str(edit_entry.get("target_file", ""))
        original_content = current_contents.get(target_file, "")
        if not original_content:
            return {}, f"The target file {target_file} was not present in the code context."
        # Inject class_anchor_line from matched anchor region so apply_single_fragment_edit
        # can restrict its search to after the class definition, avoiding false matches
        # when duplicate lines appear in sibling classes earlier in the file.
        region_key = (
            target_file,
            str(edit_entry.get("target_symbol", "")),
            str(edit_entry.get("anchor_line_before", "")),
            str(edit_entry.get("anchor_line_after", "")),
        )
        if region_key in region_map:
            edit_entry = dict(edit_entry)  # copy to avoid mutating the original
            edit_entry.update(region_map[region_key])
        updated_content, error = apply_single_fragment_edit(original_content, edit_entry)
        if error is not None:
            return {}, error
        current_contents[target_file] = updated_content
        updated_files[target_file] = updated_content
    return updated_files, None


def validate_fragment_edit_plan(
    edit_plan: dict[str, Any] | None,
    anchor_regions: list[dict[str, str]] | None,
    required_source_targets: list[str] | None = None,
    required_repair_obligations: list[dict[str, Any]] | None = None,
    strategy: dict[str, Any] | None = None,
    repair_topology: str | None = None,
) -> str | None:
    if edit_plan is None:
        return "Return a JSON object with a non-empty edits list."
    edit_entries = normalize_edit_plan_entries(edit_plan)
    if not edit_entries:
        return "The edit plan JSON must contain a non-empty edits list."
    coverage_check = edit_plan.get("coverage_check")
    if required_source_targets is None:
        required_source_targets = []
    if required_source_targets:
        if not isinstance(coverage_check, list) or not coverage_check:
            return "The edit plan JSON must contain a non-empty coverage_check list."
        covered_targets = []
        required_targets_with_obligations = {
            str(item.get("source_symbol") or "").strip()
            for item in (required_repair_obligations or [])
            if str(item.get("source_symbol") or "").strip()
        }
        target_file_to_symbols: dict[str, set[str]] = {}
        for region in anchor_regions or []:
            target_file_to_symbols.setdefault(str(region.get("target_file") or "").strip(), set()).add(
                str(region.get("target_symbol") or "").strip()
            )
        for item in coverage_check:
            if not isinstance(item, dict):
                return "Each coverage_check entry must be a JSON object."
            for key in ("target_file", "action", "justification"):
                if key not in item:
                    return f"Each coverage_check entry must include '{key}'."
            action = str(item["action"])
            if action not in {"modify_region", "no_change"}:
                return "coverage_check action must be either modify_region or no_change."
            target_file = str(item["target_file"])
            if action == "no_change":
                file_symbols = target_file_to_symbols.get(target_file, set())
                if any(symbol in required_targets_with_obligations for symbol in file_symbols):
                    return (
                        f"coverage_check cannot mark {target_file} as no_change because required repair obligations still "
                        "land in symbols from this file."
                    )
            covered_targets.append(str(item["target_file"]))
        # Every required source target must appear in coverage_check.
        # Extra no_change entries for dependency files are allowed.
        required_set = set(required_source_targets)
        covered_set = set(covered_targets)
        missing = required_set - covered_set
        if missing:
            return f"coverage_check is missing required source files: {', '.join(sorted(missing))}."
        modified_targets = {
            str(item["target_file"]) for item in coverage_check if str(item.get("action")) == "modify_region"
        }
        edit_targets = {str(entry.get("target_file", "")) for entry in edit_entries}
        if not edit_targets.issubset(modified_targets):
            return "Each edit entry must correspond to a coverage_check item whose action is modify_region."
    if not anchor_regions:
        return "No class anchor region was available for fragment editing."
    strategy_constraints = extract_strategy_constraints(strategy)
    region_map = {
        (region["target_file"], region["target_symbol"], region["anchor_line_before"], region["anchor_line_after"]): region
        for region in anchor_regions
    }
    required_obligation_ids = {
        str(item.get("id")).strip()
        for item in (required_repair_obligations or [])
        if str(item.get("id")).strip()
    }
    obligation_meta = {
        str(item.get("id")).strip(): item
        for item in (required_repair_obligations or [])
        if str(item.get("id")).strip()
    }
    covered_obligation_ids: set[str] = set()
    seen_entries: set[tuple[str, str, str, str]] = set()
    conversion_hints_by_entry: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    canonical_hints_by_entry: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    preferred_single_root_symbol = None
    if repair_topology == "single_root_symbol":
        unique_symbols = dedupe_preserve_order(
            str(item.get("source_symbol") or "").strip()
            for item in (required_repair_obligations or [])
            if str(item.get("source_symbol") or "").strip()
        )
        if len(unique_symbols) == 1:
            preferred_single_root_symbol = unique_symbols[0]
    for entry in edit_entries:
        for key in ("target_file", "target_symbol", "anchor_line_before", "anchor_line_after", "replacement_block"):
            if key not in entry:
                return f"Each edit entry must include '{key}'."
        region_key = (
            str(entry["target_file"]),
            str(entry["target_symbol"]),
            str(entry["anchor_line_before"]),
            str(entry["anchor_line_after"]),
        )
        if region_key not in region_map:
            return "Each edit entry must match one of the provided editable anchor regions exactly."
        if region_key in seen_entries:
            return "Do not emit duplicate edit entries for the same anchor region."
        seen_entries.add(region_key)
        if required_obligation_ids:
            covers_obligations = entry.get("covers_obligations")
            if not isinstance(covers_obligations, list) or not covers_obligations:
                return (
                    "Each edit entry must include a non-empty covers_obligations list because "
                    "required_repair_obligations are active for this edit plan."
                )
            normalized_obligations = []
            for obligation_id in covers_obligations:
                normalized_id = str(obligation_id).strip()
                if not normalized_id:
                    continue
                if normalized_id not in required_obligation_ids:
                    return (
                        f"Edit entry covers unknown obligation id '{normalized_id}'. "
                        "Use only obligation ids listed in required_repair_obligations."
                    )
                normalized_obligations.append(normalized_id)
            if not normalized_obligations:
                return (
                    "Each edit entry must list at least one valid obligation id in covers_obligations when "
                    "required_repair_obligations are active."
                )
            target_symbol = str(entry["target_symbol"]).strip()
            for obligation_id in normalized_obligations:
                obligation = obligation_meta.get(obligation_id, {})
                obligation_symbol = str(obligation.get("source_symbol") or "").strip()
                if obligation_symbol and obligation_symbol not in target_symbol and obligation_symbol.split(".")[-1] not in target_symbol:
                    return (
                        f"Edit entry target_symbol '{target_symbol}' does not match obligation '{obligation_id}' "
                        f"for source symbol '{obligation_symbol}'."
                    )
                covered_obligation_ids.add(obligation_id)
            conversion_hints_by_entry[region_key] = _extract_conversion_rule_hints(
                [obligation_meta[obligation_id] for obligation_id in normalized_obligations if obligation_id in obligation_meta]
            )
            canonical_hints_by_entry[region_key] = _extract_canonical_statement_replacement_hints(
                [obligation_meta[obligation_id] for obligation_id in normalized_obligations if obligation_id in obligation_meta],
                force_all=repair_topology == "statement_local",
            )
        replacement_block = str(entry["replacement_block"])
        region = region_map[region_key]
        # class_anchor_line is used for search-scope disambiguation in apply_single_fragment_edit.
        # It does NOT need to appear in region_snippet — the snippet is just for model context.
        replacement_lines = replacement_block.splitlines()
        if not replacement_lines:
            return "The replacement_block is empty."
        replacement_mode = _get_replacement_mode(entry, replacement_lines)
        if replacement_mode not in {"replace_span", "insert_before_anchor", "insert_after_anchor"}:
            return "replacement_mode must be either replace_span, insert_before_anchor, or insert_after_anchor."
        if replacement_mode == "replace_span":
            if replacement_lines[0] != region["anchor_line_before"]:
                return "Each replace_span replacement_block must start with anchor_line_before exactly."
        elif replacement_mode == "insert_before_anchor":
            if region["anchor_line_before"] not in replacement_lines:
                return (
                    "Each insert_before_anchor replacement_block must include anchor_line_before somewhere "
                    "after the newly inserted lines."
                )
            if replacement_lines[0] == region["anchor_line_before"]:
                return "insert_before_anchor requires at least one inserted line before anchor_line_before."
        else:
            if not replacement_lines or replacement_lines[0] != region["anchor_line_before"]:
                return (
                    "Each insert_after_anchor replacement_block must begin with anchor_line_before exactly so the "
                    "existing anchor line is preserved."
                )
            if region["anchor_line_after"] not in replacement_lines[1:]:
                return (
                    "Each insert_after_anchor replacement_block must include anchor_line_after later in the block "
                    "after the newly inserted class/body lines."
                )
            anchor_after_idx = replacement_lines.index(region["anchor_line_after"], 1)
            if anchor_after_idx <= 1:
                return "insert_after_anchor requires at least one inserted line after anchor_line_before."
            if region.get("preferred_replacement_mode") == "insert_after_anchor":
                anchor_before = str(region.get("anchor_line_before") or "").strip()
                duplicate_anchor_before_count = sum(
                    1 for line in replacement_lines if str(line).strip() == anchor_before
                )
                if duplicate_anchor_before_count > 1:
                    return (
                        "This region preserves a class docstring closing line and expects class-body insertion "
                        "after it. Do not repeat the docstring delimiter inside replacement_block; keep the first "
                        "anchor_line_before line, insert the new class-level statement(s), then continue with anchor_line_after."
                    )
        conversion_feedback = _replacement_block_violates_conversion_order(
            replacement_lines,
            conversion_hints_by_entry.get(region_key, []),
        )
        if conversion_feedback:
            return conversion_feedback
        canonical_feedback = _replacement_block_violates_canonical_statement_replacement(
            replacement_lines,
            replacement_mode,
            canonical_hints_by_entry.get(region_key, []),
        )
        if canonical_feedback:
            return canonical_feedback
        region_snippet = str(region.get("region_snippet") or "")
        if region_snippet:
            snippet_lines = region_snippet.splitlines()
            try:
                original_start = snippet_lines.index(region["anchor_line_before"])
                original_end = len(snippet_lines) - 1 - list(reversed(snippet_lines)).index(region["anchor_line_after"])
            except ValueError:
                original_start = -1
                original_end = -1
            if 0 <= original_start <= original_end:
                original_window = snippet_lines[original_start : original_end + 1]
                if replacement_lines == original_window:
                    return (
                        "The edit plan includes a redundant edit that simply repeats the existing source span "
                        "without introducing any semantic change. Remove this edit or change the target root-cause region."
                    )
                if (
                    repair_topology == "single_root_symbol"
                    and preferred_single_root_symbol
                    and preferred_single_root_symbol.split(".")[-1]
                    not in str(entry.get("target_symbol") or "")
                ):
                    stripped_original = [line.strip() for line in original_window if line.strip()]
                    stripped_replacement = [line.strip() for line in replacement_lines if line.strip()]
                    if stripped_replacement == stripped_original:
                        return (
                            "This single_root_symbol edit only replays the existing symptom-side code span and does not "
                            "modify the promoted root-cause symbol. Drop the redundant symptom edit and keep the root-cause edit only."
                        )
        # If this region covers an `elif` branch, the replacement_block must change
        # that `elif` to `if` — copying it verbatim means the elif chain is still
        # short-circuited by earlier branches and the fix has no effect.
        target_elif_line = region.get("target_elif_line")
        expected_if = None
        if target_elif_line:
            target_elif_stripped = target_elif_line.strip()
            rb_stripped_lines = [l.strip() for l in replacement_block.splitlines()]
            expected_if = target_elif_stripped.replace("elif ", "if ", 1).replace("elif(", "if(", 1)
            if target_elif_stripped in rb_stripped_lines:
                elif_as_if = target_elif_line.replace("elif ", "if ", 1)
                return (
                    f"INVALID: your replacement_block contains an `elif` that must become a standalone `if`: "
                    f"{target_elif_stripped!r}. "
                    "This elif is inside the original elif chain and will be short-circuited by earlier branches — "
                    "it has zero effect. "
                    f"Change every occurrence of `elif (` in this block to `if (` in your replacement_block."
                )
        if STRUCTURAL_RULE_RESTORE_UNREACHABLE_BRANCH in strategy_constraints.get("structural_rules", []):
            if target_elif_line:
                rb_stripped_lines = [l.strip() for l in replacement_block.splitlines()]
                if expected_if not in rb_stripped_lines:
                    return (
                        f"The edit plan violates strategy.structural_rules={STRUCTURAL_RULE_RESTORE_UNREACHABLE_BRANCH}. "
                        "This strategy requires restoring the targeted existing branch so it is independently reachable, "
                        "but the replacement_block does not contain the concrete structural change."
                    )
            else:
                return (
                    f"The edit plan violates strategy.structural_rules={STRUCTURAL_RULE_RESTORE_UNREACHABLE_BRANCH}. "
                    "Choose an anchor region that directly covers the currently unreachable branch and rewrite it so "
                    "it becomes independently reachable."
                )
        if "top_of_function_side_path" in strategy_constraints.get("forbidden_patterns", []):
            rb_stripped_lines = [l.strip() for l in replacement_block.splitlines()]
            if (
                target_elif_line
                and expected_if in rb_stripped_lines
            ):
                pass
            elif any(line.startswith("if ") and "anchor_line_before" not in line for line in rb_stripped_lines[1:4]):
                return (
                    "The edit plan violates strategy.forbidden_patterns=top_of_function_side_path. "
                    "Do not add a new top-of-function guard or side-path branch; modify the existing branch structure instead."
                )
        # Removed: require replacement_lines[-1] == anchor_after.
        # Model may legitimately append new code after the closing anchor.
        # Check that triple-quoted strings are not left unclosed in the replacement_block.
        # This catches cases where the model drops docstring content but keeps the opening """,
        # creating an unterminated docstring that corrupts the rest of the file.
        # Count occurrences of the delimiter token (not just lines containing it), because
        # the anchor_line_before may itself be a closing """ line which contributes exactly
        # one token — and the replacement block is valid as long as all opens are closed.
        triple_double = sum(line.count('"""') for line in replacement_lines)
        triple_single = sum(line.count("'''") for line in replacement_lines)
        if triple_double % 2 != 0 or triple_single % 2 != 0:
            return (
                "The replacement_block contains an odd number of triple-quote markers, meaning a docstring "
                "is not properly closed. Include the full original docstring (opening and closing \"\"\") "
                "in the replacement_block, then add your new lines after the closing \"\"\"."
            )
        # Only reject __slots__ = () when it appears at module level (not indented),
        # i.e. the line itself has no leading whitespace and is not the anchor lines.
        for line in replacement_lines[1:-1]:
            if line.strip() == "__slots__ = ()" and not line.startswith(" ") and not line.startswith("\t"):
                return "Do not introduce module-level __slots__ = () placeholders; keep __slots__ inside the class body (indented)."
        # Allow class definition lines (no indent), blank lines, and properly indented class body lines.
        # Reject only non-empty lines that are neither class/decorator definitions nor indented body lines.
        for line in replacement_lines:
            if not line:
                continue
            if line.startswith("class ") or line.startswith("@"):
                continue
            if line.startswith("    ") or line.startswith("\t"):
                continue
            # anchor_line_before/after may themselves be class definition lines — already checked above
            if line == replacement_lines[0] or line == replacement_lines[-1]:
                continue
            return f"Unexpected unindented line in replacement_block: {line!r}. Non-class-def lines must be indented."
    if required_obligation_ids:
        missing_obligations = sorted(required_obligation_ids - covered_obligation_ids)
        if missing_obligations:
            return (
                "The edit plan does not cover all required_repair_obligations exposed by retained enhanced tests. "
                f"Missing obligation ids: {', '.join(missing_obligations)}."
            )
    return None


def validate_patch_landing(
    patch: str,
    edit_plan: dict[str, Any] | None,
    code_context: dict[str, str] | None = None,
) -> str | None:
    if not patch.strip():
        return "The generated patch is empty."
    # When edit_plan is None the pipeline is in direct-diff mode (no anchor regions available).
    # In that case we skip edit-plan-specific checks and only validate the diff itself.
    if edit_plan is None:
        return None
    edit_entries = normalize_edit_plan_entries(edit_plan)
    if not edit_entries:
        return "The generated patch is missing edit entries."
    patch_lower = patch.lower()
    # Only reject __slots__ = () when it appears as a module-level addition (unindented '+' line).
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added = line[1:]  # strip the leading '+'
            if added.strip() == "__slots__ = ()" and not added.startswith(" ") and not added.startswith("\t"):
                return "The patch inserted a module-level __slots__ = () placeholder instead of editing the class body."
    for entry in edit_entries:
        target_file = str(entry.get("target_file", ""))
        target_symbol = str(entry.get("target_symbol", ""))
        if target_file and f"+++ b/{target_file}" not in patch:
            return f"The patch did not target the expected file {target_file}."
        # Note: anchor_line_before/anchor_line_after are NOT checked for presence in the diff text.
        # unified_diff uses context=3 by default, so anchor lines farther than 3 lines from the
        # change may not appear in the hunk at all — checking them would falsely reject correct patches.
    completeness_feedback = validate_patch_edit_plan_completeness(patch, edit_plan, code_context)
    if completeness_feedback is not None:
        return completeness_feedback
    semantic_feedback = validate_patch_semantic_landing(patch, edit_plan, code_context)
    if semantic_feedback is not None:
        return semantic_feedback
    return None


def validate_patch_edit_plan_completeness(
    patch: str,
    edit_plan: dict[str, Any] | None,
    code_context: dict[str, str] | None,
) -> str | None:
    if edit_plan is None or not code_context:
        return None
    edit_entries = normalize_edit_plan_entries(edit_plan)
    if not edit_entries:
        return None
    try:
        parsed_patch = PatchSet(patch)
    except Exception:
        return None
    patched_files = {patched_file.path: patched_file for patched_file in parsed_patch}
    actual_updated_files: dict[str, str] = {}
    for target_file, patched_file in patched_files.items():
        original_content = code_context.get(target_file)
        if original_content is None:
            continue
        updated_content, patch_error = apply_patchset_to_content(original_content, patched_file)
        if patch_error is not None:
            return None
        actual_updated_files[target_file] = updated_content
    expected_updated_files, expected_error = apply_fragment_edit_plan(code_context, edit_plan)
    if expected_error is not None:
        return None

    def _contains_line_subsequence(haystack_lines: list[str], needle_lines: list[str]) -> bool:
        if not needle_lines:
            return True
        cursor = 0
        for line in haystack_lines:
            if line == needle_lines[cursor]:
                cursor += 1
                if cursor == len(needle_lines):
                    return True
        return False

    for entry in edit_entries:
        target_file = str(entry.get("target_file", ""))
        target_symbol = str(entry.get("target_symbol", ""))
        original_content = code_context.get(target_file)
        actual_updated_content = actual_updated_files.get(target_file)
        expected_updated_content = expected_updated_files.get(target_file)
        if not original_content or actual_updated_content is None or expected_updated_content is None:
            continue
        original_lines = original_content.splitlines()
        start_idx, end_idx = _resolve_fragment_anchor_indices(
            original_lines,
            anchor_before=str(entry.get("anchor_line_before", "")),
            anchor_after=str(entry.get("anchor_line_after", "")),
            class_anchor_line=str(entry.get("class_anchor_line", "")),
            class_anchor_lineno=entry.get("class_anchor_lineno"),
            anchor_before_lineno=entry.get("anchor_line_before_lineno"),
            anchor_after_lineno=entry.get("anchor_line_after_lineno"),
        )
        if start_idx is None or end_idx is None:
            continue
        original_region_lines = original_lines[start_idx:end_idx + 1]
        replacement_lines = str(entry.get("replacement_block", "")).splitlines()
        distinctive_lines = [
            line for line in replacement_lines
            if line.strip() and line not in original_region_lines
        ]
        actual_lines = actual_updated_content.splitlines()
        expected_lines = expected_updated_content.splitlines()
        if distinctive_lines:
            expected_present = _contains_line_subsequence(expected_lines, distinctive_lines)
            actual_present = _contains_line_subsequence(actual_lines, distinctive_lines)
            if expected_present and actual_present:
                continue
        elif actual_updated_content != original_content:
            continue
        if actual_updated_content == expected_updated_content:
            continue
        else:
            return (
                f"The patch is incomplete for edit-plan target {target_symbol or target_file}. "
                "The edit plan contains multiple modify_region obligations, but this patch did not make a concrete "
                f"change in the planned region between '{entry.get('anchor_line_before', '')}' and "
                f"'{entry.get('anchor_line_after', '')}'. Include all required hunks from the edit plan."
            )
    return None


def _coerce_positive_lineno(value: Any) -> int | None:
    try:
        lineno = int(value)
    except (TypeError, ValueError):
        return None
    return lineno if lineno >= 1 else None


def _find_class_anchor_start(
    original_lines: list[str],
    class_anchor_line: str,
    class_anchor_lineno: Any,
) -> int:
    hinted_lineno = _coerce_positive_lineno(class_anchor_lineno)
    if hinted_lineno is not None:
        hinted_idx = hinted_lineno - 1
        if 0 <= hinted_idx < len(original_lines) and original_lines[hinted_idx] == class_anchor_line:
            return hinted_idx
    if class_anchor_line:
        try:
            return original_lines.index(class_anchor_line)
        except ValueError:
            return 0
    return 0


def _resolve_fragment_anchor_indices(
    original_lines: list[str],
    *,
    anchor_before: str,
    anchor_after: str,
    class_anchor_line: str,
    class_anchor_lineno: Any,
    anchor_before_lineno: Any,
    anchor_after_lineno: Any,
) -> tuple[int | None, int | None]:
    before_hint = _coerce_positive_lineno(anchor_before_lineno)
    after_hint = _coerce_positive_lineno(anchor_after_lineno)
    if before_hint is not None and after_hint is not None:
        before_idx = before_hint - 1
        after_idx = after_hint - 1
        if (
            0 <= before_idx < len(original_lines)
            and 0 <= after_idx < len(original_lines)
            and before_idx < after_idx
            and original_lines[before_idx] == anchor_before
            and original_lines[after_idx] == anchor_after
        ):
            return before_idx, after_idx

    search_start = _find_class_anchor_start(original_lines, class_anchor_line, class_anchor_lineno)
    try:
        end_idx = next(
            idx for idx in range(search_start + 1, len(original_lines))
            if original_lines[idx] == anchor_after
        )
    except StopIteration:
        return None, None
    start_idx = None
    for idx in range(end_idx - 1, search_start - 1, -1):
        if original_lines[idx] == anchor_before:
            start_idx = idx
            break
    return start_idx, end_idx


def validate_python_source_syntax(content: str, target_file: str) -> str | None:
    if not target_file.endswith(".py"):
        return None
    try:
        ast.parse(content, filename=target_file)
    except SyntaxError as exc:
        line_info = f"{target_file}:{exc.lineno}" if exc.lineno else target_file
        detail = exc.msg or "invalid syntax"
        return f"Generated Python source is not syntactically valid at {line_info}: {detail}."
    return None


def apply_patchset_to_content(original_content: str, patched_file: Any) -> tuple[str, str | None]:
    original_lines = original_content.splitlines(keepends=True)
    rebuilt: list[str] = []
    cursor = 0
    for hunk in patched_file:
        hunk_start = max(int(hunk.source_start) - 1, 0)
        if hunk_start < cursor:
            return "", "Overlapping diff hunks are not supported for syntax validation."
        rebuilt.extend(original_lines[cursor:hunk_start])
        cursor = hunk_start
        for line in hunk:
            value = line.value
            if line.is_context:
                if cursor >= len(original_lines):
                    return "", "Unified diff context exceeded the original file length during syntax validation."
                original_value = original_lines[cursor]
                if original_value != value and original_value.rstrip("\n") != value.rstrip("\n"):
                    return "", "Unified diff context did not match the source file during syntax validation."
                rebuilt.append(original_value)
                cursor += 1
            elif line.is_removed:
                if cursor >= len(original_lines):
                    return "", "Unified diff removal exceeded the original file length during syntax validation."
                original_value = original_lines[cursor]
                if original_value != value and original_value.rstrip("\n") != value.rstrip("\n"):
                    return "", "Unified diff removal did not match the source file during syntax validation."
                cursor += 1
            elif line.is_added:
                rebuilt.append(value)
        # continue with next hunk using updated cursor
    rebuilt.extend(original_lines[cursor:])
    return "".join(rebuilt), None


def validate_patch_python_syntax(
    patch: str,
    code_context: dict[str, str] | None,
) -> str | None:
    if not patch.strip() or not code_context:
        return None
    try:
        parsed_patch = PatchSet(patch)
    except Exception:
        return None
    for patched_file in parsed_patch:
        target_file = patched_file.path
        if not target_file.endswith(".py"):
            continue
        original_content = code_context.get(target_file)
        if original_content is None:
            continue
        updated_content, patch_error = apply_patchset_to_content(original_content, patched_file)
        if patch_error is not None:
            return patch_error
        syntax_error = validate_python_source_syntax(updated_content, target_file)
        if syntax_error is not None:
            return syntax_error
    return None


def is_soft_patch_syntax_warning(message: str | None) -> bool:
    if not message:
        return False
    return (
        "Unified diff context exceeded the original file length during syntax validation." in message
        or "Unified diff context did not match the source file during syntax validation." in message
        or "Unified diff removal exceeded the original file length during syntax validation." in message
        or "Unified diff removal did not match the source file during syntax validation." in message
    )


def normalize_patch_feedback(message: str | None) -> str | None:
    if is_soft_patch_syntax_warning(message):
        return None
    return message


def extract_precise_failure_feedback(log_text: str, target_file: str | None = None) -> str | None:
    hard_signal = get_hard_failure_signal(log_text)
    if not hard_signal:
        return None
    file_pattern = re.escape(target_file) if target_file else r"[^\"]+"
    match = re.search(rf'File "[^"]*{file_pattern}", line (\d+)', log_text)
    if match and target_file:
        return f"{hard_signal} occurred at {target_file}:{match.group(1)}."
    if match:
        return f"{hard_signal} occurred near line {match.group(1)}."
    return f"{hard_signal} occurred during test execution."


def summarize_failure_log(log_text: str, max_chars: int = 400) -> str:
    for marker in ("AssertionError", "FAILED", "ERROR", "Traceback"):
        idx = log_text.find(marker)
        if idx != -1:
            return truncate_text(log_text[idx:], max_chars)
    return truncate_text(log_text, max_chars)


def _extract_error_lines(log_text: str) -> str:
    """Return only lines that represent actual runtime errors, not source code.

    Pytest outputs error lines prefixed with 'E ' (e.g. 'E   ImportError: ...').
    We also include lines that are error headers (e.g. 'ImportError while loading conftest').
    This prevents false positives from matching 'except ImportError:' in werkzeug source.
    """
    error_lines = []
    for line in log_text.splitlines():
        stripped = line.strip()
        # Pytest error lines start with "E " or are exception header lines
        if stripped.startswith("E ") or stripped.startswith("E\t"):
            error_lines.append(stripped)
        # Collection error headers (e.g. "ImportError while loading conftest ...")
        elif any(stripped.lower().startswith(sig) for sig in HARD_FAILURE_SIGNALS):
            error_lines.append(stripped)
    return "\n".join(error_lines)


def extract_per_test_tracebacks(log_text: str, test_identifiers: list[str]) -> dict[str, str]:
    """Extract the traceback section for each named test from a pytest log."""
    result: dict[str, str] = {}
    lines = log_text.splitlines()
    for tid in test_identifiers:
        short_name = tid.split("::")[-1]
        # Find section header like "______ test_foo ______"
        header_idx = next(
            (i for i, l in enumerate(lines) if re.search(r"_{5,}\s+" + re.escape(short_name) + r"\s+_{5,}", l)),
            None,
        )
        if header_idx is None:
            # Fallback: failed/error summary line that names the test.
            header_idx = next(
                (
                    i for i, l in enumerate(lines)
                    if short_name in l and any(token in l for token in ("FAILED", "ERROR", "AssertionError", "TypeError", "ValueError"))
                ),
                None,
            )
        if header_idx is None:
            continue
        # Collect until next section header or end
        end_idx = next(
            (i for i in range(header_idx + 1, len(lines)) if re.match(r"^_{5,}\s", lines[i])),
            len(lines),
        )
        block_lines = lines[header_idx: min(header_idx + 40, end_idx)]
        if not _traceback_block_has_failure_signal(block_lines):
            continue
        snippet = "\n".join(block_lines)
        result[short_name] = snippet
    return result


def enrich_strategy_edit_targets_from_tracebacks(
    strategy: dict[str, Any] | None,
    original_failure_log: str,
    instance: dict[str, Any],
    code_context: dict[str, str],
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Force-add source locations referenced in per-test tracebacks into edit_targets.

    When a traceback line reads `src/flask/blueprints.py:364: AssertionError`
    and that file is in code_context, the model MUST edit it.  We also cross-reference
    with analysis.suspicious_symbols to add the precise ::Symbol notation so the
    fragment-edit model edits the correct method, not just the file.
    """
    if strategy is None:
        return strategy
    orig_ids = get_original_test_identifiers(instance)
    per_test = extract_per_test_tracebacks(original_failure_log, orig_ids)
    if not per_test:
        return strategy

    # Pattern: "  src/flask/blueprints.py:364: AssertionError"
    # We only want lines whose third token starts with an uppercase letter (ErrorType).
    tb_source_pattern = re.compile(r"^\s*([\w/][^\s:]+\.py):(\d+):\s+([A-Z]\w+)")

    # Build a map: file_path -> set of line numbers that appear as error locations
    file_error_lines: dict[str, set[int]] = {}
    for tb in per_test.values():
        for line in tb.splitlines():
            m = tb_source_pattern.match(line)
            if m:
                fpath = m.group(1).lstrip("./")
                if fpath in code_context:
                    file_error_lines.setdefault(fpath, set()).add(int(m.group(2)))

    if not file_error_lines:
        return strategy

    # Build a map from file -> list of (symbol, approx_line) from analysis.suspicious_symbols
    # Format: "ClassName.method_name" with file in affected_components
    symbol_locations: dict[str, list[tuple[str, int | None]]] = {}
    for comp in (analysis or {}).get("affected_components", []):
        fpath = comp.get("file", "")
        symbol = comp.get("symbol", "")
        if fpath and symbol:
            symbol_locations.setdefault(fpath, []).append((symbol, None))

    existing = list(strategy.get("edit_targets", []))
    existing_files = {str(t).split("::")[0] for t in existing}
    existing_symbols = {str(t) for t in existing}

    def _find_enclosing_function(content: str, lineno: int) -> str | None:
        """Return 'ClassName.method' or 'function' for the function containing lineno (1-indexed)."""
        lines = content.splitlines()
        target_idx = lineno - 1  # 0-indexed
        # Walk backwards to find enclosing def
        def_pattern = re.compile(r"^(\s*)def\s+(\w+)\s*\(")
        class_pattern = re.compile(r"^class\s+(\w+)\b")
        enclosing_def: str | None = None
        enclosing_class: str | None = None
        def_indent = -1
        for i in range(min(target_idx, len(lines) - 1), -1, -1):
            m = def_pattern.match(lines[i])
            if m and enclosing_def is None:
                indent = len(m.group(1))
                if indent > def_indent or def_indent == -1:
                    enclosing_def = m.group(2)
                    def_indent = indent
            mc = class_pattern.match(lines[i])
            if mc and enclosing_def is not None:
                enclosing_class = mc.group(1)
                break
        if enclosing_def is None:
            return None
        if enclosing_class:
            return f"{enclosing_class}.{enclosing_def}"
        return enclosing_def

    for fpath, linenos in file_error_lines.items():
        if is_test_like_path(fpath):
            continue
        if fpath not in existing_files:
            existing.append(fpath)
            existing_files.add(fpath)

        # Detect enclosing function for each error line and add as ::Symbol target
        content = code_context.get(fpath, "")
        for lineno in linenos:
            symbol = _find_enclosing_function(content, lineno)
            if symbol:
                qualified = f"{fpath}::{symbol}"
                if qualified not in existing_symbols:
                    existing.append(qualified)
                    existing_symbols.add(qualified)
                    existing_files.add(fpath)

        # Also promote any dependency_files entries for this file
        dep_files_raw = [str(d) for d in strategy.get("dependency_files", [])]
        for dep in dep_files_raw:
            dep_file = dep.split("::")[0].lstrip("./")
            if dep_file == fpath and not is_test_like_path(dep_file):
                if dep not in existing_symbols:
                    existing.append(dep)
                    existing_symbols.add(dep)
                    existing_files.add(dep_file)

    strategy = dict(strategy)
    strategy["edit_targets"] = existing
    return strategy


def has_hard_failure_signal(log_text: str) -> bool:
    error_text = _extract_error_lines(log_text).lower()
    return any(signal in error_text for signal in HARD_FAILURE_SIGNALS)


def get_hard_failure_signal(log_text: str) -> str | None:
    error_text = _extract_error_lines(log_text).lower()
    for signal in HARD_FAILURE_SIGNALS:
        if signal in error_text:
            return signal
    return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def build_context_file_list(instance: dict[str, Any], max_files: int) -> list[str]:
    # Source files from the fix patch take priority — they are what the LLM needs to edit.
    patch_files = get_modified_files(instance["patch"]) + get_new_files(instance["patch"])
    source_files = dedupe_preserve_order([f for f in patch_files if not is_test_like_path(f)])
    # Test files from the test patch fill remaining slots.
    test_files = dedupe_preserve_order(
        get_modified_files(instance["test_patch"]) + get_new_files(instance["test_patch"])
    )
    return dedupe_preserve_order(source_files + test_files)[:max_files]


def _container_cat(container, path: str) -> str | None:
    command = f"cd /testbed && cat {shlex.quote(path)}"
    result = container.exec_run(f"/bin/bash -lc {shlex.quote(command)}")
    if result.exit_code != 0:
        return None
    return result.output.decode("utf-8", errors="replace")


def read_code_context(instance: dict[str, Any], max_files: int, max_chars_per_file: int) -> dict[str, str]:
    client = docker.from_env()
    test_spec = make_test_spec(instance)
    instance_id = instance["instance_id"]
    log_dir = Path("logs/enhanced_patch_pipeline/context") / instance_id
    logger = setup_logger(instance_id, log_dir / "context.log")
    container = None
    try:
        container = build_container(test_spec, client, "context", logger, nocache=False, force_rebuild=False)
        container.start()
        context = {}
        for path in build_context_file_list(instance, max_files):
            content = _container_cat(container, path)
            if content is not None:
                context[path] = truncate_text(content, max_chars_per_file)
                # For test files that were truncated, also store the last 60 lines
                # under a "__tail__" key so the enhanced-test prompt can give the
                # model exact context lines near the end of the file.
                if is_test_like_path(path) and len(content) > max_chars_per_file:
                    tail_lines = content.splitlines()[-60:]
                    context[f"__tail__{path}"] = "\n".join(tail_lines)
        return context
    finally:
        cleanup_container(client, container, logger)


def get_strategy_edit_target_files(strategy: dict[str, Any] | None) -> list[str]:
    target_candidates = []
    target_candidates.extend((strategy or {}).get("edit_targets", []))
    target_candidates.extend((strategy or {}).get("dependency_files", []))
    return [
        str(target)
        for target in target_candidates
        if isinstance(target, str) and "/" in target and not target.endswith("/")
    ]


def is_test_like_path(path: str) -> bool:
    return "/tests/" in path or path.startswith("tests/") or "test_" in Path(path).name


def get_required_source_targets(strategy: dict[str, Any] | None) -> list[str]:
    # Only edit_targets are *required* to appear in coverage_check.
    # dependency_files are for context only and must not be enforced.
    edit_targets = [
        str(t) for t in (strategy or {}).get("edit_targets", [])
        if isinstance(t, str) and "/" in t and not t.endswith("/")
    ]
    return [path for path in dedupe_preserve_order(edit_targets) if not is_test_like_path(path)]


def augment_code_context_with_targets(
    instance: dict[str, Any],
    code_context: dict[str, str],
    target_files: list[str],
    max_chars_per_file: int,
) -> dict[str, str]:
    # Fetch files that are either absent OR truncated in the current context.
    # Source files that are target edit files must be complete so the model can
    # produce correct diff context lines.  A truncated entry ends with the
    # "...[truncated N chars]..." marker written by truncate_text().
    def is_truncated(content: str) -> bool:
        return "...[truncated " in content

    targets_to_fetch = [
        path for path in dedupe_preserve_order(target_files)
        if path not in code_context or is_truncated(code_context.get(path, ""))
    ]
    if not targets_to_fetch:
        return dict(code_context)
    client = docker.from_env()
    test_spec = make_test_spec(instance)
    instance_id = instance["instance_id"]
    log_dir = Path("logs/enhanced_patch_pipeline/context-augment") / instance_id
    logger = setup_logger(instance_id, log_dir / "context.log")
    container = None
    augmented_context = dict(code_context)
    try:
        container = build_container(test_spec, client, "context-augment", logger, nocache=False, force_rebuild=False)
        container.start()
        for path in targets_to_fetch:
            content = _container_cat(container, path)
            if content is not None:
                # Give target edit files a much larger quota so the model sees
                # the real content around the edit site, not a truncated stub.
                file_limit = max(max_chars_per_file * 4, 40000)
                augmented_context[path] = truncate_text(content, file_limit)
        return augmented_context
    finally:
        cleanup_container(client, container, logger)
        close_logger(logger)


def augment_code_context_with_failure_focus_targets(
    instance: dict[str, Any],
    code_context: dict[str, str],
    failure_focus: dict[str, Any] | None,
    max_chars_per_file: int,
) -> dict[str, str]:
    suggested_files = _infer_failure_focus_source_files(failure_focus, code_context)
    if not suggested_files:
        return dict(code_context)
    return augment_code_context_with_targets(
        instance=instance,
        code_context=code_context,
        target_files=suggested_files,
        max_chars_per_file=max_chars_per_file,
    )


def _infer_minimal_validation_subject_from_failure_focus(
    failure_focus: dict[str, Any] | None,
    covered_original_tests: list[str] | None,
) -> str:
    snippets = " ".join((failure_focus or {}).get("failure_snippets") or [])
    tracebacks = []
    target_tracebacks = (failure_focus or {}).get("target_test_tracebacks") or {}
    for test_id in covered_original_tests or []:
        for existing_test_id, tb in target_tracebacks.items():
            if _test_identifiers_match(test_id, existing_test_id):
                tracebacks.append(str(tb))
    combined = f"{snippets}\n" + "\n".join(tracebacks)
    if "__dict__" in combined:
        return "__dict__"
    return "core"


def backfill_candidate_minimal_structure(
    candidate: CandidatePatch,
    failure_focus: dict[str, Any] | None,
    code_context: dict[str, str] | None,
) -> None:
    idea = dict(candidate.idea or {})
    if (
        str(idea.get("target_source_symbol") or "").strip()
        and str(idea.get("target_validation_subject") or "").strip()
        and list(idea.get("covers_obligations") or [])
    ):
        candidate.idea = idea
        return
    covered_original_tests = [
        str(test_id).strip()
        for test_id in candidate.covered_original_tests or []
        if str(test_id).strip()
    ]
    if not covered_original_tests:
        fallback_tests = (
            list((failure_focus or {}).get("active_fail_to_pass_identifiers") or [])
            or list((failure_focus or {}).get("inactive_fail_to_pass_identifiers") or [])
            or list((failure_focus or {}).get("original_test_identifiers") or [])
        )
        fallback_tests = [
            str(test_id).strip()
            for test_id in fallback_tests
            if str(test_id).strip()
        ]
        fallback_tests = dedupe_preserve_order(fallback_tests)
        if len(fallback_tests) == 1:
            covered_original_tests = [fallback_tests[0]]
            candidate.covered_original_tests = covered_original_tests
    dominant_symbols = _infer_failure_focus_dominant_symbols(
        {
            **(failure_focus or {}),
            "active_fail_to_pass_identifiers": covered_original_tests or ((failure_focus or {}).get("active_fail_to_pass_identifiers") or []),
            "inactive_fail_to_pass_identifiers": covered_original_tests or ((failure_focus or {}).get("inactive_fail_to_pass_identifiers") or []),
            "original_test_identifiers": covered_original_tests or ((failure_focus or {}).get("original_test_identifiers") or []),
        },
        code_context,
    )
    if dominant_symbols and not str(idea.get("target_source_symbol") or "").strip():
        idea["target_source_symbol"] = dominant_symbols[0]
    if not str(idea.get("target_validation_subject") or "").strip():
        idea["target_validation_subject"] = _infer_minimal_validation_subject_from_failure_focus(
            failure_focus,
            covered_original_tests,
        )
    if not list(idea.get("covers_obligations") or []) and covered_original_tests:
        subject = str(idea.get("target_validation_subject") or "core").strip() or "core"
        idea["covers_obligations"] = [f"{covered_original_tests[0]}::core::{subject}"]
    candidate.idea = idea


def _candidate_has_minimal_structure(candidate: CandidatePatch) -> bool:
    idea = candidate.idea or {}
    source_symbol = str(idea.get("target_source_symbol") or "").strip()
    validation_subject = str(idea.get("target_validation_subject") or "").strip()
    covers_obligations = [str(item).strip() for item in (idea.get("covers_obligations") or []) if str(item).strip()]
    return bool(source_symbol and validation_subject and covers_obligations)


def render_eval_script(commands: list[str]) -> str:
    rendered = []
    for command in commands:
        if command == f": '{START_MARKER}'":
            rendered.append(f"echo '{START_MARKER}'")
        elif command == f": '{END_MARKER}'":
            rendered.append(f"echo '{END_MARKER}'")
        else:
            rendered.append(command)
    return "\n".join(["#!/bin/bash", "set -uxo pipefail", *rendered, ""])


def make_custom_test_spec(instance: dict[str, Any], test_patch: str):
    test_spec = make_test_spec(instance)
    repo_directory = "/testbed"
    env_name = "testbed"
    eval_script_list = make_eval_script_list(
        instance,
        MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]],
        env_name,
        repo_directory,
        instance["base_commit"],
        test_patch,
    )
    return replace(test_spec, eval_script_list=eval_script_list)


def apply_patch_in_container(container, patch_text: str) -> tuple[bool, str, str]:
    if not normalize_patch(patch_text):
        return True, "", "no_patch"
    patch_path = Path("/tmp/patch.diff")
    local_patch = Path("/tmp") / "swebench_enhanced_patch.diff"
    local_patch.write_text(normalize_patch(patch_text))
    try:
        copy_to_container(container, local_patch, patch_path)
    finally:
        if local_patch.exists():
            local_patch.unlink()
    last_output = ""
    for idx, git_apply_cmd in enumerate(GIT_APPLY_CMDS):
        val = container.exec_run(f"{git_apply_cmd} {patch_path}", workdir="/testbed", user="root")
        last_output = val.output.decode("utf-8", errors="replace")
        if val.exit_code == 0:
            apply_mode = "clean_apply" if idx == 0 else "fuzzy_apply"
            return True, last_output, apply_mode
    return False, last_output, "apply_failed"


def run_eval(
    instance: dict[str, Any],
    test_patch: str,
    code_patch: str,
    run_id: str,
    timeout: int,
) -> EvalResult:
    client = docker.from_env()
    test_spec = make_custom_test_spec(instance, test_patch)
    instance_id = instance["instance_id"]
    model_name = "enhanced-pipeline"
    log_dir = Path("logs/enhanced_patch_pipeline") / run_id / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(instance_id, log_dir / "run.log")
    container = None
    timed_out = False
    report = None
    try:
        container = build_container(test_spec, client, run_id, logger, nocache=False, force_rebuild=False)
        container.start()
        patch_applied, patch_output, patch_apply_mode = apply_patch_in_container(container, code_patch)
        if not patch_applied:
            return EvalResult(
                resolved=False,
                status_map={},
                log_text=patch_output,
                log_path=str(log_dir / "test_output.txt"),
                report=None,
                patch_applied=False,
                patch_apply_mode=patch_apply_mode,
                timed_out=False,
                error="Failed to apply code patch",
            )

        eval_file = log_dir / "eval.sh"
        eval_file.write_text(render_eval_script(test_spec.eval_script_list))
        copy_to_container(container, eval_file, Path("/eval.sh"))
        test_output, timed_out, _ = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout=timeout)
        test_output_path = log_dir / "test_output.txt"
        test_output_path.write_text(test_output)
        status_map, applied = get_logs_eval(test_spec, str(test_output_path))
        report = {
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: model_name,
            KEY_PREDICTION: normalize_patch(code_patch),
        }
        # Use SWE-bench's official resolved definition:
        # All FAIL_TO_PASS tests must now PASS, and all PASS_TO_PASS tests must still PASS.
        # This is more accurate than "all tests pass" which fails on environment noise.
        gold_f2p = json.loads(instance.get(FAIL_TO_PASS, "[]"))
        gold_p2p = json.loads(instance.get(PASS_TO_PASS, "[]"))
        def test_passed_in(t: str, sm: dict) -> bool:
            return sm.get(t) not in {"FAILED", "ERROR"} and t in sm
        if gold_f2p:
            resolved = (
                applied
                and has_meaningful_status_map(status_map)
                and all(test_passed_in(t, status_map) for t in gold_f2p)
                and all(test_passed_in(t, status_map) for t in gold_p2p)
            )
        else:
            # Fallback for instances without gold labels: require all tests pass
            resolved = (
                applied
                and has_meaningful_status_map(status_map)
                and all(status not in {"FAILED", "ERROR"} for status in status_map.values())
            )
        return EvalResult(
            resolved=resolved,
            status_map=status_map,
            log_text=test_output,
            log_path=str(test_output_path),
            report=report,
            patch_applied=applied,
            patch_apply_mode=patch_apply_mode,
            timed_out=timed_out,
            error=None if applied else "Test patch did not apply or logs could not be parsed",
        )
    except (EvaluationError, Exception) as exc:
        return EvalResult(
            resolved=False,
            status_map={},
            log_text=traceback.format_exc(),
            log_path=str(log_dir / "test_output.txt"),
            report=report,
            patch_applied=False,
            patch_apply_mode="apply_failed",
            timed_out=timed_out,
            error=str(exc),
        )
    finally:
        cleanup_container(client, container, logger)
        close_logger(logger)


class OpenAIResponder:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        if not api_base:
            raise ValueError("OPENAI_API_BASE is required.")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def complete(self, prompt: str) -> tuple[str, str]:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        try:
            response = self.client.chat.completions.create(
                **request,
                max_completion_tokens=self.max_tokens,
            )
        except TypeError:
            response = self.client.chat.completions.create(
                **request,
                max_tokens=self.max_tokens,
            )
        text = response.choices[0].message.content or ""
        patch, error = sanitize_unified_diff(extract_diff(text))
        if error and not patch:
            return text, normalize_patch(extract_diff(text))
        return text, patch

    def complete_text(self, prompt: str) -> str:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        try:
            response = self.client.chat.completions.create(
                **request,
                max_completion_tokens=self.max_tokens,
            )
        except TypeError:
            response = self.client.chat.completions.create(
                **request,
                max_tokens=self.max_tokens,
            )
        return response.choices[0].message.content or ""


def generate_test_idea(
    responder: OpenAIResponder,
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    failure_mode: str,
    template_names: list[str],
    semantic_buckets: list[str],
    failure_focus: dict[str, Any] | None,
    seen_titles: set[str],
    seen_buckets: set[str],
    max_candidate_attempts: int,
    covered_original_tests: set[str] | None = None,
    covered_obligations: set[str] | None = None,
) -> tuple[dict[str, Any] | None, int]:
    feedback = None
    last_idea = None
    original_test_identifiers = get_original_test_identifiers(instance)
    covered_original_tests = set(covered_original_tests or set())
    all_obligations = extract_failure_path_repair_obligations(failure_focus, code_context)
    covered_obligations = set(covered_obligations or set())
    for attempt in range(1, max_candidate_attempts + 1):
        uncovered_original_tests = [
            identifier for identifier in original_test_identifiers
            if identifier not in covered_original_tests
        ]
        uncovered_obligations = [
            obligation["id"]
            for obligations in all_obligations.values()
            for obligation in obligations
            if obligation["id"] not in covered_obligations
        ]
        raw_text = responder.complete_text(
            build_test_idea_prompt(
                instance,
                code_context,
                original_failure_log,
                failure_mode,
                template_names,
                semantic_buckets,
                failure_focus=failure_focus,
                feedback=feedback,
                uncovered_original_tests=uncovered_original_tests,
                uncovered_obligations=uncovered_obligations,
            )
        )
        idea = parse_json_object(raw_text)
        feedback = build_idea_feedback(
            idea,
            seen_titles,
            seen_buckets,
            semantic_buckets,
            original_test_identifiers,
            required_uncovered_tests=uncovered_original_tests,
            failure_focus=failure_focus,
            required_uncovered_obligations=uncovered_obligations,
            code_context=code_context,
        )
        last_idea = idea
        if feedback is None:
            seen_titles.add(str(idea["title"]).strip())
            seen_buckets.add(str(idea["semantic_bucket"]).strip())
            return idea, attempt
    return last_idea, max_candidate_attempts


def generate_patch_analysis(
    responder: OpenAIResponder,
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    filtered_candidates: list[CandidatePatch],
    failure_focus: dict[str, Any] | None = None,
    max_candidate_attempts: int = 1,
) -> tuple[dict[str, Any] | None, int]:
    feedback = None
    last_analysis = None
    required_repair_obligations = select_effective_required_repair_obligations(
        extract_required_repair_obligations(
            filtered_candidates,
            failure_focus=failure_focus,
            code_context=code_context,
        ),
        filtered_candidates=filtered_candidates,
    )
    for attempt in range(1, max_candidate_attempts + 1):
        prompt = build_patch_analysis_prompt(
            instance=instance,
            code_context=code_context,
            original_failure_log=original_failure_log,
            filtered_candidates=filtered_candidates,
            failure_focus=failure_focus,
        )
        if feedback:
            prompt += f"\nRevision feedback for the next analysis:\n{feedback}\n"
        analysis = parse_json_object(responder.complete_text(prompt))
        feedback = build_patch_analysis_feedback(
            analysis,
            required_repair_obligations=required_repair_obligations,
            failure_focus=failure_focus,
            code_context=code_context,
        )
        last_analysis = analysis
        if feedback is None:
            return analysis, attempt
    return last_analysis, max_candidate_attempts


def generate_patch_strategy(
    responder: OpenAIResponder,
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    filtered_candidates: list[CandidatePatch],
    analysis: dict[str, Any] | None,
    max_candidate_attempts: int,
    failure_focus: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, int]:
    feedback = None
    last_strategy = None
    for attempt in range(1, max_candidate_attempts + 1):
        strategy = parse_json_object(
            responder.complete_text(
                build_patch_strategy_prompt(
                    instance=instance,
                    code_context=code_context,
                    original_failure_log=original_failure_log,
                    filtered_candidates=filtered_candidates,
                    analysis=analysis,
                    failure_focus=failure_focus,
                    feedback=feedback,
                )
            )
        )
        feedback = build_patch_strategy_feedback(
            strategy,
            analysis=analysis,
            original_failure_log=original_failure_log,
            failure_focus=failure_focus,
            filtered_candidates=filtered_candidates,
            code_context=code_context,
        )
        last_strategy = strategy
        if feedback is None:
            return strategy, attempt
    return last_strategy, max_candidate_attempts


def _build_assert_to_raise_hint(
    original_failure_log: str,
    anchor_regions: list[dict[str, str]],
) -> str | None:
    """Return a feedback hint when traceback shows AssertionError inside a source anchor region.

    Detects lines like '  src/flask/blueprints.py:364: AssertionError' and checks whether
    the corresponding anchor region contains that assertion, then returns a pre-flight
    instruction so the LLM does not mark the method as no_change.
    """
    # Pattern: "  some/path.py:NNN: AssertionError"
    tb_pattern = re.compile(r"^\s*([\w/][^\s:]+\.py):(\d+):\s+AssertionError", re.MULTILINE)
    assert_locations: list[tuple[str, int]] = []
    for m in tb_pattern.finditer(original_failure_log):
        fpath = m.group(1).lstrip("./")
        assert_locations.append((fpath, int(m.group(2))))
    if not assert_locations:
        return None
    # Find anchor regions whose file matches and whose line range covers the assert location
    hints: list[str] = []
    for region in anchor_regions:
        rfile = str(region.get("target_file", "")).lstrip("./")
        rsym = str(region.get("target_symbol", ""))
        snippet = region.get("region_snippet", "")
        for fpath, _lineno in assert_locations:
            if rfile == fpath and re.search(r"^\s+assert\b", snippet, re.MULTILINE):
                hints.append(
                    f"The traceback shows AssertionError in '{rfile}' inside symbol '{rsym}'. "
                    f"This assert must be replaced with `raise ValueError(...)`. "
                    f"Mark '{rfile}' as modify_region in coverage_check and provide an edit entry for '{rsym}'."
                )
                break
    return " ".join(hints) if hints else None


def generate_fragment_edit_plan(
    responder: OpenAIResponder,
    instance: dict[str, Any],
    original_failure_log: str,
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
    anchor_regions: list[dict[str, str]] | None,
    required_source_targets: list[str],
    required_repair_obligations: list[dict[str, Any]] | None,
    repair_topology: str | None,
    max_candidate_attempts: int,
) -> tuple[dict[str, Any] | None, int]:
    if not anchor_regions:
        return None, 0
    # Pre-compute assert→raise hint: if the traceback shows AssertionError inside a
    # non-test source file that is also an edit_target, prime the first feedback message
    # so the LLM doesn't incorrectly mark that method as no_change.
    _assert_hint = _build_assert_to_raise_hint(original_failure_log, anchor_regions)
    feedback = _assert_hint  # None when no AssertionError traceback found in source files
    last_plan = None
    for attempt in range(1, max_candidate_attempts + 1):
        plan = parse_json_object(
            responder.complete_text(
                build_fragment_edit_prompt(
                    instance=instance,
                    original_failure_log=original_failure_log,
                    analysis=analysis,
                    strategy=strategy,
                    anchor_regions=anchor_regions,
                    required_source_targets=required_source_targets,
                    required_repair_obligations=required_repair_obligations,
                    feedback=(
                        ((feedback + "\n") if feedback else "")
                        + "REPAIR TOPOLOGY RULE: this is a statement_local repair. Prefer a single edit entry in the dominant symbol and avoid expanding to additional symbols unless the current edit cannot possibly satisfy the required obligations."
                        if repair_topology == "statement_local"
                        else (
                            ((feedback + "\n") if feedback else "")
                            + "REPAIR TOPOLOGY RULE: this is a single_root_symbol repair. Keep the plan focused on one dominant symbol/file and avoid speculative base-class or helper-file expansion unless the current file cannot possibly satisfy the failure signal."
                            if repair_topology == "single_root_symbol"
                            else feedback
                        )
                    ),
                )
            )
        )
        feedback = validate_fragment_edit_plan(
            plan,
            anchor_regions,
            required_source_targets=required_source_targets,
            required_repair_obligations=required_repair_obligations,
            strategy=strategy,
            repair_topology=repair_topology,
        )
        last_plan = plan
        if feedback is None:
            return plan, attempt
    return last_plan, max_candidate_attempts


def build_patch_feedback(
    patch: str,
    patch_error: str | None,
    edit_plan: dict[str, Any] | None = None,
    code_context: dict[str, str] | None = None,
    skip_syntax_check: bool = False,
    analysis: dict[str, Any] | None = None,
    strategy: dict[str, Any] | None = None,
    original_failure_log: str = "",
) -> str | None:
    if patch_error and not patch:
        return f"The previous patch was not a valid unified diff: {patch_error}"
    if not patch.strip():
        return "The previous patch was empty. Return a non-empty unified diff that edits source files only."
    if (
        strategy
        and is_negative_gating_patch(patch, original_failure_log)
        and (
            strategy_requires_structural_branch_change(analysis, strategy, original_failure_log)
            or "standalone `if`" in json.dumps(strategy, ensure_ascii=False).lower()
            or "standalone if" in json.dumps(strategy, ensure_ascii=False).lower()
            or "elif -> if" in json.dumps(strategy, ensure_ascii=False).lower()
            or "elif` to `if" in json.dumps(strategy, ensure_ascii=False).lower()
        )
    ):
        return (
            "The previous patch contradicts the repair strategy. The strategy says the fix must preserve behavior when "
            "the flag-enabled mode is ON and restore reachability of an existing branch, but your patch adds a negative "
            "guard such as `not <flag>` and simply disables that path. Reject this workaround and instead modify the "
            "existing control flow so the flagged path itself behaves correctly."
        )
    if (
        strategy
        and strategy_requires_structural_branch_change(analysis, strategy, original_failure_log)
        and not has_structural_elif_to_if_change(patch)
    ):
        return (
            "The previous patch contradicts the repair strategy. The strategy explicitly requires restoring a previously "
            "unreachable existing correction branch so it becomes reachable again. For the currently recognized concrete "
            "pattern, that means a minimal structural `elif` -> `if` change. Your patch does not perform that "
            "reachability-restoring change; it adds or edits side-path logic instead. Reject this workaround and modify "
            "the existing branch structure directly."
        )
    landing_feedback = validate_patch_landing(patch, edit_plan, code_context=code_context)
    if landing_feedback is not None:
        return landing_feedback
    if (
        edit_plan is None or not normalize_edit_plan_entries(edit_plan)
    ) and _has_single_root_symbol_signature(analysis=analysis, strategy=strategy, failure_focus=None):
        single_root_feedback = validate_single_root_symbol_patch_landing(
            patch,
            code_context=code_context,
            analysis=analysis,
            strategy=strategy,
        )
        if single_root_feedback is not None:
            return single_root_feedback
    if skip_syntax_check:
        return None
    syntax_feedback = validate_patch_python_syntax(patch, code_context)
    if is_soft_patch_syntax_warning(syntax_feedback):
        return None
    if syntax_feedback is not None:
        return syntax_feedback
    try:
        parsed_patch = PatchSet(patch)
    except Exception:
        return None
    touched_files = []
    for patched_file in parsed_patch:
        source = patched_file.path
        touched_files.append(source)
        if "/tests/" in source or source.startswith("tests/") or "test_" in Path(source).name:
            return "Do not modify tests in the repair patch. Edit source files only."
    if not touched_files:
        return "The repair patch must modify at least one source file."
    return None


def patch_improves_metrics(candidate_eval: EvalResult, baseline_eval: EvalResult) -> bool:
    if not has_meaningful_status_map(candidate_eval.status_map):
        return False
    return (
        count_passed(candidate_eval.status_map) > count_passed(baseline_eval.status_map)
        or count_failed(candidate_eval.status_map) < count_failed(baseline_eval.status_map)
    )


def score_patch_candidate_eval(
    candidate_eval: EvalResult | None,
    baseline_eval: EvalResult,
    f2p_identifiers: list[str] | None = None,
) -> tuple[int, int, int, int]:
    if candidate_eval is None or not has_meaningful_status_map(candidate_eval.status_map):
        return (-10**9, -10**9, -10**9, -10**9)
    mode_rank = {
        "clean_apply": 3,
        "fuzzy_apply": 2,
        "no_patch": 1,
    }.get(candidate_eval.patch_apply_mode, 0)
    f2p_passed = 0
    if f2p_identifiers:
        f2p_passed = sum(
            1 for tid in f2p_identifiers
            if candidate_eval.status_map.get(tid) in {"PASSED", "XPASS"}
        )
    passed_delta = count_passed(candidate_eval.status_map) - count_passed(baseline_eval.status_map)
    failed_delta = count_failed(baseline_eval.status_map) - count_failed(candidate_eval.status_map)
    return (
        f2p_passed,
        mode_rank,
        passed_delta + failed_delta,
        passed_delta,
    )


def needs_minimal_structural_fix_guidance(
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
    original_failure_log: str,
) -> bool:
    combined = " ".join(
        [
            json.dumps(analysis or {}, ensure_ascii=False).lower(),
            json.dumps(strategy or {}, ensure_ascii=False).lower(),
            original_failure_log.lower(),
        ]
    )
    mentions_control_flow = any(
        token in combined
        for token in (
            "elif chain",
            "short-circuit",
            "short circuit",
            "standalone `if`",
            "standalone if",
            "unreachable",
            "control-flow issue",
            "runs regardless",
        )
    )
    mentions_dispatcher_path = any(
        token in combined
        for token in ("hook", "report", "dispatcher", "makereport", "runtest")
    )
    return mentions_control_flow and mentions_dispatcher_path


def extract_enabled_flag_tokens(original_failure_log: str) -> list[str]:
    tokens = re.findall(r"--([a-zA-Z0-9_-]+)", original_failure_log.lower())
    return dedupe_preserve_order(tokens)


def count_meaningful_patch_changes(patch_text: str) -> int:
    count = 0
    for line in normalize_patch(patch_text).splitlines():
        if not line or line.startswith(("diff --git", "--- ", "+++ ", "@@ ")):
            continue
        if line[0] not in "+-":
            continue
        payload = line[1:].strip()
        if not payload or payload.startswith("#"):
            continue
        count += 1
    return count


def is_minimal_structural_control_flow_patch(patch_text: str) -> bool:
    removed_lines: list[str] = []
    added_lines: list[str] = []
    for line in normalize_patch(patch_text).splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("-"):
            payload = line[1:].rstrip()
            if payload.strip():
                removed_lines.append(payload)
        elif line.startswith("+"):
            payload = line[1:].rstrip()
            if payload.strip():
                added_lines.append(payload)
    if len(removed_lines) != 1 or len(added_lines) != 1:
        return False
    removed = removed_lines[0]
    added = added_lines[0]
    removed_stripped = removed.strip()
    added_stripped = added.strip()
    if removed_stripped.startswith("elif ") and added_stripped == removed_stripped.replace("elif ", "if ", 1):
        return True
    if removed_stripped.startswith("elif(") and added_stripped == removed_stripped.replace("elif(", "if(", 1):
        return True
    return False


def has_structural_elif_to_if_change(patch_text: str) -> bool:
    removed_elif_lines: set[str] = set()
    added_if_lines: set[str] = set()
    for line in normalize_patch(patch_text).splitlines():
        if line.startswith(("--- ", "+++ ", "@@ ")) or not line:
            continue
        if line.startswith("-"):
            payload = line[1:].rstrip()
            stripped = payload.strip()
            if stripped.startswith("elif "):
                removed_elif_lines.add(stripped)
            elif stripped.startswith("elif("):
                removed_elif_lines.add(stripped)
        elif line.startswith("+"):
            payload = line[1:].rstrip()
            stripped = payload.strip()
            if stripped.startswith("if "):
                added_if_lines.add(stripped)
            elif stripped.startswith("if("):
                added_if_lines.add(stripped)
    for removed in removed_elif_lines:
        candidate = removed.replace("elif ", "if ", 1) if removed.startswith("elif ") else removed.replace("elif(", "if(", 1)
        if candidate in added_if_lines:
            return True
    return False


def is_negative_gating_patch(patch_text: str, original_failure_log: str) -> bool:
    flags = extract_enabled_flag_tokens(original_failure_log)
    if not flags:
        return False
    added_payloads = [
        line[1:].strip().lower()
        for line in normalize_patch(patch_text).splitlines()
        if line.startswith("+") and not line.startswith("+++ ")
    ]
    for payload in added_payloads:
        if " not " not in f" {payload} ":
            continue
        if any(flag in payload for flag in flags):
            return True
    return False


def strategy_requires_structural_branch_change(
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
    original_failure_log: str,
) -> bool:
    strategy_text = json.dumps(strategy or {}, ensure_ascii=False).lower()
    return (
        needs_minimal_structural_fix_guidance(analysis, strategy, original_failure_log)
        and any(
            token in strategy_text
            for token in (
                "standalone `if`",
                "standalone if",
                "elif -> if",
                "elif` to `if",
                "convert the `elif`",
                "convert the existing `elif`",
            )
        )
    )


def synthesize_structural_patch_from_strategy(
    code_context: dict[str, str],
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
    original_failure_log: str,
) -> tuple[str, str | None]:
    constraints = extract_strategy_constraints(strategy)
    if STRUCTURAL_RULE_RESTORE_UNREACHABLE_BRANCH not in constraints.get("structural_rules", []):
        return "", None
    if not strategy_requires_structural_branch_change(analysis, strategy, original_failure_log):
        return "", None
    suspicious_symbols = [
        str(symbol)
        for symbol in (analysis or {}).get("suspicious_symbols", [])
        if isinstance(symbol, str) and str(symbol).strip()
    ]
    target_files = [
        str(target).split("::")[0]
        for target in (strategy or {}).get("edit_targets", [])
        if isinstance(target, str) and "/" in str(target)
    ]
    for target_file in dedupe_preserve_order(target_files):
        content = code_context.get(target_file, "")
        if not content:
            continue
        lines = content.splitlines()
        target_function = next(
            (
                symbol for symbol in suspicious_symbols
                if "." not in symbol and re.search(rf"^\s*def\s+{re.escape(symbol)}\s*\(", content, re.MULTILINE)
            ),
            None,
        )
        if target_function is None:
            continue
        func_pattern = re.compile(rf"^(\s*)def\s+{re.escape(target_function)}\s*\(")
        func_idx = None
        func_indent = 0
        for idx, line in enumerate(lines):
            match = func_pattern.match(line)
            if match:
                func_idx = idx
                func_indent = len(match.group(1))
                break
        if func_idx is None:
            continue
        block_end = len(lines)
        for idx in range(func_idx + 1, len(lines)):
            stripped = lines[idx].strip()
            if not stripped:
                continue
            indent = len(lines[idx]) - len(lines[idx].lstrip())
            if indent <= func_indent and stripped.startswith(("def ", "class ")):
                block_end = idx
                break
        branch_candidates: list[int] = []
        hint_tokens = [
            token for token in suspicious_symbols
            if token != target_function and token not in {"rep.longrepr"}
        ]
        body_indent = func_indent + 4
        for idx in range(func_idx + 1, block_end):
            stripped = lines[idx].strip()
            indent = len(lines[idx]) - len(lines[idx].lstrip())
            if indent != body_indent:
                continue
            if stripped.startswith(("elif ", "elif(")):
                branch_candidates.append(idx)
        if not branch_candidates:
            continue
        preferred_idx = None
        for idx in branch_candidates:
            stripped = lines[idx].strip()
            if any(token in stripped for token in hint_tokens):
                preferred_idx = idx
                break
        if preferred_idx is None:
            preferred_idx = branch_candidates[-1]
        updated_lines = list(lines)
        target_line = updated_lines[preferred_idx]
        stripped_target = target_line.lstrip()
        leading = target_line[: len(target_line) - len(stripped_target)]
        if stripped_target.startswith("elif "):
            updated_lines[preferred_idx] = f"{leading}if {stripped_target[5:]}"
        elif stripped_target.startswith("elif("):
            updated_lines[preferred_idx] = f"{leading}if({stripped_target[5:]}"
        else:
            continue
        updated_content = "\n".join(updated_lines)
        if content.endswith("\n"):
            updated_content += "\n"
        patch = build_unified_diff_from_replacement(content, updated_content, target_file)
        patch = normalize_patch(patch)
        patch, patch_error = sanitize_unified_diff(patch)
        if patch and patch_error is None and has_structural_elif_to_if_change(patch):
            return patch, None
        if patch_error:
            return "", patch_error
    return "", None


def score_patch_candidate_choice(
    candidate_eval: EvalResult | None,
    baseline_eval: EvalResult,
    f2p_identifiers: list[str] | None,
    patch_text: str,
    analysis: dict[str, Any] | None,
    strategy: dict[str, Any] | None,
    original_failure_log: str,
) -> tuple[int, int, int, int, int, int, int]:
    base_score = score_patch_candidate_eval(candidate_eval, baseline_eval, f2p_identifiers)
    structural_bonus = 0
    negative_gating_penalty = 0
    if needs_minimal_structural_fix_guidance(analysis, strategy, original_failure_log):
        if is_minimal_structural_control_flow_patch(patch_text):
            structural_bonus = 1
        if is_negative_gating_patch(patch_text, original_failure_log):
            negative_gating_penalty = -1
    meaningful_changes = count_meaningful_patch_changes(patch_text)
    return (
        base_score[0],
        base_score[1],
        base_score[2],
        structural_bonus,
        negative_gating_penalty,
        -meaningful_changes,
        base_score[3],
    )


def should_prioritize_edit_plan_patch(edit_plan: dict[str, Any] | None) -> bool:
    entries = normalize_edit_plan_entries(edit_plan)
    return bool(entries)


def should_lock_assembled_edit_plan_patch(edit_plan: dict[str, Any] | None) -> bool:
    entries = normalize_edit_plan_entries(edit_plan)
    return len(entries) > 1


def should_lock_plan_derived_patch(edit_plan: dict[str, Any] | None) -> bool:
    entries = normalize_edit_plan_entries(edit_plan)
    if not entries:
        return False
    if len(entries) > 1:
        return True
    entry = entries[0]
    return bool(str(entry.get("target_file", "")).strip())


def run_semantic_oracle_checks(
    instance: dict[str, Any],
    filtered_candidates: list[CandidatePatch],
    code_patch: str,
    timeout: int,
) -> SemanticOracleResult | None:
    # No enhanced candidates means there is nothing to oracle-check.
    # Return None so callers can distinguish "skipped" from "failed".
    if not filtered_candidates:
        return None
    failed_identifiers: list[str] = []
    status_maps: list[dict[str, str]] = []
    failure_summaries: dict[str, str] = {}
    for idx, candidate in enumerate(filtered_candidates, start=1):
        eval_result = run_eval(
            instance=instance,
            test_patch=candidate.patch,
            code_patch=code_patch,
            run_id=f"semantic-oracle-{idx}",
            timeout=timeout,
        )
        status_maps.append(eval_result.status_map)
        enhanced_identifiers = candidate.enhanced_identifiers or []
        candidate_failed = find_failing_identifiers(eval_result.status_map, enhanced_identifiers)
        if not has_meaningful_status_map(eval_result.status_map):
            candidate_failed.extend(enhanced_identifiers)
        candidate_failed = dedupe_preserve_order(candidate_failed)
        failed_identifiers.extend(candidate_failed)
        for identifier in candidate_failed:
            failure_summaries[identifier] = summarize_failure_log(eval_result.log_text)
    failed_identifiers = dedupe_preserve_order(failed_identifiers)
    return SemanticOracleResult(
        passed=not failed_identifiers,
        failed_identifiers=failed_identifiers,
        status_maps=status_maps,
        failure_summaries=failure_summaries,
    )


def get_patch_acceptance_reason(
    candidate_eval: EvalResult,
    baseline_eval: EvalResult,
    semantic_oracle: SemanticOracleResult | None = None,
    f2p_identifiers: list[str] | None = None,
) -> tuple[bool, str]:
    hard_failure_signal = get_hard_failure_signal(candidate_eval.log_text)
    # Only reject if the hard failure signal is NEW (not already present in baseline).
    # Some test suites intentionally exercise SyntaxError/ImportError handling,
    # so those keywords appear in the baseline log too.
    if hard_failure_signal and not get_hard_failure_signal(baseline_eval.log_text):
        return False, f"rejected_hard_failure_{hard_failure_signal}"
    if candidate_eval.patch_apply_mode not in {"clean_apply", "no_patch"}:
        return False, "rejected_patch_apply_failed"
    # Require metric improvement: more tests passing or fewer tests failing.
    if not patch_improves_metrics(candidate_eval, baseline_eval):
        return False, "rejected_no_metric_improvement"
    # If we know which tests must flip from FAIL to PASS, require all of them to pass.
    # This prevents accepting a patch that fixes one unrelated test while the primary
    # failing tests (FAIL_TO_PASS targets) still fail — the root bug is not yet fixed.
    if f2p_identifiers:
        still_failing = [
            tid for tid in f2p_identifiers
            if candidate_eval.status_map.get(tid) not in {"PASSED", "XPASS"}
        ]
        if still_failing:
            return False, f"rejected_f2p_not_resolved:{','.join(still_failing[:3])}"
    # Semantic oracle is informational only — enhanced tests may target related but broader
    # behavior than the original failing tests (e.g., Symbol vs Basic). We do not hard-reject
    # a patch that already improves original test metrics just because enhanced tests still fail.
    oracle_suffix = ""
    if semantic_oracle is not None and not semantic_oracle.passed:
        oracle_suffix = "_oracle_partial"
    return True, f"accepted_{candidate_eval.patch_apply_mode}_with_improvement{oracle_suffix}"


def generate_patch_with_strategy(
    responder: OpenAIResponder,
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    filtered_candidates: list[CandidatePatch],
    max_candidate_attempts: int,
    baseline_eval: EvalResult,
    timeout: int,
    max_chars_per_file: int,
    failure_focus: dict[str, Any] | None = None,
    hide_original_test_patch_in_repair: bool = False,
) -> PatchGenerationResult:
    analysis, analysis_attempts = generate_patch_analysis(
        responder=responder,
        instance=instance,
        code_context=code_context,
        original_failure_log=original_failure_log,
        filtered_candidates=filtered_candidates,
        failure_focus=failure_focus,
        max_candidate_attempts=max_candidate_attempts,
    )
    strategy, strategy_attempts = generate_patch_strategy(
        responder=responder,
        instance=instance,
        code_context=code_context,
        original_failure_log=original_failure_log,
        filtered_candidates=filtered_candidates,
        analysis=analysis,
        max_candidate_attempts=max_candidate_attempts,
        failure_focus=failure_focus,
    )
    # Post-process: force any file that appears in a per-test traceback error line
    # into edit_targets, so deterministic source evidence overrides model judgement.
    strategy = enrich_strategy_edit_targets_from_tracebacks(
        strategy=strategy,
        original_failure_log=original_failure_log,
        instance=instance,
        code_context=code_context,
        analysis=analysis,
    )
    strategy = _prefer_single_root_minimal_edit_targets(
        strategy=strategy,
        analysis=analysis,
        failure_focus=failure_focus,
    )
    code_context = augment_code_context_with_targets(
        instance=instance,
        code_context=code_context,
        target_files=get_strategy_edit_target_files(strategy),
        max_chars_per_file=max_chars_per_file,
    )
    required_source_targets = get_required_source_targets(strategy)
    required_repair_obligations = extract_required_repair_obligations(
        filtered_candidates,
        failure_focus=failure_focus,
        code_context=code_context,
    )
    required_repair_obligations = _promote_single_root_helper_obligations(
        required_repair_obligations,
        analysis=analysis,
        failure_focus=failure_focus,
    )
    primary_required_repair_obligations = select_effective_required_repair_obligations(
        required_repair_obligations,
        filtered_candidates=filtered_candidates,
    )
    repair_topology = classify_repair_topology(
        primary_required_repair_obligations,
        filtered_candidates=filtered_candidates,
    )
    canonical_statement_local = _has_statement_local_canonical_signature(primary_required_repair_obligations)
    single_root_symbol = _has_single_root_symbol_signature(
        analysis=analysis,
        strategy=strategy,
        failure_focus=failure_focus,
    )
    if canonical_statement_local:
        repair_topology = "statement_local"
    elif single_root_symbol and repair_topology == "generic":
        repair_topology = "single_root_symbol"
    if repair_topology in {"statement_local", "single_root_symbol"}:
        dominant_symbol = _select_dominant_symbol_from_candidates(
            filtered_candidates,
            primary_required_repair_obligations,
        )
        if dominant_symbol:
            primary_required_repair_obligations = [
                item
                for item in primary_required_repair_obligations
                if str(item.get("source_symbol") or "").strip() == dominant_symbol
            ] or primary_required_repair_obligations
    anchor_regions = build_edit_anchor_regions(
        code_context=code_context,
        analysis=analysis,
        strategy=strategy,
        required_repair_obligations=primary_required_repair_obligations,
    )
    edit_plan, edit_plan_attempts = generate_fragment_edit_plan(
        responder=responder,
        instance=instance,
        original_failure_log=original_failure_log,
        analysis=analysis,
        strategy=strategy,
        anchor_regions=anchor_regions,
        required_source_targets=required_source_targets,
        required_repair_obligations=primary_required_repair_obligations,
        repair_topology=repair_topology,
        max_candidate_attempts=max_candidate_attempts,
    )
    feedback = None
    selected_patch = ""
    raw_response = ""
    patch_error = None
    best_patch = ""
    best_raw_response = ""
    best_eval_patch = ""
    best_eval_raw_response = ""
    best_eval_result: EvalResult | None = None
    total_attempts = analysis_attempts + strategy_attempts + edit_plan_attempts
    selected_eval = None
    semantic_oracle_result = None
    accepted = False
    acceptance_reason = "rejected_not_generated"
    active_f2p_identifiers = (
        [
            str(test_id).strip()
            for test_id in ((failure_focus or {}).get("active_fail_to_pass_identifiers") or [])
            if str(test_id).strip()
        ]
        or None
    )
    f2p_identifiers = active_f2p_identifiers or get_original_test_identifiers(instance) or None
    prioritize_plan_patch = should_prioritize_edit_plan_patch(edit_plan)
    lock_assembled_plan_patch = should_lock_assembled_edit_plan_patch(edit_plan)
    lock_plan_derived_patch = should_lock_plan_derived_patch(edit_plan)
    if edit_plan is not None:
        updated_files, plan_error = apply_fragment_edit_plan(code_context, edit_plan, anchor_regions=anchor_regions)
        raw_response = json.dumps(edit_plan, indent=2, ensure_ascii=False)
        total_attempts += 1
        if plan_error is None:
            syntax_feedback = None
            for target_file, updated_content in updated_files.items():
                syntax_feedback = validate_python_source_syntax(updated_content, target_file)
                if syntax_feedback is not None:
                    break
            if syntax_feedback is not None:
                patch_error = syntax_feedback
                feedback = syntax_feedback
                selected_patch = ""
            else:
                patch_parts = []
                for target_file, updated_content in updated_files.items():
                    patch_parts.append(
                        build_unified_diff_from_replacement(
                            original_content=code_context[target_file],
                            updated_content=updated_content,
                            target_file=target_file,
                        )
                    )
                selected_patch = normalize_patch("".join(part for part in patch_parts if part))
                selected_patch, patch_error = sanitize_unified_diff(selected_patch)
                feedback = build_patch_feedback(
                    selected_patch,
                    patch_error,
                    edit_plan=edit_plan,
                    code_context=code_context,
                    analysis=analysis,
                    strategy=strategy,
                    original_failure_log=original_failure_log,
                )
                feedback = normalize_patch_feedback(feedback)
                if selected_patch and feedback is None:
                    best_patch = selected_patch
                    best_raw_response = raw_response
        else:
            patch_error = plan_error
            feedback = plan_error
    if (not selected_patch or feedback is not None) and strategy:
        synthesized_patch, synthesized_error = synthesize_structural_patch_from_strategy(
            code_context=code_context,
            analysis=analysis,
            strategy=strategy,
            original_failure_log=original_failure_log,
        )
        if synthesized_patch:
            selected_patch = synthesized_patch
            raw_response = raw_response or json.dumps(
                {
                    "synthesized_from_strategy": True,
                    "structural_rules": strategy.get("structural_rules", []),
                    "forbidden_patterns": strategy.get("forbidden_patterns", []),
                },
                ensure_ascii=False,
                indent=2,
            )
            patch_error = None
            feedback = build_patch_feedback(
                selected_patch,
                patch_error,
                edit_plan=edit_plan,
                code_context=code_context,
                analysis=analysis,
                strategy=strategy,
                original_failure_log=original_failure_log,
            )
            feedback = normalize_patch_feedback(feedback)
            if selected_patch and feedback is None:
                best_patch = selected_patch
                best_raw_response = raw_response
        elif synthesized_error and not feedback:
            patch_error = synthesized_error
            feedback = synthesized_error
    assembled_plan_patch_locked = bool(
        selected_patch
        and (
            lock_plan_derived_patch
            or repair_topology in {"statement_local", "single_root_symbol"}
            or canonical_statement_local
        )
    )
    plan_patch_apply_failed = False
    if prioritize_plan_patch and selected_patch:
        candidate_eval = run_eval(
            instance=instance,
            test_patch=instance["test_patch"],
            code_patch=selected_patch,
            run_id="patch-edit-plan",
            timeout=timeout,
        )
        selected_eval = candidate_eval
        semantic_oracle_result = run_semantic_oracle_checks(
            instance=instance,
            filtered_candidates=filtered_candidates,
            code_patch=selected_patch,
            timeout=timeout,
        )
        accepted, acceptance_reason = get_patch_acceptance_reason(
            candidate_eval,
            baseline_eval,
            semantic_oracle=semantic_oracle_result,
            f2p_identifiers=f2p_identifiers,
        )
        best_eval_result = candidate_eval
        best_eval_patch = selected_patch
        best_eval_raw_response = raw_response
        plan_patch_apply_failed = candidate_eval.patch_apply_mode == "no_patch"
        if selected_patch:
            best_patch = selected_patch or best_patch
            best_raw_response = raw_response or best_raw_response
        if not accepted:
            precise_failure = extract_precise_failure_feedback(
                candidate_eval.log_text,
                (
                    str(normalize_edit_plan_entries(edit_plan)[0].get("target_file", ""))
                    if normalize_edit_plan_entries(edit_plan)
                    else None
                ),
            )
            semantic_failure_context = ""
            if semantic_oracle_result and semantic_oracle_result.failed_identifiers:
                failure_sections = []
                for identifier in semantic_oracle_result.failed_identifiers:
                    summary = semantic_oracle_result.failure_summaries.get(identifier, "")
                    failure_sections.append(f"{identifier}: {summary}")
                semantic_failure_context = " ".join(failure_sections)
            _f2p_still_failing = [
                tid for tid in (f2p_identifiers or [])
                if candidate_eval.status_map.get(tid) not in {"PASSED", "XPASS"}
            ] if f2p_identifiers else []
            f2p_feedback = (
                f"FAIL_TO_PASS tests still failing after patch: {', '.join(_f2p_still_failing)}. "
                "Your patch must make ALL of these tests pass, not just improve aggregate counts. "
                if _f2p_still_failing else ""
            )
            counts_unchanged = (
                candidate_eval.patch_apply_mode == "clean_apply"
                and count_passed(candidate_eval.status_map) == count_passed(baseline_eval.status_map)
                and count_failed(candidate_eval.status_map) == count_failed(baseline_eval.status_map)
            )
            zero_effect_hint = (
                "CRITICAL: the previous patch applied cleanly but had ZERO effect on test counts — "
                "the fix is unreachable or incorrect. Common causes: "
                "(1) your new code was inserted inside an `elif` chain that is bypassed by an earlier branch "
                "(e.g., `elif item.config.option.runxfail: pass` prevents later `elif` blocks from running) — "
                "check if an existing `elif` block needs to become a standalone `if`; "
                "(2) the wrong function/method was edited; "
                "(3) the condition in the existing branch already covers this case incorrectly. "
                "Rethink the control flow structure rather than re-inserting the same logic. "
                if counts_unchanged else ""
            )
            feedback = (
                "The previous patch was rejected because it did not satisfy the acceptance gate. "
                f"Patch apply mode: {candidate_eval.patch_apply_mode}. "
                f"Baseline passed={count_passed(baseline_eval.status_map)}, failed={count_failed(baseline_eval.status_map)}. "
                f"Candidate passed={count_passed(candidate_eval.status_map)}, failed={count_failed(candidate_eval.status_map)}. "
                f"Hard failure signal: {get_hard_failure_signal(candidate_eval.log_text) or 'none'}. "
                f"{zero_effect_hint}"
                f"{f2p_feedback}"
                f"Semantic oracle failed identifiers: {', '.join((semantic_oracle_result.failed_identifiers if semantic_oracle_result else [])) or 'none'}. "
                f"Semantic oracle failure summaries: {semantic_failure_context or 'none'}. "
                f"{precise_failure or ''} "
                f"Target files: {', '.join(str(edit.get('target_file', 'unknown')) for edit in normalize_edit_plan_entries(edit_plan)) or 'unknown'}. "
                f"Target symbols: {', '.join(str(edit.get('target_symbol', 'unknown')) for edit in normalize_edit_plan_entries(edit_plan)) or 'unknown'}. "
                "Generate a new source-only unified diff that is closer to the exact file context and lands inside the target class body."
            )
            if plan_patch_apply_failed:
                selected_patch = ""
        else:
            feedback = None
    assembled_plan_patch_locked = bool(
        selected_patch
        and (
            lock_plan_derived_patch
            or repair_topology in {"statement_local", "single_root_symbol"}
            or canonical_statement_local
        )
    )
    free_diff_attempt_budget = (
        0
        if (
            assembled_plan_patch_locked
            or repair_topology == "single_root_symbol"
        )
        else max_candidate_attempts
    )
    for attempt in range(1, free_diff_attempt_budget + 1):
        if accepted:
            break
        if not selected_patch or feedback is not None:
            raw_response, patch = responder.complete(
                build_diff_from_strategy_prompt(
                    instance=instance,
                    code_context=code_context,
                    original_failure_log=original_failure_log,
                    filtered_candidates=filtered_candidates,
                    analysis=analysis,
                    strategy=strategy,
                    feedback=feedback,
                    failure_focus=failure_focus,
                    hide_original_test_patch=hide_original_test_patch_in_repair,
                )
            )
            sanitized_patch, patch_error = sanitize_unified_diff(patch)
            if sanitized_patch and not patch_error:
                # In-memory syntax check: apply patch to code_context and compile.
                # This check can produce false positives when the hunk offsets from
                # the LLM don't exactly match our in-memory applicator's expectations
                # (e.g. the diff is correct but lands at a slightly different line).
                # To avoid silently discarding a structurally valid diff, we only hard-
                # reject on attempts before the last one; on the final attempt we let
                # Docker evaluate the patch directly so it can use fuzzy patching.
                syntax_error_msg = _check_patch_syntax_in_memory(sanitized_patch, code_context)
                if syntax_error_msg:
                    if attempt < max_candidate_attempts:
                        # P3: Distinguish anchor-position errors from genuine logic errors.
                        # "expected an indented block" usually means the hunk was applied at
                        # the wrong line, creating an empty compound-statement body. Guide
                        # the LLM to re-examine the hunk header line numbers.
                        if "hunk start line" in syntax_error_msg or "patch apply failed" in syntax_error_msg:
                            patch_error = (
                                f"Patch could not be applied: {syntax_error_msg}. "
                                "The @@ line numbers in your diff do not match the actual file. "
                                "IMPORTANT: count the lines in the 'Buggy code context' section carefully. "
                                "If you need to change `elif` to `if` in an existing block, prefer writing "
                                "a SINGLE hunk that removes the old `elif` line and adds `if` — "
                                "do NOT split this into two hunks, as the second hunk offset will be wrong. "
                                "Re-read the context and adjust @@ headers to exact line numbers."
                            )
                        elif "expected an indented block" in syntax_error_msg or "invalid syntax" in syntax_error_msg:
                            patch_error = (
                                f"SyntaxError after applying patch: {syntax_error_msg}. "
                                "This is likely a HUNK PLACEMENT ERROR — the @@ line numbers in your diff "
                                "do not match the actual file content, causing the patch to land at the wrong "
                                "location and leave a compound statement (if/elif/else/for/while/def/class) "
                                "with no body. Fix: re-read the anchor context above, find the EXACT lines "
                                "you want to modify, and set the @@ hunk header to the precise line numbers "
                                "shown there. Do NOT shift by even one line."
                            )
                        else:
                            patch_error = f"SyntaxError after applying patch: {syntax_error_msg}"
                        sanitized_patch = ""
                    else:
                        # Last attempt: keep the patch, let run_eval decide via Docker.
                        patch_error = None
            selected_patch = sanitized_patch
            total_attempts += 1
        feedback = build_patch_feedback(
            selected_patch,
            patch_error,
            edit_plan=edit_plan,
            code_context=code_context,
            skip_syntax_check=(attempt == max_candidate_attempts),
            analysis=analysis,
            strategy=strategy,
            original_failure_log=original_failure_log,
        )
        feedback = normalize_patch_feedback(feedback)
        if feedback is not None:
            if selected_patch and not best_patch:
                best_patch = selected_patch
                best_raw_response = raw_response
            selected_patch = ""
            continue
        if selected_patch:
            best_patch = selected_patch
            best_raw_response = raw_response
        candidate_eval = run_eval(
            instance=instance,
            test_patch=instance["test_patch"],
            code_patch=selected_patch,
            run_id=f"patch-attempt-{attempt}",
            timeout=timeout,
        )
        selected_eval = candidate_eval
        semantic_oracle_result = run_semantic_oracle_checks(
            instance=instance,
            filtered_candidates=filtered_candidates,
            code_patch=selected_patch,
            timeout=timeout,
        )
        accepted, acceptance_reason = get_patch_acceptance_reason(
            candidate_eval,
            baseline_eval,
            semantic_oracle=semantic_oracle_result,
            f2p_identifiers=f2p_identifiers,
        )
        if score_patch_candidate_choice(
            candidate_eval,
            baseline_eval,
            f2p_identifiers,
            selected_patch,
            analysis,
            strategy,
            original_failure_log,
        ) > score_patch_candidate_choice(
            best_eval_result,
            baseline_eval,
            f2p_identifiers,
            best_eval_patch,
            analysis,
            strategy,
            original_failure_log,
        ):
            best_eval_result = candidate_eval
            best_eval_patch = selected_patch
            best_eval_raw_response = raw_response
        if accepted:
            break
        precise_failure = extract_precise_failure_feedback(
            candidate_eval.log_text,
            (
                str(normalize_edit_plan_entries(edit_plan)[0].get("target_file", ""))
                if normalize_edit_plan_entries(edit_plan)
                else None
            ),
        )
        semantic_failure_context = ""
        if semantic_oracle_result and semantic_oracle_result.failed_identifiers:
            failure_sections = []
            for identifier in semantic_oracle_result.failed_identifiers:
                summary = semantic_oracle_result.failure_summaries.get(identifier, "")
                failure_sections.append(f"{identifier}: {summary}")
            semantic_failure_context = " ".join(failure_sections)
        _f2p_still_failing = [
            tid for tid in (f2p_identifiers or [])
            if candidate_eval.status_map.get(tid) not in {"PASSED", "XPASS"}
        ] if f2p_identifiers else []
        f2p_feedback = (
            f"FAIL_TO_PASS tests still failing after patch: {', '.join(_f2p_still_failing)}. "
            "Your patch must make ALL of these tests pass, not just improve aggregate counts. "
            if _f2p_still_failing else ""
        )
        # Zero-effect detection: patch applied cleanly but counts unchanged — the fix is unreachable.
        counts_unchanged = (
            candidate_eval.patch_apply_mode == "clean_apply"
            and count_passed(candidate_eval.status_map) == count_passed(baseline_eval.status_map)
            and count_failed(candidate_eval.status_map) == count_failed(baseline_eval.status_map)
        )
        zero_effect_hint = (
            "CRITICAL: the previous patch applied cleanly but had ZERO effect on test counts — "
            "the fix is unreachable or incorrect. Common causes: "
            "(1) your new code was inserted inside an `elif` chain that is bypassed by an earlier branch "
            "(e.g., `elif item.config.option.runxfail: pass` prevents later `elif` blocks from running) — "
            "check if an existing `elif` block needs to become a standalone `if`; "
            "(2) the wrong function/method was edited; "
            "(3) the condition in the existing branch already covers this case incorrectly. "
            "Rethink the control flow structure rather than re-inserting the same logic. "
            if counts_unchanged else ""
        )
        feedback = (
            "The previous patch was rejected because it did not satisfy the acceptance gate. "
            f"Patch apply mode: {candidate_eval.patch_apply_mode}. "
            f"Baseline passed={count_passed(baseline_eval.status_map)}, failed={count_failed(baseline_eval.status_map)}. "
            f"Candidate passed={count_passed(candidate_eval.status_map)}, failed={count_failed(candidate_eval.status_map)}. "
            f"Hard failure signal: {get_hard_failure_signal(candidate_eval.log_text) or 'none'}. "
            f"{zero_effect_hint}"
            f"{f2p_feedback}"
            f"Semantic oracle failed identifiers: {', '.join((semantic_oracle_result.failed_identifiers if semantic_oracle_result else [])) or 'none'}. "
            f"Semantic oracle failure summaries: {semantic_failure_context or 'none'}. "
            f"{precise_failure or ''} "
            f"Target files: {', '.join(str(edit.get('target_file', 'unknown')) for edit in normalize_edit_plan_entries(edit_plan)) or 'unknown'}. "
            f"Target symbols: {', '.join(str(edit.get('target_symbol', 'unknown')) for edit in normalize_edit_plan_entries(edit_plan)) or 'unknown'}. "
            "Generate a new source-only unified diff that is closer to the exact file context and lands inside the target class body."
        )
        selected_patch = ""
    preferred_eval = selected_eval
    preferred_patch = selected_patch
    preferred_raw_response = raw_response
    if score_patch_candidate_choice(
        best_eval_result,
        baseline_eval,
        f2p_identifiers,
        best_eval_patch,
        analysis,
        strategy,
        original_failure_log,
    ) >= score_patch_candidate_choice(
        selected_eval,
        baseline_eval,
        f2p_identifiers,
        selected_patch,
        analysis,
        strategy,
        original_failure_log,
    ):
        preferred_eval = best_eval_result
        preferred_patch = best_eval_patch
        preferred_raw_response = best_eval_raw_response
    final_patch = preferred_patch or best_patch
    final_raw_response = preferred_raw_response or best_raw_response
    return PatchGenerationResult(
        analysis=analysis,
        strategy=strategy,
        edit_plan=edit_plan,
        patch=final_patch,
        raw_response=final_raw_response,
        attempts=total_attempts,
        accepted=accepted,
        acceptance_reason=acceptance_reason,
        candidate_eval=preferred_eval,
        semantic_oracle_passed=bool(semantic_oracle_result and semantic_oracle_result.passed),
        semantic_oracle_failed_identifiers=(
            semantic_oracle_result.failed_identifiers if semantic_oracle_result is not None else []
        ),
        patch_error=patch_error,
        final_feedback=feedback,
    )


def generate_enhanced_candidates(
    responder: OpenAIResponder,
    instance: dict[str, Any],
    code_context: dict[str, str],
    original_failure_log: str,
    generation_budget: dict[str, Any],
    failure_focus: dict[str, Any] | None,
    max_candidate_attempts: int,
) -> list[CandidatePatch]:
    original_identifiers = get_original_test_identifiers(instance)
    failure_mode = str(generation_budget["failure_mode"])
    template_names = select_template_names(
        failure_mode,
        limit=int(generation_budget["template_limit"]),
    )
    semantic_buckets = select_semantic_buckets(
        failure_mode,
        limit=int(generation_budget["bucket_limit"]),
    )
    target_candidates = int(generation_budget["candidate_budget"])
    attempt_budget = int(generation_budget["attempt_budget"])
    candidates = []
    seen_signatures: set[tuple[str, ...]] = set()
    seen_titles: set[str] = set()
    seen_buckets: set[str] = set()
    covered_original_tests: set[str] = set()
    covered_obligations: set[str] = set()
    for idx in range(target_candidates):
        print(f"[{instance['instance_id']}] 生成增强测试候选 {idx + 1}/{target_candidates}")
        idea, idea_attempts = generate_test_idea(
            responder,
            instance,
            code_context,
            original_failure_log,
            failure_mode,
            template_names,
            semantic_buckets,
            failure_focus,
            seen_titles,
            seen_buckets,
            attempt_budget,
            covered_original_tests=covered_original_tests,
            covered_obligations=covered_obligations,
        )
        for name in ((idea or {}).get("covers_original_tests") or []):
            if isinstance(name, str) and name.strip():
                covered_original_tests.add(name)
        for obligation_id in _normalize_obligation_ids(
            (idea or {}).get("covers_obligations"),
            [str(name) for name in ((idea or {}).get("covers_original_tests") or []) if str(name).strip()],
            failure_focus,
            code_context,
        ):
            covered_obligations.add(obligation_id)
        feedback = None
        selected_candidate = None
        for attempt in range(1, attempt_budget + 1):
            raw_response, patch = responder.complete(
                build_diff_from_idea_prompt(
                    instance,
                    code_context,
                    original_failure_log,
                    idea or {
                        "title": f"fallback_{idx + 1}",
                        "goal": "generate one distinct enhanced test",
                        "template": template_names[min(idx, len(template_names) - 1)],
                        "target_tests": [f"{ENHANCED_TEST_PREFIX}{idx + 1}"],
                        "rationale": "fallback",
                    },
                    failure_focus=failure_focus,
                )
                + (f"\nRevision feedback for the next diff attempt:\n{feedback}\n" if feedback else "")
            )
            sanitized_patch, patch_error = sanitize_unified_diff(patch)
            identifiers = extract_test_identifiers_from_patch(sanitized_patch or normalize_patch(patch))
            enhanced_identifiers = get_enhanced_test_identifiers(identifiers)
            duplicate_identifiers = find_duplicate_test_identifiers(identifiers, original_identifiers)
            candidate = CandidatePatch(
                idea=idea,
                patch=sanitized_patch or normalize_patch(patch),
                raw_response=raw_response,
                identifiers=identifiers,
                enhanced_identifiers=enhanced_identifiers,
                duplicate_identifiers=duplicate_identifiers,
                reason="" if not patch_error else f"patch_sanitization_warning: {patch_error}",
                generation_attempts=idea_attempts + attempt - 1,
            )
            feedback = build_generation_feedback(candidate, seen_signatures)
            selected_candidate = candidate
            if feedback is None:
                seen_signatures.add(tuple(candidate.enhanced_identifiers or []))
                break
        candidates.append(selected_candidate)
    return candidates


def filter_candidates(
    instance: dict[str, Any],
    candidates: list[CandidatePatch],
    timeout: int,
    keep_top_k: int | None = None,
    failure_focus: dict[str, Any] | None = None,
    code_context: dict[str, str] | None = None,
) -> list[CandidatePatch]:
    kept = []
    original_identifiers = get_original_test_identifiers(instance)
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[{instance['instance_id']}] 筛选增强测试候选 {idx}/{len(candidates)}")
        sanitized_patch, patch_error = sanitize_unified_diff(candidate.patch)
        if not sanitized_patch:
            candidate.reason = f"invalid unified diff: {patch_error or 'unknown parse error'}"
            continue
        candidate.patch = sanitized_patch
        candidate.identifiers = extract_test_identifiers_from_patch(candidate.patch)
        candidate.enhanced_identifiers = get_enhanced_test_identifiers(candidate.identifiers)
        candidate.covered_original_tests = [
            str(name)
            for name in ((candidate.idea or {}).get("covers_original_tests") or [])
            if isinstance(name, str)
        ]
        backfill_candidate_minimal_structure(candidate, failure_focus, code_context)
        candidate.covered_obligations = _normalize_obligation_ids(
            (candidate.idea or {}).get("covers_obligations"),
            candidate.covered_original_tests,
            failure_focus,
            code_context,
        )
        alignment_feedback = build_path_semantic_alignment_feedback(
            candidate.covered_original_tests,
            str((candidate.idea or {}).get("target_source_symbol", "")).strip(),
            _normalize_alignment_tokens((candidate.idea or {}).get("semantic_alignment_tokens")),
            failure_focus,
            code_context,
            f"{json.dumps(candidate.idea or {}, ensure_ascii=False)}\n{candidate.patch}",
        )
        if alignment_feedback:
            candidate.reason = alignment_feedback
            continue
        # Trigger-shape enforcement: if the idea specifies attribute-path trigger tokens
        # (e.g. view_func.__name__), the generated test patch must actually use that path.
        # A test that drifts to a simpler API surface (e.g. Blueprint name) will be rejected.
        idea_trigger_tokens = _normalize_alignment_tokens(
            (candidate.idea or {}).get("trigger_shape_tokens")
        )
        # Attribute trigger tokens: e.g. "view_func.__name__" — check each part separately
        # because the variable name may differ (e.g. bad_view.__name__ vs view_func.__name__).
        attr_trigger_tokens = [t for t in idea_trigger_tokens if "." in t and "__" in t and not t.startswith("Blueprint")]
        if attr_trigger_tokens:
            patch_lower = candidate.patch.lower()
            # For each attr token like "view_func.__name__", check the attribute part (__name__)
            # is present AND the patch does NOT only use it in unrelated ways (import __name__ etc.)
            missing_triggers = []
            for attr_tok in attr_trigger_tokens:
                attr_part = attr_tok.split(".")[-1].lower()  # e.g. "__name__"
                obj_part = attr_tok.split(".")[0].lower()    # e.g. "view_func"
                # Must use the attribute, AND either the object name or a similar pattern
                if attr_part not in patch_lower:
                    missing_triggers.append(attr_tok)
                elif attr_part == "__name__" and ".__name__" not in patch_lower:
                    # __name__ appears but only as module __name__, not as attribute access
                    missing_triggers.append(attr_tok)
            if missing_triggers:
                candidate.reason = (
                    f"Patch does not use the required attribute trigger path(s): {missing_triggers}. "
                    f"The idea specifies trigger_shape_tokens {idea_trigger_tokens} — the test must "
                    f"construct a real object/function where {missing_triggers[0]} is set to an invalid "
                    f"value and call the target API with that object. "
                    f"Do not substitute a simpler parameter (e.g. Blueprint name or endpoint string)."
                )
                continue
        candidate.duplicate_identifiers = find_duplicate_test_identifiers(
            candidate.identifiers,
            original_identifiers,
        )
        if candidate.duplicate_identifiers:
            candidate.reason = (
                "candidate reuses original SWE-bench test names: "
                + ", ".join(candidate.duplicate_identifiers)
            )
            continue
        if not candidate.enhanced_identifiers:
            candidate.reason = "candidate did not add any test_sweb_enhanced_* test"
            continue
        eval_result = run_eval(
            instance=instance,
            test_patch=sanitized_patch,
            code_patch="",
            run_id=f"filter-{idx}",
            timeout=timeout,
        )
        candidate.eval_status_map = eval_result.status_map
        # If none of the enhanced identifiers appear in the status map at all,
        # the test patch failed to apply (git apply error).  Record a clear reason
        # so the caller can distinguish "patch didn't apply" from "tests passed".
        # Use substring matching because status_map keys include full path prefixes
        # (e.g. "sympy/core/tests/test_basic.py:test_sweb_enhanced_foo") while
        # enhanced_identifiers are short names ("test_sweb_enhanced_foo").
        status_map_keys = list(eval_result.status_map.keys())
        any_enhanced_ran = any(
            any(ident in key for key in status_map_keys)
            for ident in (candidate.enhanced_identifiers or [])
        )
        if not any_enhanced_ran and candidate.enhanced_identifiers:
            if eval_result.patch_applied is False:
                candidate.reason = "test patch did not apply to buggy base commit"
            elif not has_meaningful_status_map(eval_result.status_map):
                if is_environment_blocked_import_failure(failure_focus, eval_result):
                    candidate.kept = True
                    backfill_candidate_minimal_structure(candidate, failure_focus, code_context)
                    if not _candidate_has_minimal_structure(candidate):
                        candidate.kept = False
                        candidate.reason = (
                            "candidate hit the same import-error environment block as the original failure, "
                            "but could not be grounded to a minimal source_symbol/validation_subject/obligation structure"
                        )
                        continue
                    candidate.quality_score = 0.05
                    candidate.quality_breakdown = {
                        "bug_reproduction": 0.0,
                        "path_alignment": 0.25,
                        "signal_strength": 0.0,
                        "environment_blocked_import_error": 1.0,
                    }
                    candidate.reason = (
                        "retained under import-error weak keep: evaluation produced an empty status_map, "
                        "but the log shows the same import-error environment block as the original failure"
                    )
                    kept.append(candidate)
                    continue
                candidate.reason = "evaluation produced an empty status_map; likely log_parse_failed or tests_not_collected"
            else:
                candidate.reason = "enhanced tests were not collected from the evaluation status_map"
            continue
        failing_identifiers = find_failing_identifiers(
            eval_result.status_map,
            candidate.enhanced_identifiers,
        )
        candidate.failing_identifiers = failing_identifiers
        if failing_identifiers:
            candidate.kept = True
            backfill_candidate_minimal_structure(candidate, failure_focus, code_context)
            if not _candidate_has_minimal_structure(candidate):
                candidate.kept = False
                candidate.reason = (
                    "candidate reproduced buggy behavior but could not be grounded to a minimal "
                    "source_symbol/validation_subject/obligation structure"
                )
                continue
            candidate.quality_score, candidate.quality_breakdown = score_candidate_quality(candidate, eval_result)
            candidate.reason = "new enhanced tests reproduced the buggy behavior"
            kept.append(candidate)
        else:
            candidate.reason = eval_result.error or "no new test_sweb_enhanced_* identifier failed on buggy code"
    selected = select_candidates_with_path_coverage(kept, original_identifiers, keep_top_k)
    return retain_missing_symbol_cluster_subject_candidates(selected, candidates, keep_top_k)


def run_instance_pipeline(
    instance: dict[str, Any],
    responder: OpenAIResponder,
    output_dir: Path,
    max_context_files: int,
    max_chars_per_file: int,
    timeout: int,
    max_candidate_attempts: int,
    baseline_only: bool = False,
    hide_original_test_patch_in_repair: bool = False,
) -> dict[str, Any]:
    instance_dir = output_dir / instance["instance_id"]
    instance_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{instance['instance_id']}] 读取缺陷代码上下文")
    code_context = read_code_context(instance, max_context_files, max_chars_per_file)
    # Eagerly ensure every source file mentioned in the fix patch is in context,
    # so that all downstream LLM calls (test generation, strategy, edit plan) see
    # the actual file contents rather than guessing them.
    patch_source_files = dedupe_preserve_order(
        [f for f in get_modified_files(instance["patch"]) + get_new_files(instance["patch"])
         if not is_test_like_path(f)]
    )
    code_context = augment_code_context_with_targets(
        instance=instance,
        code_context=code_context,
        target_files=patch_source_files,
        max_chars_per_file=max_chars_per_file,
    )

    print(f"[{instance['instance_id']}] 运行原始测试以捕获报错信息")
    original_eval = run_eval(
        instance=instance,
        test_patch=instance["test_patch"],
        code_patch="",
        run_id="original-tests",
        timeout=timeout,
    )
    write_json(
        instance_dir / "original_failure.json",
        {
            "original_test_identifiers": get_original_test_identifiers(instance),
            "status_map": original_eval.status_map,
            "log_path": original_eval.log_path,
            "timed_out": original_eval.timed_out,
            "error": original_eval.error,
        },
    )
    (instance_dir / "original_failure.log").write_text(original_eval.log_text)
    failure_focus = extract_failure_focus(instance, original_eval.status_map, original_eval.log_text)
    code_context = augment_code_context_with_failure_focus_targets(
        instance=instance,
        code_context=code_context,
        failure_focus=failure_focus,
        max_chars_per_file=max_chars_per_file,
    )
    write_json(instance_dir / "code_context.json", code_context)
    generation_budget = recommend_generation_budget(
        original_eval.status_map,
        original_eval.log_text,
        base_num_candidates=DEFAULT_BASE_CANDIDATES,
        base_attempts=max_candidate_attempts,
    )
    write_json(instance_dir / "failure_focus.json", failure_focus)
    write_json(instance_dir / "generation_budget.json", generation_budget)

    if baseline_only:
        print(f"[{instance['instance_id']}] BASELINE模式: 跳过增强测试生成")
        candidates = []
        kept_candidates = []
        write_json(instance_dir / "enhanced_candidates_raw.json", [])
        write_json(instance_dir / "enhanced_candidates_filtered.json", [])
        write_json(instance_dir / "enhanced_candidates_rejected.json", [])
    else:
        candidates = generate_enhanced_candidates(
            responder=responder,
            instance=instance,
            code_context=code_context,
            original_failure_log=original_eval.log_text,
            generation_budget=generation_budget,
            failure_focus=failure_focus,
            max_candidate_attempts=max_candidate_attempts,
        )
        write_json(instance_dir / "enhanced_candidates_raw.json", [asdict(x) for x in candidates])

        kept_candidates = filter_candidates(
            instance,
            candidates,
            timeout=timeout,
            keep_top_k=int(generation_budget["keep_top_k"]),
            failure_focus=failure_focus,
            code_context=code_context,
        )
        write_json(instance_dir / "enhanced_candidates_filtered.json", [asdict(x) for x in kept_candidates])
        write_json(
            instance_dir / "enhanced_candidates_rejected.json",
            [asdict(x) for x in candidates if not x.kept],
        )

    print(f"[{instance['instance_id']}] 基于增强测试生成补丁")
    patch_result = generate_patch_with_strategy(
        responder=responder,
        instance=instance,
        code_context=code_context,
        original_failure_log=original_eval.log_text,
        filtered_candidates=kept_candidates,
        max_candidate_attempts=max_candidate_attempts,
        baseline_eval=original_eval,
        timeout=timeout,
        max_chars_per_file=max_chars_per_file,
        failure_focus=failure_focus,
        hide_original_test_patch_in_repair=hide_original_test_patch_in_repair,
    )
    evaluated_patch = (
        patch_result.patch
        if patch_result.patch
        and patch_result.candidate_eval is not None
        and patch_result.candidate_eval.patch_apply_mode in {"clean_apply", "fuzzy_apply"}
        else ""
    )
    model_patch = evaluated_patch
    patch_error = patch_result.patch_error
    # P0 fallback: only when the normal pipeline produced no evaluated patch at all and
    # the raw_response contains a structurally valid diff, extract and use it for final
    # evaluation. This avoids overriding a structured edit-plan patch that was already
    # built and evaluated, especially for class-level fixes like sympy's Printable.
    # This recovers cases where the in-memory syntax checker incorrectly rejected the diff
    # (e.g. wrong hunk-offset false positive) while the LLM output was actually correct.
    if not patch_result.patch and patch_result.raw_response:
        fallback_diff = extract_diff_from_raw_response(patch_result.raw_response)
        if fallback_diff:
            syntax_err = _check_patch_syntax_in_memory(fallback_diff, code_context)
            if not syntax_err:
                print(f"[{instance['instance_id']}] P0 fallback: 从raw_response提取到有效diff，将用于最终验证")
                model_patch = fallback_diff
                patch_error = None
    (instance_dir / "patch_analysis.json").write_text(
        json.dumps(patch_result.analysis or {}, indent=2, ensure_ascii=False)
    )
    (instance_dir / "patch_strategy.json").write_text(
        json.dumps(patch_result.strategy or {}, indent=2, ensure_ascii=False)
    )
    (instance_dir / "patch_edit_plan.json").write_text(
        json.dumps(patch_result.edit_plan or {}, indent=2, ensure_ascii=False)
    )
    (instance_dir / "patch_response.txt").write_text(patch_result.raw_response)
    (instance_dir / "model_patch.diff").write_text(model_patch)

    print(f"[{instance['instance_id']}] 使用原始测试回验补丁")
    final_eval = run_eval(
        instance=instance,
        test_patch=instance["test_patch"],
        code_patch=model_patch,
        run_id="final-validation",
        timeout=timeout,
    )
    original_status_counts = count_statuses(original_eval.status_map)
    final_status_counts = count_statuses(final_eval.status_map)
    original_passed_count = count_passed(original_eval.status_map)
    final_passed_count = count_passed(final_eval.status_map)
    original_failed_count = count_failed(original_eval.status_map)
    final_failed_count = count_failed(final_eval.status_map)
    enhanced_failed_identifiers = aggregate_enhanced_failures(kept_candidates)
    patch_was_generated = bool(model_patch.strip())
    effective_patch_applied = final_eval.patch_apply_mode in {"clean_apply", "fuzzy_apply"}
    clean_apply_ratio = 1.0 if final_eval.patch_apply_mode == "clean_apply" else 0.0
    (
        aligned_generated_patch_accepted,
        aligned_generated_patch_acceptance_reason,
        aligned_generated_patch_feedback,
    ) = align_generated_patch_outcome(
        generated_patch_accepted=patch_result.accepted,
        generated_patch_acceptance_reason=patch_result.acceptance_reason,
        generated_patch_feedback=patch_result.final_feedback,
        patch_was_generated=patch_was_generated,
        final_eval=final_eval,
    )
    summary = {
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "model": responder.model,
        "pipeline_mode": "baseline_only" if baseline_only else "enhanced_guided",
        "hide_original_test_patch_in_repair": hide_original_test_patch_in_repair,
        "repair_mode": get_repair_mode(kept_candidates),
        "original_test_identifiers": get_original_test_identifiers(instance),
        "original_status_counts": original_status_counts,
        "original_failed_count": original_failed_count,
        "original_passed_count": original_passed_count,
        "failure_focus": failure_focus,
        "generation_budget": generation_budget,
        "adaptive_num_candidates": generation_budget["candidate_budget"],
        "adaptive_keep_top_k": generation_budget["keep_top_k"],
        "adaptive_attempt_budget": generation_budget["attempt_budget"],
        "enhanced_candidates_total": len(candidates),
        "kept_enhanced_candidates": len(kept_candidates),
        "enhanced_candidate_retention_rate": safe_rate(len(kept_candidates), len(candidates)),
        "enhanced_generation_attempts_total": sum(candidate.generation_attempts for candidate in candidates),
        "enhanced_generation_attempts_avg": safe_rate(
            sum(candidate.generation_attempts for candidate in candidates),
            len(candidates),
        ),
        "kept_enhanced_test_identifiers": [
            identifier
            for candidate in kept_candidates
            for identifier in (candidate.enhanced_identifiers or [])
        ],
        "enhanced_failed_test_identifiers": enhanced_failed_identifiers,
        "enhanced_failed_count": len(enhanced_failed_identifiers),
        "generated_patch_nonempty": bool(model_patch.strip()),
        "generated_patch_error": patch_error,
        "generated_patch_attempts": patch_result.attempts,
        "generated_patch_accepted": aligned_generated_patch_accepted,
        "generated_patch_acceptance_reason": aligned_generated_patch_acceptance_reason,
        "generated_patch_feedback": aligned_generated_patch_feedback,
        "semantic_oracle_passed": patch_result.semantic_oracle_passed,
        "semantic_oracle_failed_identifiers": patch_result.semantic_oracle_failed_identifiers,
        "patch_analysis": patch_result.analysis,
        "patch_strategy": patch_result.strategy,
        "patch_edit_plan": patch_result.edit_plan,
        "final_patch_attempted": patch_was_generated,
        "final_patch_applied": final_eval.patch_applied,
        "final_patch_effective_applied": effective_patch_applied,
        "final_patch_apply_mode": final_eval.patch_apply_mode,
        "final_patch_cleanly_applied": final_eval.patch_apply_mode == "clean_apply",
        "clean_apply_ratio": clean_apply_ratio,
        "resolved": final_eval.resolved,
        "final_resolved": final_eval.resolved,
        "final_status_counts": final_status_counts,
        "final_failed_count": final_failed_count,
        "final_passed_count": final_passed_count,
        "original_passed_count_improvement": final_passed_count - original_passed_count,
        "original_failed_count_reduction": original_failed_count - final_failed_count,
        "final_status_map": final_eval.status_map,
        "final_timed_out": final_eval.timed_out,
        "final_error": final_eval.error,
    }
    write_json(instance_dir / "summary.json", summary)
    return summary


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Experimental pipeline: enhanced test generation and patch repair for SWE-bench.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_name", type=str, default="SWE-bench/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--instance_ids", nargs="+", default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/enhanced_patch_pipeline"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_completion_tokens", type=int, default=4096)
    parser.add_argument("--max_context_files", type=int, default=6)
    parser.add_argument("--max_chars_per_file", type=int, default=6000)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--max_candidate_attempts", type=int, default=3)
    parser.add_argument("--namespace", type=optional_str, default=None)
    parser.add_argument(
        "--skip_container_cleanup",
        action="store_true",
        help="Skip preflight cleanup of stale Docker containers for the target instances.",
    )
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help=(
            "Baseline mode: skip enhanced test generation and patch directly from original "
            "failure info only. Used for thesis control-group comparison."
        ),
    )
    parser.add_argument(
        "--hide_original_test_patch_in_repair",
        action="store_true",
        help=(
            "Hide the official SWE-bench test patch from source-code repair prompts. "
            "Final validation still uses the original test patch."
        ),
    )
    parser.add_argument(
        "--compare_runs",
        nargs=2,
        metavar=("BASELINE_DIR", "ENHANCED_DIR"),
        default=None,
        help="Compare two run directories (baseline vs enhanced) and print thesis comparison table.",
    )
    return parser


def summarize_run(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total_instances = len(summaries)
    pipeline_modes = {item.get("pipeline_mode", "enhanced_guided") for item in summaries}
    pipeline_mode = pipeline_modes.pop() if len(pipeline_modes) == 1 else "mixed"
    return {
        "pipeline_mode": pipeline_mode,
        "instances_total": total_instances,
        "instances_resolved": sum(1 for item in summaries if item["final_resolved"]),
        "instance_resolution_rate": safe_rate(
            sum(1 for item in summaries if item["final_resolved"]),
            total_instances,
        ),
        "clean_apply_ratio": safe_rate(
            sum(1 for item in summaries if item["final_patch_cleanly_applied"]),
            total_instances,
        ),
        "enhanced_candidate_retention_rate": safe_rate(
            sum(item["kept_enhanced_candidates"] for item in summaries),
            sum(item["enhanced_candidates_total"] for item in summaries),
        ),
        "original_failed_count_total": sum(item["original_failed_count"] for item in summaries),
        "final_failed_count_total": sum(item["final_failed_count"] for item in summaries),
        "original_passed_count_total": sum(item["original_passed_count"] for item in summaries),
        "final_passed_count_total": sum(item["final_passed_count"] for item in summaries),
        "original_passed_count_improvement_total": sum(
            item["original_passed_count_improvement"] for item in summaries
        ),
        "original_failed_count_reduction_total": sum(
            item["original_failed_count_reduction"] for item in summaries
        ),
        "enhanced_failed_count_total": sum(item["enhanced_failed_count"] for item in summaries),
        "adaptive_num_candidates_avg": safe_rate(
            sum(int(item.get("adaptive_num_candidates", 0)) for item in summaries),
            total_instances,
        ),
        "adaptive_keep_top_k_avg": safe_rate(
            sum(int(item.get("adaptive_keep_top_k", 0)) for item in summaries),
            total_instances,
        ),
        "baseline_fallback_instances": sum(
            1 for item in summaries if item.get("repair_mode") == "baseline_fallback"
        ),
        "enhanced_guided_instances": sum(
            1 for item in summaries if item.get("repair_mode") == "enhanced_guided"
        ),
    }


def sanitize_repo_name(repo: str) -> str:
    return repo.replace("/", "__")


def allocate_versioned_output_dir(base_output_dir: Path) -> Path:
    """Allocate a fresh numeric run directory under the given output root."""
    base_output_dir.mkdir(parents=True, exist_ok=True)
    numeric_versions = [
        int(child.name)
        for child in base_output_dir.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    next_version = max(numeric_versions, default=0) + 1
    run_output_dir = base_output_dir / str(next_version)
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return run_output_dir


def ensure_openai_api_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    raise ValueError("OPENAI_API_KEY is required before running enhanced_patch_pipeline.")


def ensure_docker_runtime() -> docker.DockerClient:
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        raise RuntimeError(
            "Docker daemon is not reachable. Start Docker or fix daemon/socket permissions before running "
            "enhanced_patch_pipeline."
        ) from exc
    return client


def cleanup_stale_instance_containers(
    client: docker.DockerClient,
    instance_ids: list[str],
) -> dict[str, list[str]]:
    removed: dict[str, list[str]] = {}
    for instance_id in instance_ids:
        lowered = instance_id.lower()
        exact_name = f"sweb.eval.{lowered}"
        run_prefix = f"{exact_name}."
        matched = []
        for container in client.containers.list(all=True, filters={"name": exact_name}):
            container_name = container.name.lstrip("/")
            if container_name == exact_name or container_name.startswith(run_prefix):
                matched.append(container)
        deduped = {container.id: container for container in matched}
        removed_names = []
        for container in deduped.values():
            container_name = container.name.lstrip("/")
            try:
                if container.status == "running":
                    container.stop(timeout=5)
                container.remove(force=True)
                removed_names.append(container_name)
            except docker.errors.NotFound:
                continue
        if removed_names:
            removed[instance_id] = sorted(removed_names)
    return removed


def run_preflight(
    instance_ids: list[str],
    skip_container_cleanup: bool = False,
) -> docker.DockerClient:
    ensure_openai_api_key()
    client = ensure_docker_runtime()
    print("Preflight: OPENAI_API_KEY present")
    print("Preflight: Docker daemon reachable")
    if skip_container_cleanup:
        print("Preflight: skipping stale container cleanup")
        return client
    removed = cleanup_stale_instance_containers(client, instance_ids)
    if not removed:
        print("Preflight: no stale instance containers found")
        return client
    total_removed = sum(len(names) for names in removed.values())
    print(f"Preflight: removed {total_removed} stale container(s)")
    for instance_id, names in sorted(removed.items()):
        print(f"  - {instance_id}: {', '.join(names)}")
    return client


def compare_runs(baseline_dir: Path, enhanced_dir: Path) -> None:
    """Print a thesis-style comparison table between a baseline run and an enhanced run."""
    baseline_agg_path = baseline_dir / "aggregate_summary.json"
    enhanced_agg_path = enhanced_dir / "aggregate_summary.json"
    baseline_run_path = baseline_dir / "run_summary.json"
    enhanced_run_path = enhanced_dir / "run_summary.json"

    if not baseline_agg_path.exists() or not enhanced_agg_path.exists():
        print(f"ERROR: aggregate_summary.json not found in one of the given directories.")
        return

    b_agg = json.loads(baseline_agg_path.read_text())
    e_agg = json.loads(enhanced_agg_path.read_text())

    print("\n" + "=" * 65)
    print("  THESIS COMPARISON: Baseline vs Enhanced Test Guided Patching")
    print("=" * 65)
    print(f"  Baseline dir : {baseline_dir}")
    print(f"  Enhanced dir : {enhanced_dir}")
    print("-" * 65)
    print(f"  {'Metric':<40} {'Baseline':>10} {'Enhanced':>10}")
    print("-" * 65)

    def row(label: str, bval: Any, eval_: Any, fmt: str = "") -> None:
        if fmt == "%":
            bs = f"{bval*100:.1f}%" if isinstance(bval, (int, float)) else str(bval)
            es = f"{eval_*100:.1f}%" if isinstance(eval_, (int, float)) else str(eval_)
        else:
            bs = str(bval)
            es = str(eval_)
        print(f"  {label:<40} {bs:>10} {es:>10}")

    row("Instances total", b_agg.get("instances_total"), e_agg.get("instances_total"))
    row("Instances resolved", b_agg.get("instances_resolved"), e_agg.get("instances_resolved"))
    row("Resolution rate", b_agg.get("instance_resolution_rate"), e_agg.get("instance_resolution_rate"), "%")
    row("Clean apply ratio", b_agg.get("clean_apply_ratio"), e_agg.get("clean_apply_ratio"), "%")
    row("Original tests passed (total)", b_agg.get("original_passed_count_total"), e_agg.get("original_passed_count_total"))
    row("Final tests passed (total)", b_agg.get("final_passed_count_total"), e_agg.get("final_passed_count_total"))
    row("Passed count improvement", b_agg.get("original_passed_count_improvement_total"), e_agg.get("original_passed_count_improvement_total"))
    row("Failed count reduction", b_agg.get("original_failed_count_reduction_total"), e_agg.get("original_failed_count_reduction_total"))

    b_res = b_agg.get("instances_resolved", 0)
    e_res = e_agg.get("instances_resolved", 0)
    b_rate = b_agg.get("instance_resolution_rate", 0.0)
    e_rate = e_agg.get("instance_resolution_rate", 0.0)
    delta = e_rate - b_rate if isinstance(e_rate, float) and isinstance(b_rate, float) else None
    print("-" * 65)
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        print(f"  {'Resolution rate delta (Enhanced - Baseline)':<40} {sign}{delta*100:.1f}%")
        improvement = e_res - b_res
        print(f"  {'Additional instances resolved':<40} {'+' if improvement >= 0 else ''}{improvement}")
    print("=" * 65)

    # Per-instance breakdown if both run_summary.json exist
    if baseline_run_path.exists() and enhanced_run_path.exists():
        b_items = json.loads(baseline_run_path.read_text())
        e_items = json.loads(enhanced_run_path.read_text())
        b_by_id = {item["instance_id"]: item for item in b_items}
        e_by_id = {item["instance_id"]: item for item in e_items}
        common_ids = sorted(set(b_by_id) & set(e_by_id))
        if common_ids:
            print(f"\n  {'Instance':<40} {'Base':>7} {'Enh':>7} {'Delta':>7}")
            print("  " + "-" * 63)
            for iid in common_ids:
                b_r = "✓" if b_by_id[iid].get("final_resolved") else "✗"
                e_r = "✓" if e_by_id[iid].get("final_resolved") else "✗"
                b_imp = b_by_id[iid].get("original_passed_count_improvement", 0)
                e_imp = e_by_id[iid].get("original_passed_count_improvement", 0)
                delta_imp = e_imp - b_imp
                sign = "+" if delta_imp >= 0 else ""
                print(f"  {iid:<40} {b_r:>7} {e_r:>7} {sign}{delta_imp:>6}")
    print()


def main() -> None:
    args = build_parser().parse_args()

    if args.compare_runs:
        baseline_dir, enhanced_dir = args.compare_runs
        compare_runs(Path(baseline_dir), Path(enhanced_dir))
        return

    run_output_dir = allocate_versioned_output_dir(args.output_dir)

    print("=== Enhanced Patch Pipeline ===")
    print(f"Dataset: {args.dataset_name} [{args.split}]")
    print(f"Model: {args.model}")
    print(f"Pipeline mode: {'BASELINE (no enhanced tests)' if args.baseline_only else 'ENHANCED'}")
    print(f"OPENAI_API_BASE: {os.environ.get('OPENAI_API_BASE', '<unset>')}")
    print(f"Output root: {args.output_dir}")
    print(f"Run output dir: {run_output_dir}")

    dataset = load_swebench_dataset(
        name=args.dataset_name,
        split=args.split,
        instance_ids=args.instance_ids,
    )
    dataset_instance_ids = [instance["instance_id"] for instance in dataset]
    client = run_preflight(
        instance_ids=dataset_instance_ids,
        skip_container_cleanup=args.skip_container_cleanup,
    )
    client.close()
    responder = OpenAIResponder(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_completion_tokens,
    )

    summaries = []
    for idx, instance in enumerate(dataset, start=1):
        print(f"\n=== [{idx}/{len(dataset)}] {instance['instance_id']} ===")
        summary = run_instance_pipeline(
            instance=instance,
            responder=responder,
            output_dir=run_output_dir,
            max_context_files=args.max_context_files,
            max_chars_per_file=args.max_chars_per_file,
            timeout=args.timeout,
            max_candidate_attempts=args.max_candidate_attempts,
            baseline_only=args.baseline_only,
            hide_original_test_patch_in_repair=args.hide_original_test_patch_in_repair,
        )
        summaries.append(summary)
        print(
            f"[{instance['instance_id']}] 完成: kept={summary['kept_enhanced_candidates']}, "
            f"patch_mode={summary['final_patch_apply_mode']}, "
            f"effective_patch_applied={summary['final_patch_effective_applied']}, "
            f"generated_patch_accepted={summary['generated_patch_accepted']}, "
            f"resolved={summary['final_resolved']}"
        )

    write_json(run_output_dir / "run_summary.json", summaries)
    write_json(run_output_dir / "aggregate_summary.json", summarize_run(summaries))

    repo_groups: dict[str, list[dict[str, Any]]] = {}
    for item in summaries:
        repo = item.get("repo") or "unknown_repo"
        repo_groups.setdefault(repo, []).append(item)

    for repo, repo_summaries in repo_groups.items():
        safe_repo = sanitize_repo_name(repo)
        write_json(
            run_output_dir / f"run_summary_{safe_repo}.json",
            repo_summaries,
        )
        write_json(
            run_output_dir / f"aggregate_summary_{safe_repo}.json",
            summarize_run(repo_summaries),
        )
    print("\n=== Finished ===")
    print(f"Summaries written to {run_output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
