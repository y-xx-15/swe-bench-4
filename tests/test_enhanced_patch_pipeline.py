from swebench.experiments import enhanced_patch_pipeline as epp
from swebench.experiments.enhanced_patch_pipeline import (
    aggregate_enhanced_failures,
    allocate_versioned_output_dir,
    apply_fragment_edit_plan,
    build_anchor_context,
    build_class_anchor_region,
    build_edit_anchor_regions,
    build_fragment_edit_prompt,
    build_patch_analysis_prompt,
    build_patch_analysis_feedback,
    build_patch_feedback,
    get_patch_acceptance_reason,
    build_patch_strategy_feedback,
    build_patch_strategy_prompt,
    extract_multi_symbol_repair_obligations,
    extract_exception_type_repair_obligations,
    extract_failure_path_repair_obligations,
    extract_sibling_validation_repair_obligations,
    build_unified_diff_from_replacement,
    build_context_file_list,
    build_enhanced_test_prompt,
    build_idea_feedback,
    build_test_idea_prompt,
    build_patch_prompt,
    classify_failure_mode,
    count_failed,
    count_passed,
    count_statuses,
    compute_log_noise_ratio,
    extract_test_identifiers_from_patch,
    extract_failure_focus,
    extract_failure_path_signatures,
    generate_patch_with_strategy,
    get_required_source_targets,
    get_strategy_edit_target_files,
    find_failing_identifiers,
    find_duplicate_test_identifiers,
    get_enhanced_test_identifiers,
    get_hard_failure_signal,
    has_meaningful_status_map,
    get_repair_mode,
    get_original_test_identifiers,
    has_hard_failure_signal,
    normalize_patch,
    patch_improves_metrics,
    parse_json_object,
    sanitize_unified_diff,
    safe_rate,
    score_candidate_quality,
    score_patch_candidate_eval,
    select_candidates_with_path_coverage,
    should_prioritize_edit_plan_patch,
    extract_strategy_constraints,
    select_semantic_buckets,
    select_template_names,
    summarize_run,
    synthesize_structural_patch_from_strategy,
    recommend_generation_budget,
    is_soft_patch_syntax_warning,
    normalize_patch_feedback,
    needs_minimal_structural_fix_guidance,
    build_path_semantic_alignment_feedback,
    validate_fragment_edit_plan,
    validate_patch_landing,
    extract_precise_failure_feedback,
    run_semantic_oracle_checks,
    summarize_failure_log,
    is_minimal_structural_control_flow_patch,
    is_negative_gating_patch,
    has_structural_elif_to_if_change,
    score_patch_candidate_choice,
    strategy_requires_structural_branch_change,
)
from swebench.harness.grading import extract_test_output_blocks


def test_normalize_patch():
    assert normalize_patch("diff --git a/a b/a") == "diff --git a/a b/a\n"
    assert normalize_patch("   ") == ""


def test_allocate_versioned_output_dir_starts_at_one(tmp_path):
    run_dir = allocate_versioned_output_dir(tmp_path / "runs")
    assert run_dir == tmp_path / "runs" / "1"
    assert run_dir.is_dir()


def test_allocate_versioned_output_dir_increments_existing_versions(tmp_path):
    runs_dir = tmp_path / "runs"
    (runs_dir / "1").mkdir(parents=True)
    (runs_dir / "7").mkdir()
    (runs_dir / "notes").mkdir()
    run_dir = allocate_versioned_output_dir(runs_dir)
    assert run_dir == runs_dir / "8"
    assert run_dir.is_dir()


def test_extract_test_identifiers_from_patch():
    patch = """diff --git a/tests/test_demo.py b/tests/test_demo.py
--- a/tests/test_demo.py
+++ b/tests/test_demo.py
@@ -1,1 +1,7 @@
+def test_sweb_enhanced_case_one():
+    assert True
+
+class TestSuite:
+    def test_sweb_enhanced_case_two(self):
+        assert True
"""
    assert extract_test_identifiers_from_patch(patch) == [
        "test_sweb_enhanced_case_one",
        "test_sweb_enhanced_case_two",
    ]


def test_extract_test_identifiers_ignores_context_lines():
    patch = """diff --git a/tests/test_demo.py b/tests/test_demo.py
--- a/tests/test_demo.py
+++ b/tests/test_demo.py
@@ -1,3 +1,6 @@
 def test_existing_case():
     assert True
+
+def test_sweb_enhanced_case_new():
+    assert True
"""
    assert extract_test_identifiers_from_patch(patch) == [
        "test_sweb_enhanced_case_new",
    ]


def test_sanitize_unified_diff_repairs_invalid_patch_when_possible():
    patch, error = sanitize_unified_diff(
        """diff --git a/tests/test_demo.py b/tests/test_demo.py
--- a/tests/test_demo.py
+++ b/tests/test_demo.py
@@ -1,1 +1,99 @@
+def test_bad():
+    assert True
"""
    )
    assert patch
    assert "@@ -1,0 +1,2 @@" in patch
    assert error is None


def test_find_failing_identifiers():
    status_map = {
        "tests/test_demo.py::test_sweb_enhanced_case_one": "FAILED",
        "tests/test_demo.py::test_other_case": "PASSED",
    }
    assert find_failing_identifiers(
        status_map,
        ["test_sweb_enhanced_case_one", "test_missing"],
    ) == ["test_sweb_enhanced_case_one"]


def test_summary_helpers():
    status_map = {
        "a": "PASSED",
        "b": "FAILED",
        "c": "ERROR",
        "d": "XFAIL",
    }
    assert count_statuses(status_map) == {
        "PASSED": 1,
        "FAILED": 1,
        "ERROR": 1,
        "XFAIL": 1,
    }
    assert count_passed(status_map) == 2
    assert count_failed(status_map) == 2
    assert has_meaningful_status_map(status_map) is True
    assert has_meaningful_status_map({}) is False
    assert safe_rate(1, 4) == 0.25
    assert safe_rate(1, 0) == 0.0
    assert get_repair_mode([]) == "baseline_fallback"
    assert get_repair_mode([object()]) == "enhanced_guided"


def test_is_environment_blocked_import_failure_matches_same_missing_module():
    failure_focus = {
        "failure_mode": "import_error",
        "dominant_errors": ["ModuleNotFoundError", "ImportError"],
        "target_test_tracebacks": {
            "test_demo": (
                "TARGET TEST: test_demo\n"
                "WEAK TRACEBACK CONTEXT:\n"
                "ModuleNotFoundError: No module named 'roman'\n"
                "ImportError: Error importing plugin \"sphinx.testing.fixtures\": No module named 'roman'\n"
            )
        },
    }
    eval_result = epp.EvalResult(
        resolved=False,
        status_map={},
        log_text="ImportError: Error importing plugin \"sphinx.testing.fixtures\": No module named 'roman'",
        log_path="",
        report=None,
        patch_applied=True,
        patch_apply_mode="clean_apply",
        timed_out=False,
        error=None,
    )

    assert epp.is_environment_blocked_import_failure(failure_focus, eval_result) is True


def test_failure_mode_and_templates():
    log = "AssertionError\nassert not hasattr(s, '__dict__')"
    assert classify_failure_mode(log) == "attribute_error"
    assert select_template_names("attribute_error") == [
        "attribute_absence_check",
        "forbidden_attribute_assignment",
        "slots_visibility_check",
    ]
    assert select_semantic_buckets("attribute_error") == [
        "direct_symptom",
        "behavioral_consequence",
        "structural_invariant",
    ]


def test_failure_focus_and_adaptive_budget_helpers():
    instance = {
        "test_patch": """diff --git a/tests/test_demo.py b/tests/test_demo.py
--- a/tests/test_demo.py
+++ b/tests/test_demo.py
@@ -1,1 +1,2 @@
+def test_demo_case():
+    assert True
""",
    }
    status_map = {
        "tests/test_demo.py::test_demo_case": "FAILED",
        "tests/test_demo.py::test_other": "ERROR",
    }
    log = """FAILED tests/test_demo.py::test_demo_case - AssertionError: boom
ERROR tests/test_demo.py::test_other - ValueError: bad
DeprecationWarning: old path
Traceback (most recent call last):"""
    failure_focus = extract_failure_focus(instance, status_map, log)
    assert failure_focus["failure_mode"] in {"value_error", "assertion_error", "generic_failure"}
    assert failure_focus["failing_tests_sample"]
    assert compute_log_noise_ratio(log) > 0.0
    budget = recommend_generation_budget(status_map, log, base_num_candidates=3, base_attempts=3)
    assert budget["candidate_budget"] >= 5
    assert budget["keep_top_k"] >= 3
    assert budget["attempt_budget"] >= 3


def test_extract_per_test_tracebacks_filters_passed_noise_blocks():
    log = """PASSED test_requests.py::RequestsTestCase::test_HTTP_302_ALLOW_REDIRECT_GET
PASSED test_requests.py::RequestsTestCase::test_POSTBIN_GET_POST_FILES
_______________________ TestTimeout.test_encoded_methods _______________________

self = <test_requests.TestTimeout object at 0x1>

    def test_encoded_methods(self):
>       assert r.ok
E       AssertionError: bad response

test_requests.py:1395: AssertionError
"""
    result = epp.extract_per_test_tracebacks(
        log,
        ["test_HTTP_302_ALLOW_REDIRECT_GET", "test_encoded_methods"],
    )
    assert "test_HTTP_302_ALLOW_REDIRECT_GET" not in result
    assert "test_encoded_methods" in result
    assert "AssertionError" in result["test_encoded_methods"]


def test_extract_failure_focus_backfills_weak_target_tracebacks_when_missing():
    instance = {
        "FAIL_TO_PASS": '["test_HTTP_302_ALLOW_REDIRECT_GET", "test_encoded_methods"]',
        "test_patch": "",
    }
    status_map = {
        "test_requests.py::RequestsTestCase::test_HTTP_302_ALLOW_REDIRECT_GET": "PASSED",
        "test_requests.py::TestTimeout::test_encoded_methods": "FAILED",
    }
    log = """PASSED test_requests.py::RequestsTestCase::test_HTTP_302_ALLOW_REDIRECT_GET
TypeError: getresponse() got an unexpected keyword argument 'buffering'
_______________________ TestTimeout.test_encoded_methods _______________________
E       AssertionError: bad response
"""
    failure_focus = extract_failure_focus(instance, status_map, log)
    assert failure_focus["active_fail_to_pass_identifiers"] == ["test_encoded_methods"]
    assert failure_focus["inactive_fail_to_pass_identifiers"] == ["test_HTTP_302_ALLOW_REDIRECT_GET"]
    assert "test_encoded_methods" in failure_focus["target_test_tracebacks"]
    assert "test_HTTP_302_ALLOW_REDIRECT_GET" in failure_focus["target_test_tracebacks"]
    assert "TARGET TEST: test_HTTP_302_ALLOW_REDIRECT_GET" in failure_focus["target_test_tracebacks"]["test_HTTP_302_ALLOW_REDIRECT_GET"]
    assert "TypeError" in failure_focus["target_test_tracebacks"]["test_HTTP_302_ALLOW_REDIRECT_GET"]


def test_score_candidate_quality_prefers_reproducing_candidates():
    candidate = type(
        "Candidate",
        (),
        {
            "enhanced_identifiers": ["test_sweb_enhanced_case"],
            "failing_identifiers": ["test_sweb_enhanced_case"],
            "idea": {"semantic_bucket": "direct_symptom"},
            "patch": "diff --git a/tests/test_demo.py b/tests/test_demo.py\n+def test_sweb_enhanced_case():\n+    assert True\n",
        },
    )()
    eval_result = type("Eval", (), {"status_map": {"tests/test_demo.py::test_sweb_enhanced_case": "FAILED"}})()
    score, breakdown = score_candidate_quality(candidate, eval_result)
    assert score > 0.7
    assert breakdown["reproduction"] == 1.0


def test_build_idea_feedback_requires_covers_original_tests():
    feedback = build_idea_feedback(
        {
            "title": "cover dotted blueprint name",
            "semantic_bucket": "direct_symptom",
            "goal": "exercise the buggy path",
            "oracle": "regression_should_fail_on_buggy_and_pass_on_fixed",
            "target_tests": ["test_sweb_enhanced_blueprint_name_with_dot"],
            "template": "single_exception_assert",
            "target_source_symbol": "Blueprint.__init__",
            "target_validation_subject": "name",
            "semantic_alignment_tokens": ["Blueprint(", "ValueError"],
            "trigger_shape_tokens": ["Blueprint(", "name"],
            "rationale": "Check the intended fixed behavior.",
        },
        seen_titles=set(),
        seen_buckets=set(),
        allowed_buckets=["direct_symptom"],
        original_test_identifiers=[
            "test_dotted_name_not_allowed",
            "test_route_decorator_custom_endpoint_with_dots",
        ],
    )
    assert feedback is not None
    assert "covers_original_tests" in feedback


def test_build_idea_feedback_rejects_unknown_original_path_names():
    feedback = build_idea_feedback(
        {
            "title": "cover dotted endpoint",
            "semantic_bucket": "direct_symptom",
            "goal": "exercise endpoint validation",
            "oracle": "regression_should_fail_on_buggy_and_pass_on_fixed",
            "target_tests": ["test_sweb_enhanced_endpoint_with_dot"],
            "covers_original_tests": ["test_unknown_original_case"],
            "template": "single_exception_assert",
            "target_source_symbol": "Blueprint.add_url_rule",
            "target_validation_subject": "endpoint",
            "semantic_alignment_tokens": ["endpoint=", "add_url_rule", "ValueError"],
            "trigger_shape_tokens": ["endpoint=", "add_url_rule"],
            "rationale": "Check the intended fixed behavior.",
        },
        seen_titles=set(),
        seen_buckets=set(),
        allowed_buckets=["direct_symptom"],
        original_test_identifiers=[
            "test_dotted_name_not_allowed",
            "test_route_decorator_custom_endpoint_with_dots",
        ],
    )
    assert feedback is not None
    assert "may only contain original FAIL_TO_PASS identifiers" in feedback


def test_build_idea_feedback_requires_hitting_an_uncovered_original_path():
    feedback = build_idea_feedback(
        {
            "title": "cover only the already-covered name path",
            "semantic_bucket": "direct_symptom",
            "goal": "exercise blueprint name validation",
            "oracle": "regression_should_fail_on_buggy_and_pass_on_fixed",
            "target_tests": ["test_sweb_enhanced_blueprint_name_with_dot"],
            "covers_original_tests": ["test_dotted_name_not_allowed"],
            "template": "single_exception_assert",
            "target_source_symbol": "Blueprint.__init__",
            "target_validation_subject": "name",
            "semantic_alignment_tokens": ["Blueprint(", "ValueError"],
            "trigger_shape_tokens": ["Blueprint(", "name"],
            "rationale": "Check the intended fixed behavior.",
        },
        seen_titles=set(),
        seen_buckets=set(),
        allowed_buckets=["direct_symptom"],
        original_test_identifiers=[
            "test_dotted_name_not_allowed",
            "test_route_decorator_custom_endpoint_with_dots",
        ],
        required_uncovered_tests=["test_route_decorator_custom_endpoint_with_dots"],
    )
    assert feedback is not None
    assert "still-uncovered original FAIL_TO_PASS path" in feedback


def test_extract_failure_path_signatures_captures_symbol_exception_and_tokens():
    failure_focus = {
        "target_test_tracebacks": {
            "test_route_decorator_custom_endpoint_with_dots": """________________ test_route_decorator_custom_endpoint_with_dots ________________
with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
self = <Blueprint 'bp'>
src/flask/blueprints.py:364: AssertionError
"""
        }
    }
    code_context = {
        "src/flask/blueprints.py": """class Blueprint:
    def add_url_rule(self, rule, endpoint=None, view_func=None):
        pass
""",
        "src/flask/scaffold.py": """class Scaffold:
    def add_url_rule(self, rule, endpoint=None, view_func=None):
        pass
""",
    }
    signatures = extract_failure_path_signatures(failure_focus, code_context)
    signature = signatures["test_route_decorator_custom_endpoint_with_dots"]
    assert signature["source_file"] == "src/flask/blueprints.py"
    assert signature["source_symbol"] == "Blueprint.add_url_rule"
    assert signature["expected_exception"] == "ValueError"
    assert signature["observed_exception"] == "AssertionError"
    assert "endpoint=" in signature["alignment_tokens"]
    assert "route(" in signature["alignment_tokens"]
    assert "add_url_rule" in signature["alignment_tokens"]


def test_extract_failure_path_signatures_prefers_traceback_lineno_and_owner_hint_for_same_named_methods():
    failure_focus = {
        "target_test_tracebacks": {
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
self = <Blueprint 'bp'>
src/flask/blueprints.py:364: AssertionError"""
        }
    }
    code_context = {
        "src/flask/blueprints.py": """class BlueprintSetupState:
    def add_url_rule(self, rule, endpoint=None, view_func=None):
        self.app.add_url_rule(rule, endpoint, view_func)


class Blueprint:
    def add_url_rule(self, rule, endpoint=None, view_func=None):
        if endpoint:
            assert "." not in endpoint
        if view_func and hasattr(view_func, "__name__"):
            assert "." not in view_func.__name__
""",
    }
    signatures = extract_failure_path_signatures(failure_focus, code_context)
    assert (
        signatures["test_route_decorator_custom_endpoint_with_dots"]["source_symbol"]
        == "Blueprint.add_url_rule"
    )


def test_build_idea_feedback_rejects_semantically_mislabeled_path_coverage():
    failure_focus = {
        "target_test_tracebacks": {
            "test_dotted_name_not_allowed": """with pytest.raises(ValueError):
>   flask.Blueprint("app.ui", __name__)
tests/test_blueprints.py:256: Failed""",
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError""",
        }
    }
    feedback = build_idea_feedback(
        {
            "title": "wrongly relabeled constructor path",
            "semantic_bucket": "direct_symptom",
            "goal": "exercise blueprint name validation",
            "oracle": "regression_should_fail_on_buggy_and_pass_on_fixed",
            "target_tests": ["test_sweb_enhanced_wrong_label"],
            "covers_original_tests": ["test_route_decorator_custom_endpoint_with_dots"],
            "target_source_symbol": "Blueprint.__init__",
            "target_validation_subject": "name",
            "semantic_alignment_tokens": ["Blueprint(", "ValueError"],
            "trigger_shape_tokens": ["Blueprint(", "name"],
            "template": "single_exception_assert",
            "rationale": "Check the intended fixed behavior.",
        },
        seen_titles=set(),
        seen_buckets=set(),
        allowed_buckets=["direct_symptom"],
        original_test_identifiers=[
            "test_dotted_name_not_allowed",
            "test_route_decorator_custom_endpoint_with_dots",
        ],
        failure_focus=failure_focus,
    )
    assert feedback is not None
    assert "not semantically aligned" in feedback


def test_build_idea_feedback_rejects_obligation_with_wrong_trigger_shape():
    failure_focus = {
        "target_test_tracebacks": {
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError""",
        }
    }
    code_context = {
        "src/flask/blueprints.py": """class Blueprint:
    def add_url_rule(self, rule, endpoint=None, view_func=None):
        if endpoint:
            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
        if view_func and hasattr(view_func, "__name__"):
            assert "." not in view_func.__name__, "Blueprint view function name should not contain dots"
""",
    }
    feedback = build_idea_feedback(
        {
            "title": "wrong trigger shape for function-name obligation",
            "semantic_bucket": "direct_symptom",
            "goal": "exercise blueprint endpoint validation",
            "oracle": "regression_should_fail_on_buggy_and_pass_on_fixed",
            "target_tests": ["test_sweb_enhanced_wrong_trigger_shape"],
            "covers_original_tests": ["test_route_decorator_custom_endpoint_with_dots"],
            "covers_obligations": ["test_route_decorator_custom_endpoint_with_dots::view_func_name"],
            "target_source_symbol": "Blueprint.add_url_rule",
            "target_validation_subject": "view_func_name",
            "semantic_alignment_tokens": ["endpoint=", "add_url_rule", "ValueError"],
            "trigger_shape_tokens": ["endpoint=", "add_url_rule"],
            "template": "single_exception_assert",
            "rationale": "Check the intended fixed behavior.",
        },
        seen_titles=set(),
        seen_buckets=set(),
        allowed_buckets=["direct_symptom"],
        original_test_identifiers=["test_route_decorator_custom_endpoint_with_dots"],
        failure_focus=failure_focus,
        code_context=code_context,
    )
    assert feedback is not None
    assert "trigger-shape aligned" in feedback


def test_build_path_semantic_alignment_feedback_accepts_matching_endpoint_route_path():
    failure_focus = {
        "target_test_tracebacks": {
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError"""
        }
    }
    feedback = build_path_semantic_alignment_feedback(
        ["test_route_decorator_custom_endpoint_with_dots"],
        "Blueprint.add_url_rule",
        ["endpoint=", "route(", "add_url_rule", "ValueError"],
        failure_focus,
        "bp.route('/', endpoint='a.b')\nself.add_url_rule(rule, endpoint, f, **options)",
    )
    assert feedback is None


def test_extract_multi_symbol_repair_obligations_tracks_distinct_symbols_from_retained_candidates():
    failure_focus = {
        "target_test_tracebacks": {
            "test_dotted_name_not_allowed": """with pytest.raises(ValueError):
>   flask.Blueprint("app.ui", __name__)
tests/test_blueprints.py:256: Failed""",
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError""",
        }
    }
    candidates = [
        epp.CandidatePatch(
            idea={"target_source_symbol": "Blueprint.__init__"},
            patch="",
            raw_response="",
            identifiers=[],
            covered_original_tests=["test_dotted_name_not_allowed"],
        ),
        epp.CandidatePatch(
            idea={"target_source_symbol": "add_url_rule"},
            patch="",
            raw_response="",
            identifiers=[],
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        ),
    ]
    obligations = extract_multi_symbol_repair_obligations(failure_focus, candidates)
    assert obligations["active"] is True
    assert obligations["symbols"] == ["Blueprint.__init__", "add_url_rule"]


def test_extract_exception_type_repair_obligations_tracks_expected_vs_observed_exception_gap():
    failure_focus = {
        "target_test_tracebacks": {
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError""",
        }
    }
    candidates = [
        epp.CandidatePatch(
            idea={"target_source_symbol": "add_url_rule"},
            patch="",
            raw_response="",
            identifiers=[],
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        ),
    ]
    obligations = extract_exception_type_repair_obligations(failure_focus, candidates)
    assert obligations["active"] is True
    assert obligations["obligations"] == [
        {
            "test_id": "test_route_decorator_custom_endpoint_with_dots",
            "source_symbol": "add_url_rule",
            "expected_exception": "ValueError",
            "observed_exception": "AssertionError",
        }
    ]


def test_extract_sibling_validation_repair_obligations_tracks_multiple_assert_branches_in_same_symbol():
    failure_focus = {
        "target_test_tracebacks": {
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError""",
        }
    }
    candidates = [
        epp.CandidatePatch(
            idea={"target_source_symbol": "add_url_rule"},
            patch="",
            raw_response="",
            identifiers=[],
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        ),
    ]
    code_context = {
        "src/flask/blueprints.py": """class Blueprint:
    def add_url_rule(self, rule, endpoint=None, view_func=None):
        if endpoint:
            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
        if view_func and hasattr(view_func, "__name__"):
            assert "." not in view_func.__name__, "Blueprint view function name should not contain dots"
""",
    }
    obligations = extract_sibling_validation_repair_obligations(
        failure_focus=failure_focus,
        filtered_candidates=candidates,
        code_context=code_context,
    )
    assert obligations["active"] is True
    assert obligations["obligations"][0]["source_symbol"] == "Blueprint.add_url_rule"
    assert obligations["obligations"][0]["assert_count"] == 2
    marker_text = " ".join("/".join(group) for group in obligations["obligations"][0]["marker_groups"])
    assert "endpoint" in marker_text
    assert "view_func.__name__" in marker_text or "view_func" in marker_text


def test_extract_failure_path_repair_obligations_splits_sibling_assertions_into_distinct_obligations():
    failure_focus = {
        "target_test_tracebacks": {
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError""",
        }
    }
    code_context = {
        "src/flask/blueprints.py": """class Blueprint:
    def add_url_rule(self, rule, endpoint=None, view_func=None):
        if endpoint:
            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
        if view_func and hasattr(view_func, "__name__"):
            assert "." not in view_func.__name__, "Blueprint view function name should not contain dots"
""",
    }
    obligations = extract_failure_path_repair_obligations(failure_focus, code_context)
    obligation_ids = [item["id"] for item in obligations["test_route_decorator_custom_endpoint_with_dots"]]
    assert len(obligation_ids) == 2
    assert obligations["test_route_decorator_custom_endpoint_with_dots"][0]["source_symbol"] == "Blueprint.add_url_rule"
    obligation_text = " ".join(obligation_ids)
    assert "endpoint" in obligation_text
    assert "view_func" in obligation_text
    obligation_by_id = {
        item["id"]: item for item in obligations["test_route_decorator_custom_endpoint_with_dots"]
    }
    endpoint_obligation = next(item for key, item in obligation_by_id.items() if "endpoint" in key)
    view_name_obligation = next(item for key, item in obligation_by_id.items() if "view_func" in key)
    assert endpoint_obligation["validation_subject"] == "endpoint"
    assert endpoint_obligation["trigger_shape"] == "argument_value_contains_forbidden_token"
    assert "endpoint=" in endpoint_obligation["trigger_shape_tokens"]
    assert view_name_obligation["validation_subject"] in {"view_func_name", "view_func_name__name__", "view_func___name__"}
    assert view_name_obligation["trigger_shape"] == "object_attribute_value_contains_forbidden_token"
    assert "__name__" in " ".join(view_name_obligation["trigger_shape_tokens"])
    assert endpoint_obligation["obligation_level"] == "primary_direct"
    assert view_name_obligation["obligation_level"] == "primary_sibling"


def test_filter_primary_required_repair_obligations_keeps_direct_and_sibling_only():
    filtered = epp.filter_primary_required_repair_obligations(
        [
            {"id": "a", "obligation_level": "primary_direct"},
            {"id": "b", "obligation_level": "primary_sibling"},
            {"id": "c", "obligation_level": "propagated"},
        ]
    )
    assert [item["id"] for item in filtered] == ["a", "b"]


def test_select_effective_required_repair_obligations_keeps_strong_primary_siblings_for_flask_like_case():
    obligations = [
        {
            "id": "endpoint",
            "source_symbol": "Blueprint.add_url_rule",
            "validation_subject": "endpoint",
            "obligation_level": "primary_direct",
            "evidence_strength": "strong_evidence",
        },
        {
            "id": "view_func_name",
            "source_symbol": "Blueprint.add_url_rule",
            "validation_subject": "view_func_name",
            "obligation_level": "primary_sibling",
            "evidence_strength": "strong_evidence",
        },
        {
            "id": "name",
            "source_symbol": "Blueprint.__init__",
            "validation_subject": "name",
            "obligation_level": "primary_direct",
            "evidence_strength": "strong_evidence",
        },
    ]
    filtered = epp.select_effective_required_repair_obligations(obligations, filtered_candidates=[])
    assert [item["id"] for item in filtered] == ["endpoint", "view_func_name", "name"]


def test_select_effective_required_repair_obligations_does_not_drop_subjects_within_same_strong_primary_symbol():
    obligations = [
        {
            "id": "endpoint",
            "source_symbol": "Blueprint.add_url_rule",
            "validation_subject": "endpoint",
            "obligation_level": "primary_direct",
            "evidence_strength": "strong_evidence",
        },
        {
            "id": "view_func_name",
            "source_symbol": "Blueprint.add_url_rule",
            "validation_subject": "view_func_name",
            "obligation_level": "primary_sibling",
            "evidence_strength": "strong_evidence",
        },
        {
            "id": "redirect_like_weak",
            "source_symbol": "SomeOtherSymbol",
            "validation_subject": "core",
            "obligation_level": "propagated",
            "evidence_strength": "weak_evidence",
        },
    ]
    filtered = epp.select_effective_required_repair_obligations(obligations, filtered_candidates=[])
    assert [item["id"] for item in filtered] == ["endpoint", "view_func_name"]


def test_select_effective_required_repair_obligations_prefers_top_candidate_symbol_when_all_evidence_is_weak():
    obligations = [
        {
            "id": "request_core",
            "source_symbol": "Session.request",
            "validation_subject": "core",
            "obligation_level": "primary_direct",
            "evidence_strength": "weak_evidence",
        },
        {
            "id": "redirect_core",
            "source_symbol": "SessionRedirectMixin.resolve_redirects",
            "validation_subject": "core",
            "obligation_level": "primary_direct",
            "evidence_strength": "weak_evidence",
        },
    ]
    candidates = [
        epp.CandidatePatch(
            idea={"target_source_symbol": "Session.request"},
            patch="",
            raw_response="",
            identifiers=[],
            quality_score=0.9,
        ),
        epp.CandidatePatch(
            idea={"target_source_symbol": "SessionRedirectMixin.resolve_redirects"},
            patch="",
            raw_response="",
            identifiers=[],
            quality_score=0.4,
        ),
    ]
    filtered = epp.select_effective_required_repair_obligations(obligations, filtered_candidates=candidates)
    assert [item["id"] for item in filtered] == ["request_core"]


def test_select_effective_required_repair_obligations_keeps_all_primary_subjects_within_selected_dominant_symbol():
    obligations = [
        {
            "id": "request_method",
            "source_symbol": "Session.request",
            "validation_subject": "method",
            "obligation_level": "primary_direct",
            "evidence_strength": "weak_evidence",
        },
        {
            "id": "request_redirect_method",
            "source_symbol": "Session.request",
            "validation_subject": "redirect_method",
            "obligation_level": "primary_sibling",
            "evidence_strength": "weak_evidence",
        },
        {
            "id": "redirect_core",
            "source_symbol": "SessionRedirectMixin.resolve_redirects",
            "validation_subject": "core",
            "obligation_level": "primary_direct",
            "evidence_strength": "weak_evidence",
        },
    ]
    candidates = [
        epp.CandidatePatch(
            idea={"target_source_symbol": "Session.request"},
            patch="",
            raw_response="",
            identifiers=[],
            quality_score=0.9,
        ),
        epp.CandidatePatch(
            idea={"target_source_symbol": "SessionRedirectMixin.resolve_redirects"},
            patch="",
            raw_response="",
            identifiers=[],
            quality_score=0.4,
        ),
    ]
    filtered = epp.select_effective_required_repair_obligations(obligations, filtered_candidates=candidates)
    assert [item["id"] for item in filtered] == ["request_method", "request_redirect_method"]


def test_select_effective_required_repair_obligations_buckets_weak_evidence_by_original_test():
    obligations = [
        {
            "id": "name_core",
            "source_symbol": "Blueprint.__init__",
            "validation_subject": "name",
            "covered_original_tests": ["test_dotted_name_not_allowed"],
            "obligation_level": "primary_direct",
            "evidence_strength": "weak_evidence",
        },
        {
            "id": "endpoint_subject",
            "source_symbol": "Blueprint.add_url_rule",
            "validation_subject": "endpoint",
            "covered_original_tests": ["test_route_decorator_custom_endpoint_with_dots"],
            "obligation_level": "primary_direct",
            "evidence_strength": "weak_evidence",
        },
        {
            "id": "view_func_subject",
            "source_symbol": "Blueprint.add_url_rule",
            "validation_subject": "view_func_name",
            "covered_original_tests": ["test_route_decorator_custom_endpoint_with_dots"],
            "obligation_level": "primary_sibling",
            "evidence_strength": "weak_evidence",
        },
        {
            "id": "misleading_alt",
            "source_symbol": "BlueprintSetupState.add_url_rule",
            "validation_subject": "endpoint",
            "covered_original_tests": ["test_route_decorator_custom_endpoint_with_dots"],
            "obligation_level": "primary_direct",
            "evidence_strength": "weak_evidence",
        },
    ]
    candidates = [
        epp.CandidatePatch(
            idea={"target_source_symbol": "Blueprint.__init__"},
            patch="",
            raw_response="",
            identifiers=[],
            quality_score=0.8,
            covered_original_tests=["test_dotted_name_not_allowed"],
        ),
        epp.CandidatePatch(
            idea={"target_source_symbol": "Blueprint.add_url_rule"},
            patch="",
            raw_response="",
            identifiers=[],
            quality_score=0.9,
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        ),
        epp.CandidatePatch(
            idea={"target_source_symbol": "BlueprintSetupState.add_url_rule"},
            patch="",
            raw_response="",
            identifiers=[],
            quality_score=0.3,
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        ),
    ]
    filtered = epp.select_effective_required_repair_obligations(obligations, filtered_candidates=candidates)
    assert [item["id"] for item in filtered] == [
        "name_core",
        "endpoint_subject",
        "view_func_subject",
    ]


def test_retain_missing_symbol_cluster_subject_candidates_rescues_collected_subject_candidate():
    selected = [
        epp.CandidatePatch(
            idea={
                "target_source_symbol": "Blueprint.add_url_rule",
                "target_validation_subject": "endpoint",
            },
            patch="diff --git a/tests/test_blueprints.py b/tests/test_blueprints.py\n",
            raw_response="",
            identifiers=["test_sweb_enhanced_endpoint"],
            enhanced_identifiers=["test_sweb_enhanced_endpoint"],
            kept=True,
            reason="new enhanced tests reproduced the buggy behavior",
            failing_identifiers=["test_sweb_enhanced_endpoint"],
            eval_status_map={"tests/test_blueprints.py::test_sweb_enhanced_endpoint": "FAILED"},
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
            quality_score=1.0,
        )
    ]
    rescued_candidate = epp.CandidatePatch(
        idea={
            "target_source_symbol": "Blueprint.add_url_rule",
            "target_validation_subject": "view_func_name",
        },
        patch="diff --git a/tests/test_blueprints.py b/tests/test_blueprints.py\n",
        raw_response="",
        identifiers=["test_sweb_enhanced_view_func_name"],
        enhanced_identifiers=["test_sweb_enhanced_view_func_name"],
        kept=False,
        reason="no new test_sweb_enhanced_* identifier failed on buggy code",
        failing_identifiers=[],
        eval_status_map={"tests/test_blueprints.py::test_sweb_enhanced_view_func_name": "PASSED"},
        covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        quality_score=0.2,
    )
    all_candidates = selected + [rescued_candidate]
    updated = epp.retain_missing_symbol_cluster_subject_candidates(
        selected,
        all_candidates,
        keep_top_k=5,
    )
    subjects = [
        str((candidate.idea or {}).get("target_validation_subject"))
        for candidate in updated
        if str((candidate.idea or {}).get("target_source_symbol")) == "Blueprint.add_url_rule"
    ]
    assert subjects == ["endpoint", "view_func_name"]
    assert rescued_candidate.kept is True
    assert "retained_for_symbol_cluster_subject_coverage" in rescued_candidate.reason


def test_classify_repair_topology_returns_symbol_cluster_for_strong_same_symbol_subjects():
    obligations = [
        {
            "id": "endpoint",
            "source_symbol": "Blueprint.add_url_rule",
            "validation_subject": "endpoint",
            "obligation_level": "primary_direct",
            "evidence_strength": "strong_evidence",
            "statement_anchor_tokens": ["if endpoint:"],
        },
        {
            "id": "view_func_name",
            "source_symbol": "Blueprint.add_url_rule",
            "validation_subject": "view_func_name",
            "obligation_level": "primary_sibling",
            "evidence_strength": "strong_evidence",
            "statement_anchor_tokens": ["view_func.__name__"],
        },
    ]
    assert epp.classify_repair_topology(obligations, filtered_candidates=[]) == "symbol_cluster"


def test_classify_repair_topology_returns_statement_local_for_single_strong_statement_symbol():
    obligations = [
        {
            "id": "method_bytes",
            "source_symbol": "Session.request",
            "validation_subject": "method",
            "obligation_level": "primary_direct",
            "evidence_strength": "strong_evidence",
            "statement_anchor_tokens": ["method = builtin_str(method)"],
            "trigger_shape_tokens": ["method.decode("],
        },
    ]
    assert epp.classify_repair_topology(obligations, filtered_candidates=[]) == "statement_local"


def test_classify_repair_topology_returns_single_root_symbol_for_single_strong_symbol_without_statement_tokens():
    obligations = [
        {
            "id": "immutable_symbol",
            "source_symbol": "Symbol",
            "validation_subject": "test_immutable",
            "obligation_level": "primary_direct",
            "evidence_strength": "strong_evidence",
            "statement_anchor_tokens": [],
            "trigger_shape_tokens": [],
        },
    ]
    assert epp.classify_repair_topology(obligations, filtered_candidates=[]) == "single_root_symbol"


def test_has_statement_local_canonical_signature_for_single_active_symbol():
    obligations = [
        {
            "id": "encoded::method",
            "source_symbol": "requests.sessions.Session.request",
            "validation_subject": "method",
            "covered_original_tests": ["tests/test_requests.py::test_encoded_methods"],
            "is_active_fail_to_pass": True,
            "canonical_statement_required": True,
            "canonical_statement_text": "builtin_str(method)",
        }
    ]
    assert epp._has_statement_local_canonical_signature(obligations) is True


def test_has_single_root_symbol_signature_for_single_test_single_edit_target():
    analysis = {"suspicious_symbols": ["Symbol"]}
    strategy = {"edit_targets": ["sympy/core/symbol.py::Symbol"]}
    failure_focus = {"inactive_fail_to_pass_identifiers": ["test_immutable"]}
    assert epp._has_single_root_symbol_signature(
        analysis=analysis,
        strategy=strategy,
        failure_focus=failure_focus,
    ) is True


def test_prefer_single_root_minimal_edit_targets_prefers_single_analysis_source_file():
    strategy = {
        "edit_targets": [
            "sympy/core/_print_helpers.py::Printable",
            "sympy/core/symbol.py::Symbol",
        ],
        "dependency_files": ["sympy/core/tests/test_basic.py"],
        "sufficiency_assessment": "Editing only Printable is sufficient.",
    }
    analysis = {
        "suspicious_symbols": ["Printable"],
        "suggested_repair_scope": ["sympy/core/_print_helpers.py"],
        "affected_components": [
            {"file": "sympy/core/_print_helpers.py", "symbol": "Printable", "reason": "root cause"}
        ],
    }
    failure_focus = {"active_fail_to_pass_identifiers": ["test_immutable"]}

    updated = epp._prefer_single_root_minimal_edit_targets(strategy, analysis, failure_focus)

    assert updated["edit_targets"] == ["sympy/core/_print_helpers.py::Printable"]
    assert "sympy/core/symbol.py" in updated["dependency_files"]


def test_prefer_single_root_minimal_edit_targets_overrides_symptom_file_with_helper_root_cause():
    strategy = {
        "edit_targets": ["sympy/core/symbol.py::Symbol"],
        "dependency_files": ["sympy/core/tests/test_basic.py"],
        "sufficiency_assessment": "Inspect parents too.",
    }
    analysis = {
        "root_cause": "The Symbol symptom comes from a missing __slots__ definition in a parent mixin in the inheritance chain.",
        "propagation_path": "Printable missing slots introduces __dict__ into Symbol instances.",
        "suspicious_symbols": ["Symbol", "Printable"],
        "suggested_repair_scope": ["sympy/core/_print_helpers.py"],
        "affected_components": [
            {"file": "sympy/core/_print_helpers.py", "symbol": "Printable", "reason": "root cause"}
        ],
    }
    failure_focus = {"active_fail_to_pass_identifiers": ["test_immutable"]}

    updated = epp._prefer_single_root_minimal_edit_targets(strategy, analysis, failure_focus)

    assert updated["edit_targets"] == ["sympy/core/_print_helpers.py::Printable"]
    assert "sympy/core/symbol.py" in updated["dependency_files"]


def test_get_active_fail_to_pass_identifiers_matches_short_and_path_style_ids():
    instance = {"FAIL_TO_PASS": '["test_immutable"]', "test_patch": ""}
    status_map = {
        "sympy/core/tests/test_basic.py:test_immutable": "FAILED",
        "sympy/core/tests/test_basic.py::test_other": "PASSED",
    }
    assert epp.get_active_fail_to_pass_identifiers(instance, status_map) == ["test_immutable"]


def test_infer_failure_focus_source_files_uses_test_imports_for_dominant_symbol():
    failure_focus = {
        "active_fail_to_pass_identifiers": [],
        "inactive_fail_to_pass_identifiers": ["test_immutable"],
        "original_test_identifiers": ["test_immutable"],
    }
    code_context = {
        "sympy/core/tests/test_basic.py": """
from sympy.core.symbol import symbols, Symbol, Dummy

def test_immutable():
    assert not hasattr(Symbol('x'), '__dict__')
""",
    }
    inferred = epp._infer_failure_focus_source_files(failure_focus, code_context)
    assert inferred == ["sympy/core/symbol.py"]


def test_backfill_candidate_minimal_structure_uses_failure_focus_symbol_and_subject():
    candidate = epp.CandidatePatch(
        idea={},
        patch="",
        raw_response="",
        identifiers=[],
        covered_original_tests=["test_immutable"],
    )
    failure_focus = {
        "active_fail_to_pass_identifiers": ["test_immutable"],
        "target_test_tracebacks": {
            "test_immutable": "AssertionError: hasattr(Symbol('x'), '__dict__')",
        },
    }
    code_context = {
        "sympy/core/tests/test_basic.py": """
from sympy.core.symbol import Symbol

def test_immutable():
    assert not hasattr(Symbol('x'), '__dict__')
""",
    }
    epp.backfill_candidate_minimal_structure(candidate, failure_focus, code_context)
    assert candidate.idea["target_source_symbol"] == "Symbol"
    assert candidate.idea["target_validation_subject"] == "__dict__"
    assert candidate.idea["covers_obligations"] == ["test_immutable::core::__dict__"]


def test_backfill_candidate_minimal_structure_uses_single_active_test_when_candidate_has_no_covered_tests():
    candidate = epp.CandidatePatch(
        idea={"target_validation_subject": "core"},
        patch="",
        raw_response="",
        identifiers=[],
        covered_original_tests=[],
    )
    failure_focus = {
        "active_fail_to_pass_identifiers": ["test_immutable"],
        "target_test_tracebacks": {
            "test_immutable": "AssertionError: hasattr(Symbol('x'), '__dict__')",
        },
    }
    code_context = {
        "sympy/core/tests/test_basic.py": """
from sympy.core.symbol import Symbol

def test_immutable():
    assert not hasattr(Symbol('x'), '__dict__')
""",
    }
    epp.backfill_candidate_minimal_structure(candidate, failure_focus, code_context)
    assert candidate.covered_original_tests == ["test_immutable"]
    assert candidate.idea["target_source_symbol"] == "Symbol"
    assert candidate.idea["covers_obligations"] == ["test_immutable::core::__dict__"]


def test_build_patch_analysis_feedback_requires_dominant_single_root_symbol_and_file():
    analysis = {
        "root_cause": "helper issue",
        "affected_components": [{"file": "sympy/core/_print_helpers.py", "symbol": "Printable", "reason": "mixin"}],
        "failing_signal": "immutability broken",
        "propagation_path": "Printable adds dict",
        "repair_constraint": "no __dict__",
        "suggested_repair_scope": ["sympy/core/_print_helpers.py"],
        "suspicious_symbols": ["Printable"],
        "repair_obligations": [],
    }
    feedback = epp.build_patch_analysis_feedback(
        analysis,
        required_repair_obligations=[],
        failure_focus={"active_fail_to_pass_identifiers": ["test_immutable"]},
        code_context={
            "sympy/core/tests/test_basic.py": """
from sympy.core.symbol import Symbol

def test_immutable():
    assert not hasattr(Symbol('x'), '__dict__')
""",
        },
    )
    assert feedback is not None
    assert "dominant failing-test symbol" in feedback or "dominant source file" in feedback


def test_has_single_root_helper_root_cause_signature_allows_single_analysis_file_with_two_suspicious_symbols():
    analysis = {
        "root_cause": "A parent mixin in the inheritance chain is missing __slots__ = (), so Symbol gains __dict__.",
        "propagation_path": "Printable breaks the slots chain for Symbol.",
        "suggested_repair_scope": ["sympy/core/_print_helpers.py"],
        "affected_components": [{"file": "sympy/core/_print_helpers.py", "symbol": "Printable", "reason": "root cause"}],
        "suspicious_symbols": ["Symbol", "Printable"],
    }
    failure_focus = {"active_fail_to_pass_identifiers": ["test_immutable"]}
    assert epp._has_single_root_helper_root_cause_signature(analysis, failure_focus) is True


def test_build_patch_strategy_feedback_rejects_extra_files_for_single_root_minimality():
    strategy = {
        "title": "fix symbol slots",
        "approach": "Add __slots__ = () to Printable so Symbol instances no longer get __dict__.",
        "edit_targets": [
            "sympy/core/_print_helpers.py::Printable",
            "sympy/core/symbol.py::Symbol",
        ],
        "dependency_files": ["sympy/core/tests/test_basic.py"],
        "sufficiency_assessment": "Editing only Printable is sufficient because Symbol already defines slots correctly.",
        "risks": [],
    }
    analysis = {
        "suspicious_symbols": ["Printable"],
        "suggested_repair_scope": ["sympy/core/_print_helpers.py"],
        "affected_components": [
            {"file": "sympy/core/_print_helpers.py", "symbol": "Printable", "reason": "root cause"},
            {"file": "sympy/core/tests/test_basic.py", "symbol": "test_immutable", "reason": "failing test"},
        ],
    }
    failure_focus = {"active_fail_to_pass_identifiers": ["test_immutable"]}

    feedback = epp.build_patch_strategy_feedback(
        strategy,
        analysis=analysis,
        failure_focus=failure_focus,
        filtered_candidates=[],
        code_context={"sympy/core/_print_helpers.py": "class Printable:\n    pass\n"},
    )

    assert feedback is not None
    assert "single_root_symbol repair" in feedback


def test_build_patch_strategy_feedback_rejects_symptom_file_when_helper_root_cause_is_minimal():
    strategy = {
        "title": "fix symbol slots",
        "approach": "Reintroduce __slots__ on Symbol and inspect parents.",
        "edit_targets": ["sympy/core/symbol.py::Symbol"],
        "dependency_files": ["sympy/core/_print_helpers.py", "sympy/core/tests/test_basic.py"],
        "sufficiency_assessment": "Editing Symbol should be sufficient, but parents may need inspection.",
        "risks": [],
    }
    analysis = {
        "root_cause": "A parent mixin in the inheritance chain is missing __slots__ = (), so Symbol gains __dict__.",
        "propagation_path": "Printable breaks the slots chain for Symbol.",
        "suspicious_symbols": ["Symbol", "Printable", "__slots__", "__dict__"],
        "suggested_repair_scope": ["sympy/core/_print_helpers.py"],
        "affected_components": [
            {"file": "sympy/core/_print_helpers.py", "symbol": "Printable", "reason": "root cause"},
            {"file": "sympy/core/tests/test_basic.py", "symbol": "test_immutable", "reason": "failing test"},
        ],
    }
    failure_focus = {"active_fail_to_pass_identifiers": ["test_immutable"]}

    feedback = epp.build_patch_strategy_feedback(
        strategy,
        analysis=analysis,
        failure_focus=failure_focus,
        filtered_candidates=[],
        code_context={"sympy/core/_print_helpers.py": "class Printable:\n    pass\n"},
    )

    assert feedback is not None
    assert "helper/mixin/root-cause" in feedback


def test_candidate_has_minimal_structure_requires_backfilled_fields():
    candidate = epp.CandidatePatch(idea={}, patch="", raw_response="", identifiers=[])
    assert epp._candidate_has_minimal_structure(candidate) is False
    candidate.idea = {
        "target_source_symbol": "Symbol",
        "target_validation_subject": "__dict__",
        "covers_obligations": ["test_immutable::core::__dict__"],
    }
    assert epp._candidate_has_minimal_structure(candidate) is True


def test_validate_single_root_symbol_patch_landing_rejects_module_level_insert():
    code_context = {
        "sympy/core/symbol.py": "from .basic import Atom\n\nclass Symbol(Atom):\n    pass\n",
    }
    analysis = {
        "suspicious_symbols": ["Symbol"],
        "affected_components": [{"file": "sympy/core/symbol.py", "symbol": "Symbol", "reason": "root"}],
        "suggested_repair_scope": ["sympy/core/symbol.py"],
    }
    strategy = {
        "edit_targets": ["sympy/core/symbol.py::Symbol"],
    }
    patch = """diff --git a/sympy/core/symbol.py b/sympy/core/symbol.py
--- a/sympy/core/symbol.py
+++ b/sympy/core/symbol.py
@@ -1,4 +1,5 @@
 from .basic import Atom
 
+__slots__ = ('name',)
 class Symbol(Atom):
     pass
"""
    feedback = epp.validate_single_root_symbol_patch_landing(
        patch,
        code_context=code_context,
        analysis=analysis,
        strategy=strategy,
    )
    assert feedback is not None
    assert "outside the target symbol body" in feedback or "did not land inside" in feedback


def test_get_replacement_mode_tolerates_mislabeled_insert_before_anchor_for_replace_span():
    entry = {
        "replacement_mode": "insert_before_anchor",
        "anchor_line_before": "    \"\"\"",
    }
    replacement_lines = [
        "    \"\"\"",
        "",
        "    __slots__ = ()",
        "",
        "    def __str__(self):",
    ]
    assert epp._get_replacement_mode(entry, replacement_lines) == "replace_span"


def test_build_path_semantic_alignment_feedback_allows_single_root_symbol_alignment_when_signature_is_test_name():
    failure_focus = {
        "target_test_tracebacks": {
            "test_immutable": "Traceback\nAssertionError",
        },
        "active_fail_to_pass_identifiers": ["test_immutable"],
        "inactive_fail_to_pass_identifiers": [],
        "original_test_identifiers": ["test_immutable"],
    }
    code_context = {
        "sympy/core/tests/test_basic.py": """
from sympy.core.symbol import Symbol

def test_immutable():
    assert not hasattr(Symbol('x'), '__dict__')
""",
    }
    original_extract = epp.extract_failure_path_signatures
    try:
        epp.extract_failure_path_signatures = lambda failure_focus, code_context=None: {
            "test_immutable": {
                "source_symbol": "test_immutable",
                "alignment_tokens": ["assertionerror"],
            }
        }
        feedback = epp.build_path_semantic_alignment_feedback(
            ["test_immutable"],
            "sympy.core.symbol.Symbol",
            ["assertionerror", "Symbol(", "__dict__"],
            failure_focus,
            code_context,
            "assert not hasattr(Symbol('x'), '__dict__')",
        )
    finally:
        epp.extract_failure_path_signatures = original_extract
    assert feedback is None


def test_build_path_semantic_alignment_feedback_allows_assertionerror_to_source_tokens_for_single_root():
    failure_focus = {
        "target_test_tracebacks": {
            "test_immutable": "Traceback\nAssertionError",
        },
        "active_fail_to_pass_identifiers": ["test_immutable"],
        "inactive_fail_to_pass_identifiers": [],
        "original_test_identifiers": ["test_immutable"],
    }
    code_context = {
        "sympy/core/tests/test_basic.py": """
from sympy.core.symbol import Symbol

def test_immutable():
    assert not hasattr(Symbol('x'), '__dict__')
""",
    }
    original_extract = epp.extract_failure_path_signatures
    try:
        epp.extract_failure_path_signatures = lambda failure_focus, code_context=None: {
            "test_immutable": {
                "source_symbol": "test_immutable",
                "alignment_tokens": ["assertionerror"],
            }
        }
        feedback = epp.build_path_semantic_alignment_feedback(
            ["test_immutable"],
            "sympy.core.symbol.Symbol",
            ["Symbol(", "__dict__", "__slots__"],
            failure_focus,
            code_context,
            "assert not hasattr(Symbol('x'), '__dict__')",
        )
    finally:
        epp.extract_failure_path_signatures = original_extract
    assert feedback is None


def test_single_root_symbol_alignment_match_allows_symbol_basename_and_module_variants():
    assert epp._single_root_symbol_alignment_match(
        "test_immutable",
        "sympy.Symbol",
        "assert not hasattr(Symbol('x'), '__dict__')",
        ["sympy.core.symbol.Symbol"],
    ) is True
    assert epp._single_root_symbol_alignment_match(
        "sympy.core.symbol.Symbol",
        "sympy.Symbol",
        "s = Symbol('x')",
        ["sympy.core.symbol.Symbol"],
    ) is True


def test_extract_required_repair_obligations_preserves_candidate_specific_subject_when_lookup_is_core():
    failure_focus = {
        "target_test_tracebacks": {
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError""",
        }
    }
    code_context = {
        "src/flask/blueprints.py": """class Blueprint:
    def add_url_rule(self, rule, endpoint=None, view_func=None, **options):
        if endpoint:
            assert "." not in endpoint
        if view_func and hasattr(view_func, "__name__"):
            assert "." not in view_func.__name__
""",
    }
    candidates = [
        epp.CandidatePatch(
            idea={
                "target_source_symbol": "Blueprint.add_url_rule",
                "target_validation_subject": "view_func_name",
                "trigger_shape_tokens": ["view_func.__name__", "route("],
                "covers_obligations": ["test_route_decorator_custom_endpoint_with_dots::core"],
            },
            patch="",
            raw_response="",
            identifiers=[],
            quality_score=0.8,
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        )
    ]
    obligations = epp.extract_required_repair_obligations(
        candidates,
        failure_focus=failure_focus,
        code_context=code_context,
    )
    assert [item["id"] for item in obligations] == [
        "test_route_decorator_custom_endpoint_with_dots::core::view_func_name"
    ]
    assert obligations[0]["validation_subject"] == "view_func_name"
    assert obligations[0]["source_symbol"] == "Blueprint.add_url_rule"


def test_patch_strategy_feedback_rejects_single_symbol_sufficiency_when_multiple_symbols_are_covered():
    failure_focus = {
        "target_test_tracebacks": {
            "test_dotted_name_not_allowed": """with pytest.raises(ValueError):
>   flask.Blueprint("app.ui", __name__)
tests/test_blueprints.py:256: Failed""",
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError""",
        }
    }
    candidates = [
        epp.CandidatePatch(
            idea={"target_source_symbol": "Blueprint.__init__"},
            patch="",
            raw_response="",
            identifiers=[],
            covered_original_tests=["test_dotted_name_not_allowed"],
        ),
        epp.CandidatePatch(
            idea={"target_source_symbol": "add_url_rule"},
            patch="",
            raw_response="",
            identifiers=[],
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        ),
    ]
    feedback = build_patch_strategy_feedback(
        {
            "title": "validate_blueprint_name_only",
            "approach": "Add validation in Blueprint.__init__ to reject dotted names.",
            "edit_targets": [
                "src/flask/blueprints.py::Blueprint.__init__",
                "src/flask/blueprints.py::Blueprint.add_url_rule",
            ],
            "dependency_files": [],
            "sufficiency_assessment": "Editing only Blueprint.__init__ is sufficient. No changes are required in add_url_rule.",
            "risks": [],
        },
        failure_focus=failure_focus,
        filtered_candidates=candidates,
    )
    assert feedback is not None
    assert "multiple repair obligations" in feedback or "multiple original failure paths" in feedback or "distinct source symbols" in feedback


def test_patch_strategy_feedback_requires_explicit_exception_type_repair_when_observed_exception_is_wrong():
    failure_focus = {
        "target_test_tracebacks": {
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError""",
        }
    }
    candidates = [
        epp.CandidatePatch(
            idea={"target_source_symbol": "add_url_rule"},
            patch="",
            raw_response="",
            identifiers=[],
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        ),
    ]
    feedback = build_patch_strategy_feedback(
        {
            "title": "validate_endpoint_name",
            "approach": "Add extra validation in Blueprint.add_url_rule for dotted endpoints.",
            "edit_targets": ["src/flask/blueprints.py::Blueprint.add_url_rule"],
            "dependency_files": [],
            "sufficiency_assessment": "Editing add_url_rule is sufficient to validate endpoint names.",
            "risks": [],
        },
        failure_focus=failure_focus,
        filtered_candidates=candidates,
    )
    assert feedback is not None
    assert "exception-type repair obligations" in feedback


def test_patch_strategy_feedback_requires_covering_sibling_assert_validations_in_same_symbol():
    failure_focus = {
        "target_test_tracebacks": {
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError""",
        }
    }
    candidates = [
        epp.CandidatePatch(
            idea={"target_source_symbol": "add_url_rule"},
            patch="",
            raw_response="",
            identifiers=[],
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        ),
    ]
    code_context = {
        "src/flask/blueprints.py": """class Blueprint:
    def add_url_rule(self, rule, endpoint=None, view_func=None):
        if endpoint:
            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
        if view_func and hasattr(view_func, "__name__"):
            assert "." not in view_func.__name__, "Blueprint view function name should not contain dots"
""",
    }
    feedback = build_patch_strategy_feedback(
        {
            "title": "convert_endpoint_assert_only",
            "approach": "In add_url_rule, replace the endpoint assert with raise ValueError for dotted endpoints.",
            "edit_targets": ["src/flask/blueprints.py::Blueprint.add_url_rule"],
            "dependency_files": [],
            "sufficiency_assessment": "Editing add_url_rule to replace the endpoint assert with ValueError is sufficient.",
            "risks": [],
        },
        failure_focus=failure_focus,
        filtered_candidates=candidates,
        code_context=code_context,
    )
    assert feedback is not None
    assert "sibling assert-based validation branches" in feedback


def test_patch_strategy_feedback_requires_all_covered_validation_subjects_in_approach_and_sufficiency():
    candidates = [
        epp.CandidatePatch(
            idea={
                "target_source_symbol": "Blueprint.add_url_rule",
                "target_validation_subject": "endpoint",
                "trigger_shape_tokens": ["endpoint="],
            },
            patch="",
            raw_response="",
            identifiers=[],
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        ),
        epp.CandidatePatch(
            idea={
                "target_source_symbol": "Blueprint.add_url_rule",
                "target_validation_subject": "view_func_name",
                "trigger_shape_tokens": ["view_func", "__name__"],
            },
            patch="",
            raw_response="",
            identifiers=[],
            covered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
        ),
    ]
    feedback = build_patch_strategy_feedback(
        {
            "title": "validate endpoint only",
            "approach": "In Blueprint.add_url_rule, raise ValueError for dotted endpoints.",
            "edit_targets": ["src/flask/blueprints.py::Blueprint.add_url_rule"],
            "dependency_files": [],
            "sufficiency_assessment": "Editing add_url_rule to handle endpoint validation is sufficient.",
            "risks": [],
        },
        filtered_candidates=candidates,
    )
    assert feedback is not None
    assert "validation_subject" in feedback
    assert "endpoint" in feedback
    assert "view_func_name" in feedback


def test_select_candidates_with_path_coverage_prefers_one_per_original_failure_first():
    def make_candidate(name, score, covered_original_tests):
        return epp.CandidatePatch(
            idea={
                "title": name,
                "semantic_bucket": "direct_symptom",
                "covers_original_tests": covered_original_tests,
            },
            patch=(
                "diff --git a/tests/test_demo.py b/tests/test_demo.py\n"
                "--- a/tests/test_demo.py\n"
                "+++ b/tests/test_demo.py\n"
                f"@@ -1,0 +1,2 @@\n+def {name}():\n+    assert True\n"
            ),
            raw_response="",
            identifiers=[name],
            enhanced_identifiers=[name],
            failing_identifiers=[name],
            quality_score=score,
            covered_original_tests=covered_original_tests,
        )

    candidates = [
        make_candidate("test_sweb_enhanced_name_path_best", 0.95, ["test_dotted_name_not_allowed"]),
        make_candidate("test_sweb_enhanced_name_path_second", 0.90, ["test_dotted_name_not_allowed"]),
        make_candidate("test_sweb_enhanced_endpoint_path", 0.60, ["test_route_decorator_custom_endpoint_with_dots"]),
    ]
    selected = select_candidates_with_path_coverage(
        candidates,
        [
            "test_dotted_name_not_allowed",
            "test_route_decorator_custom_endpoint_with_dots",
        ],
        keep_top_k=2,
    )
    assert [candidate.idea["title"] for candidate in selected] == [
        "test_sweb_enhanced_name_path_best",
        "test_sweb_enhanced_endpoint_path",
    ]


def test_select_candidates_with_path_coverage_balances_paths_before_global_fill():
    def make_candidate(name, score, covered_original_tests):
        return epp.CandidatePatch(
            idea={
                "title": name,
                "semantic_bucket": "direct_symptom",
                "covers_original_tests": covered_original_tests,
            },
            patch=(
                "diff --git a/tests/test_demo.py b/tests/test_demo.py\n"
                "--- a/tests/test_demo.py\n"
                "+++ b/tests/test_demo.py\n"
                f"@@ -1,0 +1,2 @@\n+def {name}():\n+    assert True\n"
            ),
            raw_response="",
            identifiers=[name],
            enhanced_identifiers=[name],
            failing_identifiers=[name],
            quality_score=score,
            covered_original_tests=covered_original_tests,
        )

    candidates = [
        make_candidate("test_sweb_enhanced_name_path_best", 0.99, ["test_dotted_name_not_allowed"]),
        make_candidate("test_sweb_enhanced_name_path_second", 0.95, ["test_dotted_name_not_allowed"]),
        make_candidate("test_sweb_enhanced_name_path_third", 0.90, ["test_dotted_name_not_allowed"]),
        make_candidate("test_sweb_enhanced_endpoint_path_best", 0.80, ["test_route_decorator_custom_endpoint_with_dots"]),
        make_candidate("test_sweb_enhanced_endpoint_path_second", 0.75, ["test_route_decorator_custom_endpoint_with_dots"]),
    ]
    selected = select_candidates_with_path_coverage(
        candidates,
        [
            "test_dotted_name_not_allowed",
            "test_route_decorator_custom_endpoint_with_dots",
        ],
        keep_top_k=4,
    )
    assert [candidate.idea["title"] for candidate in selected] == [
        "test_sweb_enhanced_name_path_best",
        "test_sweb_enhanced_endpoint_path_best",
        "test_sweb_enhanced_name_path_second",
        "test_sweb_enhanced_endpoint_path_second",
    ]


def test_select_candidates_with_path_coverage_prefers_distinct_obligations_before_duplicate_same_path():
    def make_candidate(name, score, covered_original_tests, covered_obligations):
        return epp.CandidatePatch(
            idea={"title": name, "semantic_bucket": "direct_symptom"},
            patch="diff --git a/tests/test_demo.py b/tests/test_demo.py\n--- a/tests/test_demo.py\n+++ b/tests/test_demo.py\n@@ -1,0 +1,2 @@\n+def x():\n+    assert True\n",
            raw_response="",
            identifiers=[name],
            enhanced_identifiers=[name],
            failing_identifiers=[name],
            quality_score=score,
            covered_original_tests=covered_original_tests,
            covered_obligations=covered_obligations,
        )

    candidates = [
        make_candidate("path_a_best", 0.99, ["path_a"], ["path_a::endpoint"]),
        make_candidate("path_a_second_same_obligation", 0.95, ["path_a"], ["path_a::endpoint"]),
        make_candidate("path_a_other_obligation", 0.80, ["path_a"], ["path_a::view_func"]),
        make_candidate("path_b_best", 0.70, ["path_b"], ["path_b::core"]),
    ]
    selected = select_candidates_with_path_coverage(
        candidates,
        ["path_a", "path_b"],
        keep_top_k=3,
    )
    assert [candidate.idea["title"] for candidate in selected] == [
        "path_a_best",
        "path_b_best",
        "path_a_other_obligation",
    ]


def test_select_candidates_with_path_coverage_balances_validation_subjects_within_same_path():
    def make_candidate(name, score, covered_original_tests, covered_obligations, subject):
        return epp.CandidatePatch(
            idea={
                "title": name,
                "semantic_bucket": "direct_symptom",
                "target_validation_subject": subject,
            },
            patch="diff --git a/tests/test_demo.py b/tests/test_demo.py\n--- a/tests/test_demo.py\n+++ b/tests/test_demo.py\n@@ -1,0 +1,2 @@\n+def x():\n+    assert True\n",
            raw_response="",
            identifiers=[name],
            enhanced_identifiers=[name],
            failing_identifiers=[name],
            quality_score=score,
            covered_original_tests=covered_original_tests,
            covered_obligations=covered_obligations,
        )

    candidates = [
        make_candidate("path_a_name_best", 0.99, ["path_a"], ["path_a::core"], "name"),
        make_candidate("path_a_name_second", 0.95, ["path_a"], ["path_a::core"], "name"),
        make_candidate("path_a_endpoint", 0.80, ["path_a"], ["path_a::core"], "endpoint"),
        make_candidate("path_b_core", 0.70, ["path_b"], ["path_b::core"], "core"),
    ]
    selected = select_candidates_with_path_coverage(
        candidates,
        ["path_a", "path_b"],
        keep_top_k=4,
    )
    assert [candidate.idea["title"] for candidate in selected] == [
        "path_a_name_best",
        "path_b_core",
        "path_a_endpoint",
        "path_a_name_second",
    ]


def test_score_patch_candidate_eval_prefers_clean_and_f2p_progress():
    baseline = type(
        "Eval",
        (),
        {
            "patch_apply_mode": "no_patch",
            "status_map": {"f2p": "FAILED", "other": "PASSED"},
        },
    )()
    fuzzy = type(
        "Eval",
        (),
        {
            "patch_apply_mode": "fuzzy_apply",
            "status_map": {"f2p": "FAILED", "other": "PASSED"},
        },
    )()
    clean = type(
        "Eval",
        (),
        {
            "patch_apply_mode": "clean_apply",
            "status_map": {"f2p": "PASSED", "other": "PASSED"},
        },
    )()
    assert score_patch_candidate_eval(clean, baseline, ["f2p"]) > score_patch_candidate_eval(
        fuzzy,
        baseline,
        ["f2p"],
    )


def test_structural_patch_helpers_prefer_minimal_elif_to_if_and_penalize_negative_gating():
    minimal_patch = """diff --git a/src/_pytest/skipping.py b/src/_pytest/skipping.py
--- a/src/_pytest/skipping.py
+++ b/src/_pytest/skipping.py
@@ -1,3 +1,3 @@
-    elif (
+    if (
         item._store.get(skipped_by_mark_key, True)
"""
    gating_patch = """diff --git a/src/_pytest/skipping.py b/src/_pytest/skipping.py
--- a/src/_pytest/skipping.py
+++ b/src/_pytest/skipping.py
@@ -1,3 +1,3 @@
-        and rep.skipped
+        and rep.skipped and not item.config.option.runxfail
         and type(rep.longrepr) is tuple
"""
    assert is_minimal_structural_control_flow_patch(minimal_patch) is True
    assert has_structural_elif_to_if_change(minimal_patch) is True
    assert is_minimal_structural_control_flow_patch(gating_patch) is False
    assert is_negative_gating_patch(gating_patch, "--runxfail") is True


def test_score_patch_candidate_choice_prefers_minimal_structural_fix_for_same_eval():
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    baseline = DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})
    candidate = DummyEval("clean_apply", {"f2p": "FAILED", "other": "PASSED"})
    minimal_patch = """diff --git a/src/_pytest/skipping.py b/src/_pytest/skipping.py
--- a/src/_pytest/skipping.py
+++ b/src/_pytest/skipping.py
@@ -1,3 +1,3 @@
-    elif (
+    if (
         item._store.get(skipped_by_mark_key, True)
"""
    gating_patch = """diff --git a/src/_pytest/skipping.py b/src/_pytest/skipping.py
--- a/src/_pytest/skipping.py
+++ b/src/_pytest/skipping.py
@@ -1,3 +1,3 @@
-        and rep.skipped
+        and rep.skipped and not item.config.option.runxfail
         and type(rep.longrepr) is tuple
"""
    analysis = {"root_cause": "report hook has an unreachable elif chain under runxfail"}
    strategy = {"approach": "convert the final elif to a standalone if in pytest_runtest_makereport"}
    assert score_patch_candidate_choice(
        candidate,
        baseline,
        ["f2p"],
        minimal_patch,
        analysis,
        strategy,
        "--runxfail",
    ) > score_patch_candidate_choice(
        candidate,
        baseline,
        ["f2p"],
        gating_patch,
        analysis,
        strategy,
        "--runxfail",
    )


def test_build_patch_feedback_rejects_negative_gating_when_strategy_requires_structural_fix():
    patch = """diff --git a/src/_pytest/skipping.py b/src/_pytest/skipping.py
--- a/src/_pytest/skipping.py
+++ b/src/_pytest/skipping.py
@@ -1,3 +1,3 @@
-        and rep.skipped
+        and rep.skipped and not item.config.option.runxfail
         and type(rep.longrepr) is tuple
"""
    feedback = build_patch_feedback(
        patch,
        None,
        edit_plan=None,
        code_context={"src/_pytest/skipping.py": "def pytest_runtest_makereport():\n    pass\n"},
        analysis={"root_cause": "an elif chain in the report hook becomes unreachable under runxfail"},
        strategy={"approach": "convert the existing elif to a standalone if so the flagged path still executes"},
        original_failure_log="--runxfail causes wrong skip location",
    )
    assert feedback is not None
    assert "contradicts the repair strategy" in feedback
    assert "negative guard" in feedback


def test_build_patch_feedback_rejects_side_path_workaround_when_strategy_requires_elif_to_if():
    side_path_patch = """diff --git a/src/_pytest/skipping.py b/src/_pytest/skipping.py
--- a/src/_pytest/skipping.py
+++ b/src/_pytest/skipping.py
@@ -1,3 +1,7 @@
 def pytest_runtest_makereport(item, call):
     outcome = yield
     rep = outcome.get_result()
+    skipped_by_mark = item._store.get(skipped_by_mark_key, None)
+    if skipped_by_mark and rep.outcome == "skipped":
+        rep.longrepr = item.location
"""
    analysis = {"root_cause": "an elif chain in the report hook becomes unreachable under runxfail"}
    strategy = {"approach": "convert the existing elif to a standalone if so the flagged path still executes"}
    assert strategy_requires_structural_branch_change(analysis, strategy, "--runxfail") is True
    assert has_structural_elif_to_if_change(side_path_patch) is False
    feedback = build_patch_feedback(
        side_path_patch,
        None,
        edit_plan=None,
        code_context={"src/_pytest/skipping.py": "def pytest_runtest_makereport():\n    pass\n"},
        analysis=analysis,
        strategy=strategy,
        original_failure_log="--runxfail causes wrong skip location",
    )
    assert feedback is not None
    assert "standalone `if`" in feedback
    assert "side-path logic" in feedback


def test_validate_patch_landing_rejects_patch_that_only_implements_one_of_multiple_edit_plan_hunks():
    code_context = {
        "src/flask/blueprints.py": """class Blueprint:
    def __init__(self, name, import_name):
        self.name = name
        self.url_prefix = None

    def add_url_rule(self, rule, endpoint=None):
        if endpoint:
            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
""",
    }
    edit_plan = {
        "coverage_check": [
            {
                "target_file": "src/flask/blueprints.py",
                "action": "modify_region",
                "justification": "Validate dotted blueprint names in __init__.",
            },
            {
                "target_file": "src/flask/blueprints.py",
                "action": "modify_region",
                "justification": "Raise ValueError for dotted endpoints in add_url_rule.",
            },
        ],
        "edits": [
            {
                "target_file": "src/flask/blueprints.py",
                "target_symbol": "Blueprint.__init__",
                "anchor_line_before": "        self.name = name",
                "anchor_line_after": "        self.url_prefix = None",
                "replacement_block": """        self.name = name
        if "." in name:
            raise ValueError("Blueprint names should not contain dots.")
        self.url_prefix = None""",
            },
            {
                "target_file": "src/flask/blueprints.py",
                "target_symbol": "Blueprint.add_url_rule__branch",
                "anchor_line_before": "        if endpoint:",
                "anchor_line_after": "            assert \".\" not in endpoint, \"Blueprint endpoints should not contain dots\"",
                "replacement_block": """        if endpoint:
            if "." in endpoint:
                raise ValueError("Blueprint endpoints should not contain dots.")""",
            },
        ],
    }
    partial_patch = """diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py
--- a/src/flask/blueprints.py
+++ b/src/flask/blueprints.py
@@ -1,4 +1,6 @@
 class Blueprint:
     def __init__(self, name, import_name):
         self.name = name
+        if "." in name:
+            raise ValueError("Blueprint names should not contain dots.")
         self.url_prefix = None
"""
    feedback = validate_patch_landing(partial_patch, edit_plan, code_context=code_context)
    assert feedback is not None
    assert "incomplete" in feedback
    assert "add_url_rule__branch" in feedback


def test_validate_patch_landing_accepts_multi_hunk_patch_when_all_planned_replacements_apply():
    code_context = {
        "src/flask/blueprints.py": """class Blueprint:
    def __init__(self, name, import_name):
        self.name = name
        self.url_prefix = None

    def add_url_rule(self, rule, endpoint=None, view_func=None):
        if endpoint:
            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
        if view_func and hasattr(view_func, "__name__"):
            assert (
                "." not in view_func.__name__
            ), "Blueprint view function name should not contain dots"
""",
    }
    edit_plan = {
        "coverage_check": [
            {
                "target_file": "src/flask/blueprints.py",
                "action": "modify_region",
                "justification": "Validate dotted blueprint names in __init__.",
            },
            {
                "target_file": "src/flask/blueprints.py",
                "action": "modify_region",
                "justification": "Raise ValueError for dotted endpoints and view function names in add_url_rule.",
            },
        ],
        "edits": [
            {
                "target_file": "src/flask/blueprints.py",
                "target_symbol": "Blueprint.__init__",
                "covers_obligations": ["test_dotted_name_not_allowed::core"],
                "anchor_line_before": "        self.name = name",
                "anchor_line_after": "        self.url_prefix = None",
                "replacement_block": """        self.name = name
        if "." in name:
            raise ValueError("Blueprint names should not contain dots.")
        self.url_prefix = None""",
            },
            {
                "target_file": "src/flask/blueprints.py",
                "target_symbol": "Blueprint.add_url_rule__branch",
                "covers_obligations": ["test_route_decorator_custom_endpoint_with_dots::endpoint"],
                "anchor_line_before": "        if endpoint:",
                "anchor_line_after": "            assert \".\" not in endpoint, \"Blueprint endpoints should not contain dots\"",
                "replacement_block": """        if endpoint:
            if "." in endpoint:
                raise ValueError("Blueprint endpoints should not contain dots.")""",
            },
            {
                "target_file": "src/flask/blueprints.py",
                "target_symbol": "Blueprint.add_url_rule__branch_tail",
                "covers_obligations": ["test_route_decorator_custom_endpoint_with_dots::view_func_name"],
                "anchor_line_before": "        if view_func and hasattr(view_func, \"__name__\"):",
                "anchor_line_after": "            assert (",
                "replacement_block": """        if view_func and hasattr(view_func, "__name__"):
            if "." in view_func.__name__:
                raise ValueError("Blueprint view function name should not contain dots.")""",
            },
        ],
    }
    patch = """diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py
--- a/src/flask/blueprints.py
+++ b/src/flask/blueprints.py
@@ -1,8 +1,10 @@
 class Blueprint:
     def __init__(self, name, import_name):
         self.name = name
+        if "." in name:
+            raise ValueError("Blueprint names should not contain dots.")
         self.url_prefix = None
 
     def add_url_rule(self, rule, endpoint=None, view_func=None):
         if endpoint:
-            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
+            if "." in endpoint:
+                raise ValueError("Blueprint endpoints should not contain dots.")
         if view_func and hasattr(view_func, "__name__"):
-            assert (
-                "." not in view_func.__name__
-            ), "Blueprint view function name should not contain dots"
+            if "." in view_func.__name__:
+                raise ValueError("Blueprint view function name should not contain dots.")
"""
    assert validate_patch_landing(patch, edit_plan, code_context=code_context) is None


def test_should_prioritize_edit_plan_patch_for_function_targets_only():
    function_plan = {
        "edits": [
            {
                "target_file": "src/_pytest/skipping.py",
                "target_symbol": "pytest_runtest_makereport",
                "anchor_line_before": "a",
                "anchor_line_after": "b",
                "replacement_block": "a\nb",
            }
        ]
    }
    class_plan = {
        "edits": [
            {
                "target_file": "sympy/core/_print_helpers.py",
                "target_symbol": "Printable",
                "anchor_line_before": "a",
                "anchor_line_after": "b",
                "replacement_block": "a\nb",
            }
        ]
    }
    assert should_prioritize_edit_plan_patch(function_plan) is True
    assert should_prioritize_edit_plan_patch(class_plan) is True


def test_parse_json_object_from_wrapped_text():
    parsed = parse_json_object(
        'Here is the idea:\n{"title":"a","goal":"b","template":"t","target_tests":["test_sweb_enhanced_x"],"rationale":"r"}'
    )
    assert parsed["title"] == "a"


def test_patch_strategy_feedback_and_patch_feedback():
    assert (
        build_patch_strategy_feedback(
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["sympy/core/symbol.py"],
                "dependency_files": [],
                "structural_rules": ["restore_unreachable_existing_branch"],
                "forbidden_patterns": ["negative_gating"],
                "sufficiency_assessment": "single-file edit is sufficient",
            }
        )
        is None
    )
    assert build_patch_strategy_feedback({"title": "t"}) is not None
    assert (
        build_patch_strategy_feedback(
            {
                "title": "t",
                "approach": "only validate Blueprint.__init__",
                "edit_targets": [
                    "src/flask/blueprints.py::Blueprint.__init__",
                    "src/flask/blueprints.py::Blueprint.add_url_rule",
                ],
                "dependency_files": [],
                "sufficiency_assessment": "editing only __init__ is enough",
            }
        )
        is not None
    )
    assert (
        build_patch_strategy_feedback(
            {
                "title": "t",
                "approach": "Add ValueError validation in Blueprint.__init__ and Blueprint.add_url_rule.",
                "edit_targets": [
                    "src/flask/blueprints.py::Blueprint.__init__",
                    "src/flask/blueprints.py::Blueprint.add_url_rule",
                ],
                "dependency_files": [],
                "structural_rules": ["elif_to_if"],
                "sufficiency_assessment": "Editing both Blueprint.__init__ and Blueprint.add_url_rule is sufficient.",
                "risks": [],
            },
            analysis={"root_cause": "missing validation for dotted blueprint names and dotted endpoints"},
            original_failure_log="Failed: DID NOT RAISE <class 'ValueError'>\nAssertionError: Blueprint endpoints should not contain dots",
        )
        is not None
    )
    assert (
        build_patch_feedback(
            """diff --git a/tests/test_demo.py b/tests/test_demo.py
--- a/tests/test_demo.py
+++ b/tests/test_demo.py
@@ -1,1 +1,1 @@
-a
+b
""",
            None,
    )
        is not None
    )


def test_extract_strategy_constraints_reads_optional_structural_fields():
    constraints = extract_strategy_constraints(
        {
            "structural_rules": ["elif_to_if", "restore_unreachable_existing_branch"],
            "forbidden_patterns": ["negative_gating", "top_of_function_side_path"],
        }
    )
    assert constraints["structural_rules"] == ["restore_unreachable_existing_branch"]
    assert constraints["forbidden_patterns"] == ["negative_gating", "top_of_function_side_path"]
    assert (
        build_patch_feedback(
            """diff --git a/sympy/core/symbol.py b/sympy/core/symbol.py
--- a/sympy/core/symbol.py
+++ b/sympy/core/symbol.py
@@ -1,1 +1,2 @@
+    __slots__ = ('name',)
""",
            None,
        )
        is None
    )
    assert is_soft_patch_syntax_warning(
        "Unified diff context did not match the source file during syntax validation."
    ) is True
    assert (
        normalize_patch_feedback(
            "Unified diff context did not match the source file during syntax validation."
        )
        is None
    )


def test_fragment_edit_plan_helpers():
    code_context = {
        "sympy/core/symbol.py": """class Symbol(AtomicExpr):
    \"\"\"Represents a symbol.\"\"\"

    is_commutative = True

    def __new__(cls, name, **assumptions):
        return obj
"""
    }
    analysis = {"suspicious_symbols": ["Symbol", "__slots__"]}
    strategy = {"edit_targets": ["sympy/core/symbol.py"]}
    anchor_region = build_class_anchor_region(code_context, analysis, strategy)
    assert anchor_region["target_symbol"] == "Symbol"
    assert anchor_region["anchor_line_before"] == "    is_commutative = True"
    assert anchor_region["anchor_line_after"] == "    def __new__(cls, name, **assumptions):"

    edit_plan = {
        "coverage_check": [
            {
                "target_file": "sympy/core/symbol.py",
                "action": "modify_region",
                "justification": "The Symbol class body needs a local slots fix.",
            }
        ],
        "edits": [
            {
                "target_file": "sympy/core/symbol.py",
                "target_symbol": "Symbol",
                "anchor_line_before": "    is_commutative = True",
                "anchor_line_after": "    def __new__(cls, name, **assumptions):",
                "replacement_block": """    is_commutative = True
    __slots__ = ('name',)

    def __new__(cls, name, **assumptions):""",
            }
        ]
    }
    assert validate_fragment_edit_plan(
        edit_plan,
        [anchor_region],
        required_source_targets=["sympy/core/symbol.py"],
    ) is None
    updated_files, error = apply_fragment_edit_plan(code_context, edit_plan)
    assert error is None
    updated_content = updated_files["sympy/core/symbol.py"]
    assert "__slots__ = ('name',)" in updated_content
    patch = build_unified_diff_from_replacement(
        code_context["sympy/core/symbol.py"],
        updated_content,
        "sympy/core/symbol.py",
    )
    assert "+++ b/sympy/core/symbol.py" in patch
    assert validate_patch_landing(patch, edit_plan, code_context=code_context) is None


def test_validate_patch_landing_rejects_executable_code_inside_docstring():
    code_context = {
        "sympy/core/_print_helpers.py": '''class Printable:
    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way.
    """

    def __str__(self):
        return "x"
'''
    }
    edit_plan = {
        "coverage_check": [
            {
                "target_file": "sympy/core/_print_helpers.py",
                "action": "modify_region",
                "justification": "Printable needs a class-level structural fix.",
            }
        ],
        "edits": [
            {
                "target_file": "sympy/core/_print_helpers.py",
                "target_symbol": "Printable",
                "anchor_line_before": '    """',
                "anchor_line_after": "    def __str__(self):",
                "replacement_block": '''    """
    The default implementation of printing for SymPy classes.

    __slots__ = ()

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way.
    """

    def __str__(self):''',
            }
        ],
    }
    updated_files, error = apply_fragment_edit_plan(code_context, edit_plan)
    assert error is None
    patch = build_unified_diff_from_replacement(
        code_context["sympy/core/_print_helpers.py"],
        updated_files["sympy/core/_print_helpers.py"],
        "sympy/core/_print_helpers.py",
    )
    feedback = validate_patch_landing(patch, edit_plan, code_context=code_context)
    assert "inside the docstring of Printable" in feedback


def test_validate_patch_landing_allows_class_body_edit_after_docstring():
    code_context = {
        "sympy/core/_print_helpers.py": '''class Printable:
    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way.
    """

    def __str__(self):
        return "x"
'''
    }
    edit_plan = {
        "coverage_check": [
            {
                "target_file": "sympy/core/_print_helpers.py",
                "action": "modify_region",
                "justification": "Printable needs a class-level structural fix.",
            }
        ],
        "edits": [
            {
                "target_file": "sympy/core/_print_helpers.py",
                "target_symbol": "Printable",
                "anchor_line_before": '    """',
                "anchor_line_after": "    def __str__(self):",
                "replacement_block": '''    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way.
    """

    __slots__ = ()

    def __str__(self):''',
            }
        ],
    }
    anchor_regions = [
        {
            "target_file": "sympy/core/_print_helpers.py",
            "target_symbol": "Printable",
            "class_anchor_line": "class Printable:",
            "class_anchor_lineno": 1,
            "anchor_line_before": '    """',
            "anchor_line_before_lineno": 7,
            "anchor_line_after": "    def __str__(self):",
            "anchor_line_after_lineno": 9,
            "region_snippet": 'class Printable:\n    """\n    ...\n    """\n\n    def __str__(self):',
        }
    ]
    updated_files, error = apply_fragment_edit_plan(code_context, edit_plan, anchor_regions=anchor_regions)
    assert error is None
    patch = build_unified_diff_from_replacement(
        code_context["sympy/core/_print_helpers.py"],
        updated_files["sympy/core/_print_helpers.py"],
        "sympy/core/_print_helpers.py",
    )
    assert validate_patch_landing(patch, edit_plan, code_context=code_context) is None


def test_class_anchor_prefers_docstring_closing_line_for_structural_insertions():
    code_context = {
        "sympy/core/_print_helpers.py": '''class Printable:
    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way. Natively Python uses ``repr()``
    even if ``str()`` was explicitly requested. Mix in this trait into
    a class to get proper default printing.

    This also adds support for LaTeX printing in jupyter notebooks.
    """

    # Note, we always use the default ordering (lex) in __str__ and __repr__,
    # regardless of the global setting. See issue 5487.
    def __str__(self):
        return "x"
'''
    }
    analysis = {"suspicious_symbols": ["Printable", "__slots__"]}
    strategy = {"edit_targets": ["sympy/core/_print_helpers.py::Printable"]}
    regions = build_edit_anchor_regions(code_context, analysis, strategy)
    printable_region = next(region for region in regions if region["target_symbol"] == "Printable")
    assert printable_region["anchor_line_before"] == '    """'
    assert printable_region["anchor_line_before_lineno"] > 2
    assert code_context["sympy/core/_print_helpers.py"].splitlines()[printable_region["anchor_line_before_lineno"] - 1] == '    """'


def test_apply_fragment_edit_plan_uses_anchor_line_numbers_to_disambiguate_duplicate_quotes():
    code_context = {
        "sympy/core/_print_helpers.py": '''class Printable:
    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way.
    """

    def __str__(self):
        return "x"
'''
    }
    edit_plan = {
        "coverage_check": [
            {
                "target_file": "sympy/core/_print_helpers.py",
                "action": "modify_region",
                "justification": "Printable needs a class-level structural fix.",
            }
        ],
        "edits": [
            {
                "target_file": "sympy/core/_print_helpers.py",
                "target_symbol": "Printable",
                "anchor_line_before": '    """',
                "anchor_line_after": "    def __str__(self):",
                "replacement_block": '''    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way.
    """

    __slots__ = ()

    def __str__(self):''',
            }
        ],
    }
    anchor_regions = [
        {
            "target_file": "sympy/core/_print_helpers.py",
            "target_symbol": "Printable",
            "class_anchor_line": "class Printable:",
            "class_anchor_lineno": 1,
            "anchor_line_before": '    """',
            "anchor_line_before_lineno": 6,
            "anchor_line_after": "    def __str__(self):",
            "anchor_line_after_lineno": 8,
            "region_snippet": 'class Printable:\n    """\n    ...\n    """\n\n    def __str__(self):',
        }
    ]

    updated_files, error = apply_fragment_edit_plan(code_context, edit_plan, anchor_regions=anchor_regions)
    assert error is None
    updated_lines = updated_files["sympy/core/_print_helpers.py"].splitlines()
    assert updated_lines[5] == '    """'
    assert updated_lines[7] == "    __slots__ = ()"
    assert updated_lines[9] == "    def __str__(self):"


def test_fragment_edit_prompt_and_precise_failure_feedback():
    anchor_region = [
        {
            "target_file": "sympy/core/symbol.py",
            "target_symbol": "Symbol",
            "class_anchor_line": "class Symbol(AtomicExpr):",
            "anchor_line_before": "    is_commutative = True",
            "anchor_line_after": "    def __new__(cls, name, **assumptions):",
            "region_snippet": "class Symbol(AtomicExpr):\n    is_commutative = True\n\n    def __new__(cls, name, **assumptions):",
        }
    ]
    prompt = build_fragment_edit_prompt(
        instance={"problem_statement": "bug", "instance_id": "x", "repo": "r"},
        original_failure_log="IndentationError",
        analysis={"root_cause": "bad slots"},
        strategy={"title": "restore slots"},
        anchor_regions=anchor_region,
        required_source_targets=["sympy/core/symbol.py"],
    )
    assert '"edits"' in prompt
    assert "coverage_check" in prompt
    assert "Every replacement_block must begin with anchor_line_before" in prompt
    assert "FUNCTION CONTROL-FLOW RULE" in prompt
    assert "shared pre-branch setup or the earliest controlling branch" in prompt
    assert "STRUCTURED REPORT OBJECT RULE" in prompt
    precise = extract_precise_failure_feedback(
        'File "/testbed/sympy/core/symbol.py", line 13\nIndentationError: unexpected indent',
        "sympy/core/symbol.py",
    )
    assert precise == "indentationerror occurred at sympy/core/symbol.py:13."


def test_fragment_edit_prompt_includes_required_repair_obligations():
    anchor_region = [
        {
            "target_file": "src/flask/blueprints.py",
            "target_symbol": "Blueprint.add_url_rule__branch",
            "class_anchor_line": "class Blueprint:",
            "anchor_line_before": "        if endpoint:",
            "anchor_line_after": "            assert \".\" not in endpoint, \"Blueprint endpoints should not contain dots\"",
            "region_snippet": "        if endpoint:\n            assert \".\" not in endpoint, \"Blueprint endpoints should not contain dots\"",
        }
    ]
    prompt = build_fragment_edit_prompt(
        instance={"problem_statement": "bug", "instance_id": "x", "repo": "r"},
        original_failure_log="AssertionError",
        analysis={"root_cause": "bad validation"},
        strategy={"title": "restore validation"},
        anchor_regions=anchor_region,
        required_source_targets=["src/flask/blueprints.py"],
        required_repair_obligations=[
            {
                "id": "test_route_decorator_custom_endpoint_with_dots::endpoint",
                "source_symbol": "Blueprint.add_url_rule",
                "validation_subject": "endpoint",
                "trigger_shape_tokens": ["endpoint="],
            }
        ],
    )
    assert "Required repair obligations from retained enhanced tests" in prompt
    assert "covers_obligations" in prompt
    assert "test_route_decorator_custom_endpoint_with_dots::endpoint" in prompt


def test_fragment_edit_prompt_includes_pre_conversion_order_rule():
    anchor_region = [
        {
            "target_file": "requests/sessions.py",
            "target_symbol": "Session.request",
            "class_anchor_line": "class Session:",
            "anchor_line_before": "        method = builtin_str(method)",
            "anchor_line_after": "        # Create the Request.",
            "region_snippet": "class Session:\n    def request(...):\n        method = builtin_str(method)\n        # Create the Request.",
        }
    ]
    prompt = build_fragment_edit_prompt(
        instance={"problem_statement": "bug", "instance_id": "x", "repo": "r"},
        original_failure_log="TypeError",
        analysis={"root_cause": "bad conversion order"},
        strategy={"title": "decode bytes before conversion"},
        anchor_regions=anchor_region,
        required_source_targets=["requests/sessions.py"],
        required_repair_obligations=[
            {
                "id": "obligation_1",
                "source_symbol": "Session.request",
                "validation_subject": "method",
                "statement_anchor_tokens": ["builtin_str(method)", "method.decode("],
            }
        ],
    )
    assert "PRE-CONVERSION ORDER RULE" in prompt
    assert "canonical_conversion_statement=builtin_str(method)" in prompt


def test_fragment_edit_prompt_includes_canonical_statement_replacement_rule_for_statement_local():
    anchor_region = [
        {
            "target_file": "requests/sessions.py",
            "target_symbol": "Session.request",
            "class_anchor_line": "class Session:",
            "anchor_line_before": "        method = builtin_str(method)",
            "anchor_line_after": "        # Create the Request.",
            "region_snippet": "class Session:\n    def request(...):\n        method = builtin_str(method)\n        # Create the Request.",
        }
    ]
    prompt = build_fragment_edit_prompt(
        instance={"problem_statement": "bug", "instance_id": "x", "repo": "r"},
        original_failure_log="TypeError",
        analysis={"root_cause": "bad conversion order"},
        strategy={"title": "decode bytes before conversion"},
        anchor_regions=anchor_region,
        required_source_targets=["requests/sessions.py"],
        required_repair_obligations=[
            {
                "id": "obligation_1",
                "source_symbol": "Session.request",
                "validation_subject": "method",
                "statement_anchor_tokens": ["builtin_str(method)", "method.decode("],
            }
        ],
        repair_topology="statement_local",
    )
    assert "CANONICAL STATEMENT REPLACEMENT RULE" in prompt
    assert "Use replacement_mode=replace_span" in prompt
    assert "canonical_statement_text=builtin_str(method)" in prompt


def test_diff_prompt_discourages_lossy_report_location_rewrites():
    prompt = build_diff_from_strategy_prompt(
        instance={"problem_statement": "bug", "instance_id": "x", "repo": "r", "test_patch": ""},
        code_context={"src/_pytest/skipping.py": "def pytest_runtest_makereport():\n    pass\n"},
        original_failure_log="AssertionError",
        filtered_candidates=[],
        analysis={"root_cause": "wrong skip location"},
        strategy={"edit_targets": ["src/_pytest/skipping.py::pytest_runtest_makereport"]},
        feedback=None,
        failure_focus=None,
    )
    assert "STRUCTURED REPORT OBJECT RULE" in prompt
    assert "rep.longrepr" in prompt
    assert "item.location" in prompt


def test_diff_prompt_discourages_negative_gating_for_flag_enabled_hook_bug():
    prompt = build_diff_from_strategy_prompt(
        instance={"problem_statement": "bug", "instance_id": "x", "repo": "r", "test_patch": ""},
        code_context={"src/_pytest/skipping.py": "def pytest_runtest_makereport():\n    pass\n"},
        original_failure_log="--runxfail causes wrong skip location",
        filtered_candidates=[],
        analysis={"root_cause": "hook report logic wrong under runxfail"},
        strategy={"edit_targets": ["src/_pytest/skipping.py::pytest_runtest_makereport"]},
        feedback=None,
        failure_focus=None,
    )
    assert "FLAG-PATH FIX RULE" in prompt
    assert "and not <flag>" in prompt
    assert "if not <flag>" in prompt


def test_minimal_structural_fix_guidance_is_enabled_for_unreachable_elif_hook_bug():
    analysis = {"root_cause": "control-flow issue: later branch is unreachable due to an elif chain in makereport"}
    strategy = {"approach": "change the final elif into a standalone if in the report hook"}
    assert needs_minimal_structural_fix_guidance(
        analysis,
        strategy,
        "--runxfail causes wrong skip location",
    ) is True


def test_diff_prompt_prefers_minimal_structural_fix_when_analysis_points_to_unreachable_branch():
    prompt = build_diff_from_strategy_prompt(
        instance={"problem_statement": "bug", "instance_id": "x", "repo": "r", "test_patch": ""},
        code_context={"src/_pytest/skipping.py": "def pytest_runtest_makereport():\n    pass\n"},
        original_failure_log="--runxfail causes wrong skip location",
        filtered_candidates=[],
        analysis={"root_cause": "an elif chain makes the correction branch unreachable in the report hook"},
        strategy={"edit_targets": ["src/_pytest/skipping.py::pytest_runtest_makereport"]},
        feedback=None,
        failure_focus=None,
    )
    assert "MINIMAL STRUCTURAL FIX RULE" in prompt
    assert "single `elif` -> `if` change" in prompt


def test_multi_edit_anchor_regions_and_apply_fragment_plan():
    code_context = {
        "pkg/a.py": """class Alpha(Base):
    flag = True

    def build(self):
        return 1
""",
        "pkg/b.py": """class Beta(Base):
    enabled = False

    def build(self):
        return 2
""",
    }
    analysis = {"suspicious_symbols": ["Alpha", "Beta"]}
    strategy = {"edit_targets": ["pkg/a.py", "pkg/b.py"]}
    regions = build_edit_anchor_regions(code_context, analysis, strategy)
    assert [region["target_file"] for region in regions] == ["pkg/a.py", "pkg/b.py"]
    edit_plan = {
        "coverage_check": [
            {"target_file": "pkg/a.py", "action": "modify_region", "justification": "Alpha needs a slots fix."},
            {"target_file": "pkg/b.py", "action": "modify_region", "justification": "Beta also needs a slots fix."},
        ],
        "edits": [
            {
                "target_file": "pkg/a.py",
                "target_symbol": "Alpha",
                "anchor_line_before": "    flag = True",
                "anchor_line_after": "    def build(self):",
                "replacement_block": """    flag = True
    __slots__ = ('x',)

    def build(self):""",
            },
            {
                "target_file": "pkg/b.py",
                "target_symbol": "Beta",
                "anchor_line_before": "    enabled = False",
                "anchor_line_after": "    def build(self):",
                "replacement_block": """    enabled = False
    __slots__ = ('y',)

    def build(self):""",
            },
        ]
    }
    assert validate_fragment_edit_plan(edit_plan, regions, required_source_targets=["pkg/a.py", "pkg/b.py"]) is None
    updated_files, error = apply_fragment_edit_plan(code_context, edit_plan)
    assert error is None
    assert "__slots__ = ('x',)" in updated_files["pkg/a.py"]
    assert "__slots__ = ('y',)" in updated_files["pkg/b.py"]


def test_function_branch_anchor_regions_are_exposed():
    code_context = {
        "src/_pytest/skipping.py": """def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if unexpectedsuccess_key in item._store and rep.when == "call":
        rep.outcome = "failed"
    elif item.config.option.runxfail:
        pass
    elif item._store.get(skipped_by_mark_key, True) and rep.skipped:
        rep.longrepr = ("f", 1, "reason")
""",
    }
    analysis = {"suspicious_symbols": ["pytest_runtest_makereport", "skipped_by_mark_key"]}
    strategy = {"edit_targets": ["src/_pytest/skipping.py::pytest_runtest_makereport"]}
    regions = build_edit_anchor_regions(code_context, analysis, strategy)
    region_symbols = [region["target_symbol"] for region in regions]
    assert "pytest_runtest_makereport" in region_symbols
    assert "pytest_runtest_makereport__branch" in region_symbols
    assert "pytest_runtest_makereport__branch_tail" not in region_symbols
    assert "pytest_runtest_makereport__tail" not in region_symbols
    branch_region = next(region for region in regions if region["target_symbol"] == "pytest_runtest_makereport__branch")
    assert "if unexpectedsuccess_key in item._store" in branch_region["anchor_line_before"]


def test_build_edit_anchor_regions_prefers_statement_anchor_tokens_inside_method_body():
    code_context = {
        "requests/sessions.py": """class Session:
    def request(
        self,
        method=None,
        url=None,
        params=None,
        data=None,
    ):
        \"\"\"Constructs a Request.\"\"\"
        method = builtin_str(method)

        # Create the Request.
        req = Request(method=method, url=url, params=params, data=data)
""",
    }
    analysis = {"suspicious_symbols": ["Session.request", "builtin_str"]}
    strategy = {"edit_targets": ["requests/sessions.py::Session.request"]}
    regions = build_edit_anchor_regions(
        code_context,
        analysis,
        strategy,
        required_repair_obligations=[
            {
                "id": "obligation_1",
                "source_symbol": "Session.request",
                "validation_subject": "method",
                "statement_anchor_tokens": ["builtin_str(method)", "method ="],
            }
        ],
    )
    request_region = next(region for region in regions if region["target_symbol"] == "Session.request")
    assert request_region["anchor_line_before"] == "        method = builtin_str(method)"
    assert "# Create the Request." in request_region["anchor_line_after"]


def test_non_dispatcher_long_function_still_exposes_tail_anchor():
    code_context = {
        "pkg/mod.py": """def compute_value(data):
    first = data[0]
    if first > 10:
        return 1
    middle = first + 1
    if middle > 20:
        return 2
    final = middle + 1
    elif_like = False
    if final > 30:
        return 3
    return final
""",
    }
    analysis = {"suspicious_symbols": ["compute_value"]}
    strategy = {"edit_targets": ["pkg/mod.py::compute_value"]}
    regions = build_edit_anchor_regions(code_context, analysis, strategy)
    region_symbols = [region["target_symbol"] for region in regions]
    assert "compute_value__tail" in region_symbols


def test_validate_fragment_edit_plan_enforces_strategy_structural_rules():
    code_context = {
        "src/_pytest/skipping.py": """def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if unexpectedsuccess_key in item._store and rep.when == "call":
        rep.outcome = "failed"
    elif item.config.option.runxfail:
        pass
    elif item._store.get(skipped_by_mark_key, True) and rep.skipped:
        rep.longrepr = ("f", 1, "reason")
""",
    }
    analysis = {"suspicious_symbols": ["pytest_runtest_makereport", "skipped_by_mark_key"]}
    strategy = {
        "edit_targets": ["src/_pytest/skipping.py::pytest_runtest_makereport"],
        "structural_rules": ["elif_to_if"],
        "forbidden_patterns": ["top_of_function_side_path"],
    }
    regions = build_edit_anchor_regions(code_context, analysis, strategy)
    branch_region = next(region for region in regions if region["target_symbol"] == "pytest_runtest_makereport__branch")
    bad_plan = {
        "coverage_check": [
            {
                "target_file": "src/_pytest/skipping.py",
                "action": "modify_region",
                "justification": "fix control flow",
            }
        ],
        "edits": [
            {
                "target_file": branch_region["target_file"],
                "target_symbol": branch_region["target_symbol"],
                "anchor_line_before": branch_region["anchor_line_before"],
                "anchor_line_after": branch_region["anchor_line_after"],
                "replacement_block": """    rep.outcome = "failed"
    if skipped_by_mark_key in item._store and rep.when == "setup":
        rep.longrepr = item.location
    elif item.config.option.runxfail:
        pass""",
            }
        ],
    }
    feedback = validate_fragment_edit_plan(
        bad_plan,
        regions,
        required_source_targets=["src/_pytest/skipping.py"],
        strategy=strategy,
    )
    assert feedback is not None
    assert "strategy.structural_rules=elif_to_if" in feedback or "forbidden_patterns=top_of_function_side_path" in feedback


def test_validate_fragment_edit_plan_requires_covering_all_required_repair_obligations():
    code_context = {
        "src/flask/blueprints.py": """class Blueprint:
    def add_url_rule(self, rule, endpoint=None, view_func=None):
        if endpoint:
            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
        if view_func and hasattr(view_func, "__name__"):
            assert "." not in view_func.__name__, "Blueprint view function name should not contain dots"
""",
    }
    analysis = {"suspicious_symbols": ["Blueprint.add_url_rule"]}
    strategy = {"edit_targets": ["src/flask/blueprints.py::Blueprint.add_url_rule"]}
    regions = build_edit_anchor_regions(code_context, analysis, strategy)
    endpoint_region = next(region for region in regions if region["target_symbol"] == "Blueprint.add_url_rule__branch")
    bad_plan = {
        "coverage_check": [
            {
                "target_file": "src/flask/blueprints.py",
                "action": "modify_region",
                "justification": "convert endpoint validation",
            }
        ],
        "edits": [
            {
                "target_file": endpoint_region["target_file"],
                "target_symbol": endpoint_region["target_symbol"],
                "covers_obligations": ["test_route_decorator_custom_endpoint_with_dots::endpoint"],
                "anchor_line_before": endpoint_region["anchor_line_before"],
                "anchor_line_after": endpoint_region["anchor_line_after"],
                "replacement_block": """        if endpoint:
            if "." in endpoint:
                raise ValueError("Blueprint endpoints should not contain dots")""",
            }
        ],
    }
    feedback = validate_fragment_edit_plan(
        bad_plan,
        regions,
        required_source_targets=["src/flask/blueprints.py"],
        required_repair_obligations=[
            {
                "id": "test_route_decorator_custom_endpoint_with_dots::endpoint",
                "source_symbol": "Blueprint.add_url_rule",
                "validation_subject": "endpoint",
            },
            {
                "id": "test_route_decorator_custom_endpoint_with_dots::view_func_name",
                "source_symbol": "Blueprint.add_url_rule",
                "validation_subject": "view_func_name",
            },
        ],
        strategy=strategy,
    )
    assert feedback is not None
    assert "Missing obligation ids" in feedback


def test_validate_fragment_edit_plan_rejects_no_change_when_file_has_required_obligations():
    code_context = {
        "src/flask/blueprints.py": """class Blueprint:
    def add_url_rule(self, rule, endpoint=None, view_func=None):
        if endpoint:
            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
        if view_func and hasattr(view_func, "__name__"):
            assert "." not in view_func.__name__, "Blueprint view function name should not contain dots"
""",
    }
    analysis = {"suspicious_symbols": ["Blueprint.add_url_rule"]}
    strategy = {"edit_targets": ["src/flask/blueprints.py::Blueprint.add_url_rule"]}
    regions = build_edit_anchor_regions(code_context, analysis, strategy)
    endpoint_region = next(region for region in regions if region["target_symbol"] == "Blueprint.add_url_rule__branch")
    bad_plan = {
        "coverage_check": [
            {
                "target_file": "src/flask/blueprints.py",
                "action": "no_change",
                "justification": "existing assert is enough",
            }
        ],
        "edits": [
            {
                "target_file": endpoint_region["target_file"],
                "target_symbol": endpoint_region["target_symbol"],
                "covers_obligations": ["test_route_decorator_custom_endpoint_with_dots::endpoint"],
                "anchor_line_before": endpoint_region["anchor_line_before"],
                "anchor_line_after": endpoint_region["anchor_line_after"],
                "replacement_block": """        if endpoint:
            if "." in endpoint:
                raise ValueError("Blueprint endpoints should not contain dots")""",
            }
        ],
    }
    feedback = validate_fragment_edit_plan(
        bad_plan,
        regions,
        required_source_targets=["src/flask/blueprints.py"],
        required_repair_obligations=[
            {
                "id": "test_route_decorator_custom_endpoint_with_dots::endpoint",
                "source_symbol": "Blueprint.add_url_rule__branch",
                "validation_subject": "endpoint",
            },
        ],
        strategy=strategy,
    )
    assert feedback is not None
    assert "cannot mark src/flask/blueprints.py as no_change" in feedback


def test_validate_fragment_edit_plan_allows_insert_before_anchor_mode():
    code_context = {
        "requests/sessions.py": """class Session:
    def request(self, method=None):
        method = builtin_str(method)

        # Create the Request.
        req = Request(method=method)
""",
    }
    analysis = {"suspicious_symbols": ["Session.request", "builtin_str"]}
    strategy = {"edit_targets": ["requests/sessions.py::Session.request"]}
    regions = build_edit_anchor_regions(
        code_context,
        analysis,
        strategy,
        required_repair_obligations=[
            {
                "id": "obligation_1",
                "source_symbol": "Session.request",
                "validation_subject": "method",
                "statement_anchor_tokens": ["builtin_str(method)", "method.decode("],
            }
        ],
    )
    request_region = next(region for region in regions if region["target_symbol"] == "Session.request")
    plan = {
        "coverage_check": [
            {
                "target_file": "requests/sessions.py",
                "action": "modify_region",
                "justification": "decode bytes before canonical conversion",
            }
        ],
        "edits": [
            {
                "target_file": request_region["target_file"],
                "target_symbol": request_region["target_symbol"],
                "replacement_mode": "insert_before_anchor",
                "covers_obligations": ["obligation_1"],
                "anchor_line_before": request_region["anchor_line_before"],
                "anchor_line_after": request_region["anchor_line_after"],
                "replacement_block": """        if isinstance(method, bytes):
            method = method.decode('utf-8')
        method = builtin_str(method)

        # Create the Request.""",
            }
        ],
    }
    assert validate_fragment_edit_plan(
        plan,
        regions,
        required_source_targets=["requests/sessions.py"],
        required_repair_obligations=[
            {
                "id": "obligation_1",
                "source_symbol": "Session.request",
                "validation_subject": "method",
                "statement_anchor_tokens": ["builtin_str(method)", "method.decode("],
            }
        ],
        strategy=strategy,
    ) is None


def test_validate_fragment_edit_plan_allows_insert_after_anchor_mode_for_class_body_fix():
    code_context = {
        "sympy/core/_print_helpers.py": '''class Printable:
    """
    The default implementation of printing for SymPy classes.
    """

    def __str__(self):
        return "x"
''',
    }
    analysis = {"suspicious_symbols": ["Printable", "__slots__"]}
    strategy = {"edit_targets": ["sympy/core/_print_helpers.py::Printable"]}
    regions = build_edit_anchor_regions(code_context, analysis, strategy)
    printable_region = next(region for region in regions if region["target_symbol"] == "Printable")
    plan = {
        "coverage_check": [
            {
                "target_file": "sympy/core/_print_helpers.py",
                "action": "modify_region",
                "justification": "insert class-level __slots__ after docstring",
            }
        ],
        "edits": [
            {
                "target_file": printable_region["target_file"],
                "target_symbol": printable_region["target_symbol"],
                "replacement_mode": "insert_after_anchor",
                "covers_obligations": ["test_immutable::core::__dict__ attribute"],
                "anchor_line_before": printable_region["anchor_line_before"],
                "anchor_line_after": printable_region["anchor_line_after"],
                "replacement_block": '''    """

    __slots__ = ()

    def __str__(self):''',
            }
        ],
    }
    assert validate_fragment_edit_plan(
        plan,
        regions,
        required_source_targets=["sympy/core/_print_helpers.py"],
        required_repair_obligations=[
            {
                "id": "test_immutable::core::__dict__ attribute",
                "source_symbol": "Printable",
                "validation_subject": "__dict__ attribute",
            },
        ],
        strategy=strategy,
    ) is None


def test_validate_fragment_edit_plan_rejects_duplicate_docstring_delimiter_for_insert_after_anchor():
    code_context = {
        "sympy/core/_print_helpers.py": '''class Printable:
    """
    The default implementation of printing for SymPy classes.
    """

    def __str__(self):
        return "x"
''',
    }
    analysis = {"suspicious_symbols": ["Printable", "__slots__"]}
    strategy = {"edit_targets": ["sympy/core/_print_helpers.py::Printable"]}
    regions = build_edit_anchor_regions(code_context, analysis, strategy)
    printable_region = next(region for region in regions if region["target_symbol"] == "Printable")
    plan = {
        "coverage_check": [
            {
                "target_file": "sympy/core/_print_helpers.py",
                "action": "modify_region",
                "justification": "bad docstring insertion",
            }
        ],
        "edits": [
            {
                "target_file": printable_region["target_file"],
                "target_symbol": printable_region["target_symbol"],
                "replacement_mode": "insert_after_anchor",
                "covers_obligations": ["test_immutable::core::__dict__ attribute"],
                "anchor_line_before": printable_region["anchor_line_before"],
                "anchor_line_after": printable_region["anchor_line_after"],
                "replacement_block": '''    """

    __slots__ = ()

    """
    def __str__(self):''',
            }
        ],
    }
    feedback = validate_fragment_edit_plan(
        plan,
        regions,
        required_source_targets=["sympy/core/_print_helpers.py"],
        required_repair_obligations=[
            {
                "id": "test_immutable::core::__dict__ attribute",
                "source_symbol": "Printable",
                "validation_subject": "__dict__ attribute",
            },
        ],
        strategy=strategy,
    )
    assert feedback is not None
    assert "Do not repeat the docstring delimiter" in feedback


def test_validate_fragment_edit_plan_rejects_redundant_single_root_symptom_edit():
    anchor_regions = [
        {
            "target_file": "sympy/core/symbol.py",
            "target_symbol": "Symbol",
            "anchor_line_before": "class Symbol(AtomicExpr):",
            "anchor_line_after": "    __slots__ = ('name',)",
            "region_snippet": "class Symbol(AtomicExpr):\n    __slots__ = ('name',)\n    is_symbol = True",
        }
    ]
    edit_plan = {
        "coverage_check": [
            {
                "target_file": "sympy/core/symbol.py",
                "action": "modify_region",
                "justification": "touch symbol",
            }
        ],
        "edits": [
            {
                "target_file": "sympy/core/symbol.py",
                "target_symbol": "Symbol",
                "anchor_line_before": "class Symbol(AtomicExpr):",
                "anchor_line_after": "    __slots__ = ('name',)",
                "replacement_mode": "replace_span",
                "replacement_block": "class Symbol(AtomicExpr):\n    __slots__ = ('name',)",
                "covers_obligations": ["test_immutable::core::__dict__"],
            }
        ],
    }
    feedback = validate_fragment_edit_plan(
        edit_plan,
        anchor_regions,
        required_source_targets=["sympy/core/symbol.py"],
        required_repair_obligations=[
            {
                "id": "test_immutable::core::__dict__",
                "source_symbol": "Symbol",
                "validation_subject": "__dict__",
            }
        ],
        repair_topology="single_root_symbol",
    )

    assert feedback is not None
    assert "redundant edit" in feedback


def test_validate_fragment_edit_plan_rejects_insert_before_anchor_for_statement_local_canonical_repair():
    code_context = {
        "requests/sessions.py": """class Session:
    def request(self, method=None):
        method = builtin_str(method)

        # Create the Request.
        req = Request(method=method)
""",
    }
    analysis = {"suspicious_symbols": ["Session.request", "builtin_str"]}
    strategy = {"edit_targets": ["requests/sessions.py::Session.request"]}
    regions = build_edit_anchor_regions(
        code_context,
        analysis,
        strategy,
        required_repair_obligations=[
            {
                "id": "obligation_1",
                "source_symbol": "Session.request",
                "validation_subject": "method",
                "statement_anchor_tokens": ["builtin_str(method)", "method.decode("],
            }
        ],
    )
    request_region = next(region for region in regions if region["target_symbol"] == "Session.request")
    plan = {
        "coverage_check": [
            {
                "target_file": "requests/sessions.py",
                "action": "modify_region",
                "justification": "decode bytes before canonical conversion",
            }
        ],
        "edits": [
            {
                "target_file": request_region["target_file"],
                "target_symbol": request_region["target_symbol"],
                "replacement_mode": "insert_before_anchor",
                "covers_obligations": ["obligation_1"],
                "anchor_line_before": request_region["anchor_line_before"],
                "anchor_line_after": request_region["anchor_line_after"],
                "replacement_block": """        if isinstance(method, bytes):
            method = method.decode('utf-8')
        method = builtin_str(method)

        # Create the Request.""",
            }
        ],
    }
    feedback = validate_fragment_edit_plan(
        plan,
        regions,
        required_source_targets=["requests/sessions.py"],
        required_repair_obligations=[
            {
                "id": "obligation_1",
                "source_symbol": "Session.request",
                "validation_subject": "method",
                "statement_anchor_tokens": ["builtin_str(method)", "method.decode("],
            }
        ],
        strategy=strategy,
        repair_topology="statement_local",
    )
    assert feedback is not None
    assert "full statement replacement" in feedback
    assert "insert_before_anchor" in feedback


def test_apply_fragment_edit_plan_supports_inferred_insert_before_anchor_mode():
    code_context = {
        "requests/sessions.py": """class Session:
    def request(self, method=None):
        method = builtin_str(method)

        # Create the Request.
        req = Request(method=method)
""",
    }
    edit_plan = {
        "coverage_check": [
            {
                "target_file": "requests/sessions.py",
                "action": "modify_region",
                "justification": "decode before conversion",
            }
        ],
        "edits": [
            {
                "target_file": "requests/sessions.py",
                "target_symbol": "Session.request",
                "anchor_line_before": "        method = builtin_str(method)",
                "anchor_line_after": "        # Create the Request.",
                "replacement_block": """        if isinstance(method, bytes):
            method = method.decode('utf-8')
        method = builtin_str(method)

        # Create the Request.""",
            }
        ],
    }
    updated_files, error = apply_fragment_edit_plan(code_context, edit_plan)
    assert error is None
    updated = updated_files["requests/sessions.py"]
    assert "if isinstance(method, bytes):" in updated
    assert updated.index("if isinstance(method, bytes):") < updated.index("method = builtin_str(method)")


def test_select_effective_required_repair_obligations_prioritizes_active_fail_to_pass_symbols():
    obligations = [
        {
            "id": "encoded::method",
            "source_symbol": "requests.sessions.Session.request",
            "validation_subject": "method",
            "covered_original_tests": ["tests/test_requests.py::test_encoded_methods"],
            "is_active_fail_to_pass": True,
            "obligation_level": "primary_direct",
            "evidence_strength": "weak_evidence",
            "statement_anchor_tokens": ["builtin_str(method)"],
        },
        {
            "id": "redirect::method",
            "source_symbol": "requests.sessions.SessionRedirectMixin.resolve_redirects",
            "validation_subject": "method",
            "covered_original_tests": ["tests/test_requests.py::test_HTTP_302_ALLOW_REDIRECT_GET"],
            "is_active_fail_to_pass": False,
            "obligation_level": "primary_direct",
            "evidence_strength": "weak_evidence",
            "statement_anchor_tokens": ["method ="],
        },
    ]
    filtered = epp.select_effective_required_repair_obligations(obligations, filtered_candidates=[])
    assert [item["id"] for item in filtered] == ["encoded::method"]


def test_apply_fragment_edit_plan_tolerates_mislabeled_replace_span_for_insert_before_anchor():
    code_context = {
        "requests/sessions.py": """class Session:
    def request(self, method=None):
        method = builtin_str(method)

        # Create the Request.
        req = Request(method=method)
""",
    }
    edit_plan = {
        "coverage_check": [
            {
                "target_file": "requests/sessions.py",
                "action": "modify_region",
                "justification": "decode before conversion",
            }
        ],
        "edits": [
            {
                "target_file": "requests/sessions.py",
                "target_symbol": "Session.request",
                "replacement_mode": "replace_span",
                "anchor_line_before": "        method = builtin_str(method)",
                "anchor_line_after": "        # Create the Request.",
                "replacement_block": """        if isinstance(method, bytes):
            method = method.decode('utf-8')
        method = builtin_str(method)

        # Create the Request.""",
            }
        ],
    }
    updated_files, error = apply_fragment_edit_plan(code_context, edit_plan)
    assert error is None
    updated = updated_files["requests/sessions.py"]
    assert "if isinstance(method, bytes):" in updated
    assert updated.index("if isinstance(method, bytes):") < updated.index("method = builtin_str(method)")


def test_validate_fragment_edit_plan_rejects_post_conversion_guard_order():
    code_context = {
        "requests/sessions.py": """class Session:
    def request(self, method=None):
        method = builtin_str(method)
        # Create the Request.
        req = Request(method=method)
""",
    }
    analysis = {"suspicious_symbols": ["Session.request"]}
    strategy = {"edit_targets": ["requests/sessions.py::Session.request"]}
    regions = build_edit_anchor_regions(
        code_context,
        analysis,
        strategy,
        required_repair_obligations=[
            {
                "id": "obligation_1",
                "source_symbol": "Session.request",
                "validation_subject": "method",
                "statement_anchor_tokens": ["builtin_str(method)", "method.decode("],
            }
        ],
    )
    request_region = next(region for region in regions if region["target_symbol"] == "Session.request")
    bad_plan = {
        "coverage_check": [
            {
                "target_file": "requests/sessions.py",
                "action": "modify_region",
                "justification": "fix conversion order",
            }
        ],
        "edits": [
            {
                "target_file": request_region["target_file"],
                "target_symbol": request_region["target_symbol"],
                "covers_obligations": ["obligation_1"],
                "anchor_line_before": request_region["anchor_line_before"],
                "anchor_line_after": request_region["anchor_line_after"],
                "replacement_block": """        method = builtin_str(method)
        if isinstance(method, bytes):
            method = method.decode('utf-8')
        # Create the Request.""",
            }
        ],
    }
    feedback = validate_fragment_edit_plan(
        bad_plan,
        regions,
        required_source_targets=["requests/sessions.py"],
        required_repair_obligations=[
            {
                "id": "obligation_1",
                "source_symbol": "Session.request",
                "validation_subject": "method",
                "statement_anchor_tokens": ["builtin_str(method)", "method.decode("],
            }
        ],
        strategy=strategy,
    )
    assert feedback is not None
    assert "before the canonical conversion statement" in feedback
    assert "view_func_name" in feedback


def test_synthesize_structural_patch_from_strategy_builds_minimal_elif_to_if_diff():
    code_context = {
        "src/_pytest/skipping.py": """def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if unexpectedsuccess_key in item._store and rep.when == "call":
        rep.outcome = "failed"
    elif item.config.option.runxfail:
        pass
    elif item._store.get(skipped_by_mark_key, True) and rep.skipped:
        rep.longrepr = ("f", 1, "reason")
""",
    }
    analysis = {"suspicious_symbols": ["pytest_runtest_makereport", "skipped_by_mark_key"]}
    strategy = {
        "edit_targets": ["src/_pytest/skipping.py::pytest_runtest_makereport"],
        "structural_rules": ["elif_to_if"],
        "forbidden_patterns": ["negative_gating", "top_of_function_side_path"],
        "approach": "convert the existing elif to a standalone if so the flagged path still executes",
    }
    patch, error = synthesize_structural_patch_from_strategy(
        code_context=code_context,
        analysis=analysis,
        strategy=strategy,
        original_failure_log="--runxfail causes wrong skip location",
    )
    assert error is None
    assert has_structural_elif_to_if_change(patch) is True
    assert "-    elif item._store.get(skipped_by_mark_key, True) and rep.skipped:" in patch
    assert "+    if item._store.get(skipped_by_mark_key, True) and rep.skipped:" in patch


def test_enhanced_test_prompt_prefers_near_neighbor_skip_location_variants():
    instance = {
        "instance_id": "pytest-dev__pytest-7432",
        "repo": "pytest-dev/pytest",
        "problem_statement": "wrong skip location under --runxfail",
        "test_patch": "",
    }
    prompt = build_enhanced_test_prompt(
        instance=instance,
        code_context={"testing/test_skipping.py": "def test_x():\n    pass\n"},
        original_failure_log=(
            "Failed: nomatch: 'SKIPPED [1] test_sample.py:2: unconditional skip'\n"
            "=========================== short test summary info ============================\n"
        ),
        failure_focus={"dominant_errors": ["AssertionError"]},
    )
    assert "single skip-location reporting mismatch" in prompt
    assert "Do NOT broaden into nested functions, multiple skipped tests, multiple marks, or skipif/xfail combinations" in prompt


def test_required_source_targets_and_failure_summary_helpers():
    strategy = {
        "edit_targets": ["sympy/core/symbol.py", "tests/test_demo.py"],
        "dependency_files": ["sympy/core/basic.py", "sympy/core/tests/test_basic.py"],
    }
    assert get_required_source_targets(strategy) == ["sympy/core/symbol.py"]
    summary = summarize_failure_log("Traceback\nAssertionError: boom\nextra context")
    assert "AssertionError" in summary


def test_patch_acceptance_rules():
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    baseline = DummyEval("no_patch", {"a": "FAILED", "b": "PASSED"})
    clean_candidate = DummyEval("clean_apply", {"a": "FAILED", "b": "PASSED"})
    improved_candidate = DummyEval("clean_apply", {"a": "PASSED", "b": "PASSED"})
    bad_candidate = DummyEval("fuzzy_apply", {"a": "FAILED", "b": "PASSED"})
    empty_candidate = DummyEval("clean_apply", {})
    hard_fail_candidate = DummyEval("clean_apply", {"a": "PASSED"}, "IndentationError: unexpected indent")
    semantic_fail = type("SemanticFail", (), {"passed": False, "failed_identifiers": ["test_sweb_enhanced_x"]})()

    assert patch_improves_metrics(improved_candidate, baseline) is True
    assert patch_improves_metrics(empty_candidate, baseline) is False
    assert get_patch_acceptance_reason(clean_candidate, baseline) == (
        False,
        "rejected_no_metric_improvement",
    )
    assert get_patch_acceptance_reason(improved_candidate, baseline) == (
        True,
        "accepted_clean_apply_with_improvement",
    )
    assert get_patch_acceptance_reason(bad_candidate, baseline) == (
        False,
        "rejected_patch_apply_failed",
    )
    assert get_patch_acceptance_reason(empty_candidate, baseline) == (
        False,
        "rejected_no_metric_improvement",
    )
    assert get_patch_acceptance_reason(hard_fail_candidate, baseline) == (
        False,
        "rejected_hard_failure_indentationerror",
    )
    assert get_patch_acceptance_reason(clean_candidate, baseline, semantic_oracle=semantic_fail) == (
        False,
        "rejected_no_metric_improvement",
    )


def test_generate_patch_with_strategy_retains_best_plan_patch(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.model = "dummy-model"

        def complete(self, prompt):
            bad_patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,2 @@
 class Foo:
-    x = 1
+    x = 999
 """
            return "raw bad patch", bad_patch

    instance = {
        "instance_id": "demo-instance",
        "repo": "demo/repo",
        "problem_statement": "bug",
        "test_patch": """diff --git a/tests/test_demo.py b/tests/test_demo.py
--- a/tests/test_demo.py
+++ b/tests/test_demo.py
@@ -1,0 +1,2 @@
+def test_demo_bug():
+    assert True
""",
    }
    code_context = {
        "pkg/mod.py": """class Foo:
    x = 1
""",
    }
    baseline_eval = DummyEval("no_patch", {"test_demo_bug": "FAILED", "other": "PASSED"})
    rejected_eval = DummyEval("clean_apply", {"test_demo_bug": "FAILED", "other": "PASSED"})

    monkeypatch.setattr(epp, "generate_patch_analysis", lambda **kwargs: ({}, 1))
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["pkg/mod.py::Foo"],
                "dependency_files": [],
                "sufficiency_assessment": "enough",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(
        epp,
        "generate_fragment_edit_plan",
        lambda **kwargs: (
            {
                "coverage_check": [
                    {"target_file": "pkg/mod.py", "action": "modify_region", "justification": "fix foo"}
                ],
                "edits": [
                    {
                        "target_file": "pkg/mod.py",
                        "target_symbol": "Foo",
                        "anchor_line_before": "class Foo:",
                        "anchor_line_after": "    x = 1",
                        "replacement_block": "class Foo:\n    x = 1\n    y = 2",
                    }
                ],
            },
            1,
        ),
    )
    monkeypatch.setattr(
        epp,
        "build_edit_anchor_regions",
        lambda **kwargs: [
            {
                "target_file": "pkg/mod.py",
                "target_symbol": "Foo",
                "class_anchor_line": "class Foo:",
                "anchor_line_before": "class Foo:",
                "anchor_line_after": "    x = 1",
                "region_snippet": "class Foo:\n    x = 1",
            }
        ],
    )
    monkeypatch.setattr(
        epp,
        "run_eval",
        lambda **kwargs: rejected_eval,
    )
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)

    original_build_patch_feedback = epp.build_patch_feedback

    def fake_build_patch_feedback(
        patch,
        patch_error,
        edit_plan=None,
        code_context=None,
        skip_syntax_check=False,
        analysis=None,
        strategy=None,
        original_failure_log="",
    ):
        if "y = 2" in patch:
            return None
        return "bad patch"

    monkeypatch.setattr(epp, "build_patch_feedback", fake_build_patch_feedback)

    result = generate_patch_with_strategy(
        responder=DummyResponder(),
        instance=instance,
        code_context=code_context,
        original_failure_log="AssertionError",
        filtered_candidates=[],
        max_candidate_attempts=2,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus=None,
    )

    assert result.accepted is False
    assert "y = 2" in result.patch
    assert result.acceptance_reason == "rejected_no_metric_improvement"


def test_generate_patch_with_strategy_prefers_best_evaluated_patch(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.model = "dummy-model"
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            if self.calls == 1:
                patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,3 @@
 class Foo:
     x = 1
+    y = 2
 """
            else:
                patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,3 @@
 class Foo:
     x = 1
+    z = 999
 """
            return "raw", patch

    instance = {
        "instance_id": "demo-instance",
        "repo": "demo/repo",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["f2p"]',
        "test_patch": "",
    }
    code_context = {
        "pkg/mod.py": """class Foo:
    x = 1
""",
    }
    baseline_eval = DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})
    evals = iter(
        [
            DummyEval("clean_apply", {"f2p": "PASSED", "other": "PASSED"}),
            DummyEval("fuzzy_apply", {"f2p": "FAILED", "other": "FAILED"}, "SyntaxError: bad"),
        ]
    )

    monkeypatch.setattr(epp, "generate_patch_analysis", lambda **kwargs: ({}, 1))
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["pkg/mod.py::Foo"],
                "dependency_files": [],
                "sufficiency_assessment": "enough",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(epp, "build_edit_anchor_regions", lambda **kwargs: [])
    monkeypatch.setattr(epp, "generate_fragment_edit_plan", lambda **kwargs: (None, 0))
    monkeypatch.setattr(epp, "run_eval", lambda **kwargs: next(evals))
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)
    monkeypatch.setattr(epp, "build_patch_feedback", lambda *args, **kwargs: None)

    result = generate_patch_with_strategy(
        responder=DummyResponder(),
        instance=instance,
        code_context=code_context,
        original_failure_log="AssertionError",
        filtered_candidates=[],
        max_candidate_attempts=2,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus=None,
    )

    assert "y = 2" in result.patch
    assert result.candidate_eval.patch_apply_mode == "clean_apply"


def test_generate_patch_with_strategy_prefers_earlier_best_eval_on_tie(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.model = "dummy-model"
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            if self.calls == 1:
                patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,3 @@
 class Foo:
     x = 1
+    y = 2
 """
            else:
                patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,3 @@
 class Foo:
     x = 1
+    z = 3
 """
            return "raw", patch

    instance = {
        "instance_id": "demo-instance",
        "repo": "demo/repo",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["f2p"]',
        "test_patch": "",
    }
    code_context = {"pkg/mod.py": "class Foo:\n    x = 1\n"}
    baseline_eval = DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})
    evals = iter(
        [
            DummyEval("clean_apply", {"f2p": "FAILED", "other": "PASSED"}),
            DummyEval("clean_apply", {"f2p": "FAILED", "other": "PASSED"}),
        ]
    )

    monkeypatch.setattr(epp, "generate_patch_analysis", lambda **kwargs: ({}, 1))
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["pkg/mod.py::Foo"],
                "dependency_files": [],
                "sufficiency_assessment": "enough",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(epp, "build_edit_anchor_regions", lambda **kwargs: [])
    monkeypatch.setattr(epp, "generate_fragment_edit_plan", lambda **kwargs: (None, 0))
    monkeypatch.setattr(epp, "run_eval", lambda **kwargs: next(evals))
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)
    monkeypatch.setattr(epp, "build_patch_feedback", lambda *args, **kwargs: None)

    result = generate_patch_with_strategy(
        responder=DummyResponder(),
        instance=instance,
        code_context=code_context,
        original_failure_log="AssertionError",
        filtered_candidates=[],
        max_candidate_attempts=2,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus=None,
    )

    assert "y = 2" in result.patch


def test_generate_patch_with_strategy_prioritizes_function_edit_plan_patch(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.model = "dummy-model"

        def complete(self, prompt):
            bad_patch = """diff --git a/src/_pytest/skipping.py b/src/_pytest/skipping.py
--- a/src/_pytest/skipping.py
+++ b/src/_pytest/skipping.py
@@ -1,2 +1,2 @@
-a
+b
 """
            return "raw bad patch", bad_patch

    instance = {
        "instance_id": "pytest-dev__pytest-7432",
        "repo": "pytest-dev/pytest",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["f2p"]',
        "test_patch": "",
    }
    code_context = {
        "src/_pytest/skipping.py": """def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    return rep
""",
    }
    baseline_eval = DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})
    recorded_patches = []

    def fake_run_eval(**kwargs):
        recorded_patches.append(kwargs["code_patch"])
        if "skipped_by_mark" in kwargs["code_patch"]:
            return DummyEval("clean_apply", {"f2p": "FAILED", "other": "PASSED"})
        return DummyEval("apply_failed", {}, "reject")

    monkeypatch.setattr(epp, "generate_patch_analysis", lambda **kwargs: ({}, 1))
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["src/_pytest/skipping.py::pytest_runtest_makereport"],
                "dependency_files": [],
                "sufficiency_assessment": "enough",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(
        epp,
        "build_edit_anchor_regions",
        lambda **kwargs: [
            {
                "target_file": "src/_pytest/skipping.py",
                "target_symbol": "pytest_runtest_makereport",
                "class_anchor_line": "def pytest_runtest_makereport(item, call):",
                "anchor_line_before": "    outcome = yield",
                "anchor_line_after": "    rep = outcome.get_result()",
                "region_snippet": "def pytest_runtest_makereport(item, call):\n    outcome = yield\n    rep = outcome.get_result()",
            }
        ],
    )
    monkeypatch.setattr(
        epp,
        "generate_fragment_edit_plan",
        lambda **kwargs: (
            {
                "coverage_check": [
                    {
                        "target_file": "src/_pytest/skipping.py",
                        "action": "modify_region",
                        "justification": "fix function",
                    }
                ],
                "edits": [
                    {
                        "target_file": "src/_pytest/skipping.py",
                        "target_symbol": "pytest_runtest_makereport",
                        "anchor_line_before": "    outcome = yield",
                        "anchor_line_after": "    rep = outcome.get_result()",
                        "replacement_block": "    outcome = yield\n    rep = outcome.get_result()\n    skipped_by_mark = item._store.get(skipped_by_mark_key, False)",
                    }
                ],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "run_eval", fake_run_eval)
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)
    monkeypatch.setattr(epp, "build_patch_feedback", lambda *args, **kwargs: None)

    result = generate_patch_with_strategy(
        responder=DummyResponder(),
        instance=instance,
        code_context=code_context,
        original_failure_log="AssertionError",
        filtered_candidates=[],
        max_candidate_attempts=1,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus=None,
    )

    assert recorded_patches
    assert "skipped_by_mark" in recorded_patches[0]
    assert "skipped_by_mark" in result.patch


def test_should_lock_assembled_edit_plan_patch_requires_multiple_edits():
    assert epp.should_lock_assembled_edit_plan_patch(None) is False
    assert epp.should_lock_assembled_edit_plan_patch({"edits": [{}]}) is False
    assert epp.should_lock_assembled_edit_plan_patch({"edits": [{}, {}]}) is True


def test_should_lock_plan_derived_patch_for_single_or_multi_edit_plans():
    assert epp.should_lock_plan_derived_patch(None) is False
    assert epp.should_lock_plan_derived_patch({"edits": []}) is False
    assert epp.should_lock_plan_derived_patch({"edits": [{"target_file": "pkg/mod.py"}]}) is True
    assert epp.should_lock_plan_derived_patch({"edits": [{"target_file": "pkg/a.py"}, {"target_file": "pkg/b.py"}]}) is True


def test_generate_patch_with_strategy_locks_multi_edit_plan_patch_before_free_diff(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.model = "dummy-model"
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            bad_patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,4 +1,4 @@
 class Foo:
     x = 1
-    y = 2
+    y = 999
 """
            return "raw bad patch", bad_patch

    responder = DummyResponder()
    instance = {
        "instance_id": "demo-instance",
        "repo": "demo/repo",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["f2p"]',
        "test_patch": "",
    }
    code_context = {
        "pkg/mod.py": """class Foo:
    x = 1
    y = 2
    z = 3
""",
    }
    baseline_eval = DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})
    recorded_patches = []

    def fake_run_eval(**kwargs):
        recorded_patches.append(kwargs["code_patch"])
        return DummyEval("clean_apply", {"f2p": "FAILED", "other": "PASSED"})

    monkeypatch.setattr(epp, "generate_patch_analysis", lambda **kwargs: ({}, 1))
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["pkg/mod.py::Foo"],
                "dependency_files": [],
                "sufficiency_assessment": "enough",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(
        epp,
        "build_edit_anchor_regions",
        lambda **kwargs: [
            {
                "target_file": "pkg/mod.py",
                "target_symbol": "Foo",
                "class_anchor_line": "class Foo:",
                "anchor_line_before": "    x = 1",
                "anchor_line_after": "    y = 2",
                "region_snippet": "class Foo:\n    x = 1\n    y = 2",
            },
            {
                "target_file": "pkg/mod.py",
                "target_symbol": "Foo",
                "class_anchor_line": "class Foo:",
                "anchor_line_before": "    y = 2",
                "anchor_line_after": "    z = 3",
                "region_snippet": "class Foo:\n    y = 2\n    z = 3",
            },
        ],
    )
    monkeypatch.setattr(
        epp,
        "generate_fragment_edit_plan",
        lambda **kwargs: (
            {
                "coverage_check": [
                    {"target_file": "pkg/mod.py", "action": "modify_region", "justification": "first"},
                    {"target_file": "pkg/mod.py", "action": "modify_region", "justification": "second"},
                ],
                "edits": [
                    {
                        "target_file": "pkg/mod.py",
                        "target_symbol": "Foo",
                        "anchor_line_before": "    x = 1",
                        "anchor_line_after": "    y = 2",
                        "replacement_block": "    x = 10\n    y = 2",
                    },
                    {
                        "target_file": "pkg/mod.py",
                        "target_symbol": "Foo",
                        "anchor_line_before": "    y = 2",
                        "anchor_line_after": "    z = 3",
                        "replacement_block": "    y = 20\n    z = 3",
                    },
                ],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "run_eval", fake_run_eval)
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)
    monkeypatch.setattr(epp, "build_patch_feedback", lambda *args, **kwargs: None)

    result = generate_patch_with_strategy(
        responder=responder,
        instance=instance,
        code_context=code_context,
        original_failure_log="AssertionError",
        filtered_candidates=[],
        max_candidate_attempts=2,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus=None,
    )

    assert responder.calls == 0
    assert recorded_patches
    assert "x = 10" in recorded_patches[0]
    assert "y = 20" in recorded_patches[0]
    assert "x = 10" in result.patch
    assert "y = 20" in result.patch


def test_generate_patch_with_strategy_locks_single_edit_plan_patch_before_free_diff(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            bad_patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,4 +1,4 @@
 class Foo:
-    value = 1
+    value = 999
 """
            return "raw bad patch", bad_patch

    responder = DummyResponder()
    instance = {
        "instance_id": "demo-instance",
        "repo": "demo/repo",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["f2p"]',
        "test_patch": "",
    }
    code_context = {
        "pkg/mod.py": """class Foo:
    value = 1
    other = 2
""",
    }
    baseline_eval = DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})
    recorded_patches = []

    def fake_run_eval(**kwargs):
        recorded_patches.append(kwargs["code_patch"])
        return DummyEval("clean_apply", {"f2p": "FAILED", "other": "PASSED"})

    monkeypatch.setattr(epp, "generate_patch_analysis", lambda **kwargs: ({}, 1))
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["pkg/mod.py::Foo"],
                "dependency_files": [],
                "sufficiency_assessment": "enough",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(
        epp,
        "build_edit_anchor_regions",
        lambda **kwargs: [
            {
                "target_file": "pkg/mod.py",
                "target_symbol": "Foo",
                "class_anchor_line": "class Foo:",
                "anchor_line_before": "    value = 1",
                "anchor_line_after": "    other = 2",
                "region_snippet": "class Foo:\n    value = 1\n    other = 2",
            },
        ],
    )
    monkeypatch.setattr(
        epp,
        "generate_fragment_edit_plan",
        lambda **kwargs: (
            {
                "coverage_check": [
                    {"target_file": "pkg/mod.py", "action": "modify_region", "justification": "single"},
                ],
                "edits": [
                    {
                        "target_file": "pkg/mod.py",
                        "target_symbol": "Foo",
                        "anchor_line_before": "    value = 1",
                        "anchor_line_after": "    other = 2",
                        "replacement_block": "    value = 10\n    other = 2",
                    },
                ],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "run_eval", fake_run_eval)
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)
    monkeypatch.setattr(epp, "build_patch_feedback", lambda *args, **kwargs: None)

    result = generate_patch_with_strategy(
        responder=responder,
        instance=instance,
        code_context=code_context,
        original_failure_log="AssertionError",
        filtered_candidates=[],
        max_candidate_attempts=2,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus=None,
    )

    assert responder.calls == 0
    assert recorded_patches
    assert "value = 10" in recorded_patches[0]
    assert "value = 10" in result.patch


def test_generate_patch_with_strategy_locks_single_edit_plan_patch_even_with_prefeedback(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            bad_patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,4 +1,4 @@
 class Foo:
-    value = 1
+    value = 999
 """
            return "raw bad patch", bad_patch

    responder = DummyResponder()
    instance = {
        "instance_id": "demo-instance",
        "repo": "demo/repo",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["f2p"]',
        "test_patch": "",
    }
    code_context = {
        "pkg/mod.py": """class Foo:
    value = 1
    other = 2
""",
    }
    baseline_eval = DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})
    recorded_patches = []

    def fake_run_eval(**kwargs):
        recorded_patches.append(kwargs["code_patch"])
        return DummyEval("clean_apply", {"f2p": "FAILED", "other": "PASSED"})

    monkeypatch.setattr(epp, "generate_patch_analysis", lambda **kwargs: ({}, 1))
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["pkg/mod.py::Foo"],
                "dependency_files": [],
                "sufficiency_assessment": "enough",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(
        epp,
        "build_edit_anchor_regions",
        lambda **kwargs: [
            {
                "target_file": "pkg/mod.py",
                "target_symbol": "Foo",
                "class_anchor_line": "class Foo:",
                "anchor_line_before": "    value = 1",
                "anchor_line_after": "    other = 2",
                "region_snippet": "class Foo:\n    value = 1\n    other = 2",
            },
        ],
    )
    monkeypatch.setattr(
        epp,
        "generate_fragment_edit_plan",
        lambda **kwargs: (
            {
                "coverage_check": [
                    {"target_file": "pkg/mod.py", "action": "modify_region", "justification": "single"},
                ],
                "edits": [
                    {
                        "target_file": "pkg/mod.py",
                        "target_symbol": "Foo",
                        "anchor_line_before": "    value = 1",
                        "anchor_line_after": "    other = 2",
                        "replacement_block": "    value = 10\n    other = 2",
                    },
                ],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "run_eval", fake_run_eval)
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)
    monkeypatch.setattr(epp, "build_patch_feedback", lambda *args, **kwargs: "warning only")

    result = generate_patch_with_strategy(
        responder=responder,
        instance=instance,
        code_context=code_context,
        original_failure_log="AssertionError",
        filtered_candidates=[],
        max_candidate_attempts=2,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus=None,
    )

    assert responder.calls == 0
    assert recorded_patches
    assert "value = 10" in recorded_patches[0]
    assert "value = 10" in result.patch


def test_generate_patch_with_strategy_allows_free_diff_only_when_plan_patch_cannot_apply_for_generic_topology(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            fallback_patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,3 +1,3 @@
 class Foo:
-    value = 1
+    value = 20
     other = 2
"""
            return "raw fallback patch", fallback_patch

    responder = DummyResponder()
    instance = {
        "instance_id": "demo-instance",
        "repo": "demo/repo",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["f2p"]',
        "test_patch": "",
    }
    code_context = {
        "pkg/mod.py": """class Foo:
    value = 1
    other = 2
""",
    }
    baseline_eval = DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})
    recorded_patches = []

    def fake_run_eval(**kwargs):
        recorded_patches.append(kwargs["code_patch"])
        if len(recorded_patches) == 1:
            return DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})
        return DummyEval("clean_apply", {"f2p": "FAILED", "other": "PASSED"})

    monkeypatch.setattr(
        epp,
        "generate_patch_analysis",
        lambda **kwargs: (
            {
                "root_cause": "generic bug",
                "affected_components": [],
                "failing_signal": "AssertionError",
                "propagation_path": "generic path",
                "repair_constraint": "keep behavior",
                "suggested_repair_scope": ["pkg/mod.py"],
                "suspicious_symbols": ["Foo"],
                "repair_obligations": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["pkg/mod.py::Foo"],
                "dependency_files": [],
                "sufficiency_assessment": "enough",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(
        epp,
        "build_edit_anchor_regions",
        lambda **kwargs: [
            {
                "target_file": "pkg/mod.py",
                "target_symbol": "Foo",
                "class_anchor_line": "class Foo:",
                "anchor_line_before": "    value = 1",
                "anchor_line_after": "    other = 2",
                "region_snippet": "class Foo:\n    value = 1\n    other = 2",
            },
        ],
    )
    monkeypatch.setattr(
        epp,
        "generate_fragment_edit_plan",
        lambda **kwargs: (
            {
                "coverage_check": [
                    {"target_file": "pkg/mod.py", "action": "modify_region", "justification": "single"},
                ],
                "edits": [
                    {
                        "target_file": "pkg/mod.py",
                        "target_symbol": "Foo",
                        "anchor_line_before": "    value = 1",
                        "anchor_line_after": "    other = 2",
                        "replacement_block": "    value = 10\n    other = 2",
                    },
                ],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "run_eval", fake_run_eval)
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)
    monkeypatch.setattr(epp, "build_patch_feedback", lambda *args, **kwargs: None)

    result = generate_patch_with_strategy(
        responder=responder,
        instance=instance,
        code_context=code_context,
        original_failure_log="AssertionError",
        filtered_candidates=[],
        max_candidate_attempts=2,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus=None,
    )

    assert responder.calls == 1
    assert len(recorded_patches) == 2
    assert "value = 10" in recorded_patches[0]
    assert "value = 20" in recorded_patches[1]
    assert "value = 20" in result.patch


def test_generate_patch_with_strategy_does_not_fallback_to_free_diff_for_statement_local(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            fallback_patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,3 +1,3 @@
 class Foo:
-    value = 1
+    value = 20
     other = 2
"""
            return "raw fallback patch", fallback_patch

    responder = DummyResponder()
    instance = {
        "instance_id": "demo-instance",
        "repo": "demo/repo",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["f2p"]',
        "test_patch": "",
    }
    code_context = {
        "pkg/mod.py": """class Foo:
    value = 1
    other = 2
""",
    }
    baseline_eval = DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})
    recorded_patches = []

    def fake_run_eval(**kwargs):
        recorded_patches.append(kwargs["code_patch"])
        return DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})

    monkeypatch.setattr(
        epp,
        "generate_patch_analysis",
        lambda **kwargs: (
            {
                "root_cause": "single statement conversion bug",
                "affected_components": [{"file": "pkg/mod.py", "symbol": "Foo", "reason": "localized"}],
                "failing_signal": "TypeError",
                "propagation_path": "single path",
                "repair_constraint": "keep behavior",
                "suggested_repair_scope": ["pkg/mod.py"],
                "suspicious_symbols": ["Foo"],
                "repair_obligations": [
                    {
                        "id": "f2p::core::value",
                        "source_symbol": "Foo",
                        "validation_subject": "value",
                        "why_it_matters": "localized statement fix",
                    }
                ],
            },
            1,
        ),
    )
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["pkg/mod.py::Foo"],
                "dependency_files": [],
                "sufficiency_assessment": "enough",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(
        epp,
        "build_edit_anchor_regions",
        lambda **kwargs: [
            {
                "target_file": "pkg/mod.py",
                "target_symbol": "Foo",
                "class_anchor_line": "class Foo:",
                "anchor_line_before": "    value = 1",
                "anchor_line_after": "    other = 2",
                "region_snippet": "class Foo:\n    value = 1\n    other = 2",
            },
        ],
    )
    monkeypatch.setattr(
        epp,
        "generate_fragment_edit_plan",
        lambda **kwargs: (
            {
                "coverage_check": [
                    {"target_file": "pkg/mod.py", "action": "modify_region", "justification": "single"},
                ],
                "edits": [
                    {
                        "target_file": "pkg/mod.py",
                        "target_symbol": "Foo",
                        "anchor_line_before": "    value = 1",
                        "anchor_line_after": "    other = 2",
                        "replacement_block": "    value = 10\n    other = 2",
                        "covers_obligations": ["f2p::core::value"],
                    },
                ],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "run_eval", fake_run_eval)
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)
    monkeypatch.setattr(epp, "build_patch_feedback", lambda *args, **kwargs: None)

    result = generate_patch_with_strategy(
        responder=responder,
        instance=instance,
        code_context=code_context,
        original_failure_log="TypeError",
        filtered_candidates=[
            epp.CandidatePatch(
                idea={
                    "target_source_symbol": "Foo",
                    "target_validation_subject": "value",
                    "covers_obligations": ["f2p::core::value"],
                    "trigger_shape_tokens": ["value = 1"],
                },
                patch="",
                raw_response="",
                identifiers=[],
                covered_original_tests=["f2p"],
            )
        ],
        max_candidate_attempts=2,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus={"active_fail_to_pass_identifiers": ["f2p"]},
    )

    assert responder.calls == 0
    assert len(recorded_patches) == 1
    assert "value = 10" in recorded_patches[0]
    assert result.patch == ""


def test_generate_patch_with_strategy_does_not_fallback_to_free_diff_for_single_root_symbol(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            raise AssertionError("free diff should not be used for single_root_symbol")

    responder = DummyResponder()
    instance = {
        "instance_id": "demo-instance",
        "repo": "demo/repo",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["test_immutable"]',
        "test_patch": "",
    }
    code_context = {
        "sympy/core/_print_helpers.py": "class Printable:\n    \"\"\"doc\"\"\"\n    pass\n",
    }
    baseline_eval = DummyEval("no_patch", {"test_immutable": "FAILED"})
    recorded_patches = []

    def fake_run_eval(**kwargs):
        recorded_patches.append(kwargs["code_patch"])
        return DummyEval("no_patch", {"test_immutable": "FAILED"})

    monkeypatch.setattr(
        epp,
        "generate_patch_analysis",
        lambda **kwargs: (
            {
                "root_cause": "A parent mixin in the inheritance chain is missing __slots__ = (), so Symbol gains __dict__.",
                "affected_components": [{"file": "sympy/core/_print_helpers.py", "symbol": "Printable", "reason": "root cause"}],
                "failing_signal": "AssertionError",
                "propagation_path": "Printable breaks the slots chain for Symbol.",
                "repair_constraint": "no __dict__ on Symbol",
                "suggested_repair_scope": ["sympy/core/_print_helpers.py"],
                "suspicious_symbols": ["Symbol", "Printable"],
                "repair_obligations": [{"id": "test_immutable::core::__dict__", "source_symbol": "Symbol", "validation_subject": "__dict__"}],
            },
            1,
        ),
    )
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "add __slots__ = () to Printable",
                "edit_targets": ["sympy/core/_print_helpers.py::Printable"],
                "dependency_files": ["sympy/core/symbol.py"],
                "sufficiency_assessment": "Editing only Printable is sufficient.",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(
        epp,
        "build_edit_anchor_regions",
        lambda **kwargs: [
            {
                "target_file": "sympy/core/_print_helpers.py",
                "target_symbol": "Printable",
                "class_anchor_line": "class Printable:",
                "anchor_line_before": '    """doc"""',
                "anchor_line_after": "    pass",
                "region_snippet": 'class Printable:\n    """doc"""\n    pass',
            },
        ],
    )
    monkeypatch.setattr(
        epp,
        "generate_fragment_edit_plan",
        lambda **kwargs: (
            None,
            1,
        ),
    )
    monkeypatch.setattr(epp, "run_eval", fake_run_eval)
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)
    monkeypatch.setattr(epp, "build_patch_feedback", lambda *args, **kwargs: None)

    result = epp.generate_patch_with_strategy(
        responder=responder,
        instance=instance,
        code_context=code_context,
        original_failure_log="AssertionError",
        filtered_candidates=[],
        max_candidate_attempts=2,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus={"active_fail_to_pass_identifiers": ["test_immutable"]},
    )

    assert responder.calls == 0
    assert recorded_patches == []
    assert result.patch == ""


def test_filter_candidates_weak_keeps_import_error_environment_block(monkeypatch):
    instance = {
        "instance_id": "sphinx-doc__sphinx-7686",
        "repo": "sphinx-doc/sphinx",
        "test_patch": (
            "diff --git a/tests/test_ext_autosummary.py b/tests/test_ext_autosummary.py\n"
            "--- a/tests/test_ext_autosummary.py\n"
            "+++ b/tests/test_ext_autosummary.py\n"
        ),
    }
    candidate = epp.CandidatePatch(
        idea={
            "target_source_symbol": "sphinx.ext.autosummary.generate.generate_autosummary_content",
            "target_validation_subject": "members variable",
            "covers_original_tests": [
                "test_autosummary_generate_content_for_module",
                "test_autosummary_generate_content_for_module_skipped",
            ],
            "covers_obligations": [
                "test_autosummary_generate_content_for_module::core",
            ],
            "semantic_alignment_tokens": [
                "autosummary_imported_members",
                "generate_autosummary_content",
                "members",
            ],
        },
        patch=(
            "diff --git a/tests/test_ext_autosummary.py b/tests/test_ext_autosummary.py\n"
            "--- a/tests/test_ext_autosummary.py\n"
            "+++ b/tests/test_ext_autosummary.py\n"
            "@@ -1,0 +1,2 @@\n"
            "+def test_sweb_enhanced_imported_members_exclusion():\n"
            "+    assert True\n"
        ),
        raw_response="",
        identifiers=[],
    )
    failure_focus = {
        "failure_mode": "import_error",
        "dominant_errors": ["ModuleNotFoundError", "ImportError"],
        "target_test_tracebacks": {
            "test_autosummary_generate_content_for_module": (
                "ModuleNotFoundError: No module named 'roman'\n"
                "ImportError: Error importing plugin \"sphinx.testing.fixtures\": No module named 'roman'\n"
            )
        },
        "inactive_fail_to_pass_identifiers": [
            "test_autosummary_generate_content_for_module",
            "test_autosummary_generate_content_for_module_skipped",
        ],
        "original_test_identifiers": [
            "test_autosummary_generate_content_for_module",
            "test_autosummary_generate_content_for_module_skipped",
        ],
    }
    code_context = {
        "sphinx/ext/autosummary/generate.py": (
            "def generate_autosummary_content(name, obj, parent, template, template_name,\n"
            "                                 imported_members, app, recursive, context):\n"
            "    pass\n"
        ),
    }

    def fake_run_eval(**kwargs):
        return epp.EvalResult(
            resolved=False,
            status_map={},
            log_text="ImportError: Error importing plugin \"sphinx.testing.fixtures\": No module named 'roman'",
            log_path="",
            report=None,
            patch_applied=True,
            patch_apply_mode="clean_apply",
            timed_out=False,
            error=None,
        )

    monkeypatch.setattr(epp, "run_eval", fake_run_eval)

    kept = epp.filter_candidates(
        instance,
        [candidate],
        timeout=1,
        keep_top_k=1,
        failure_focus=failure_focus,
        code_context=code_context,
    )

    assert len(kept) == 1
    assert kept[0].kept is True
    assert "import-error weak keep" in kept[0].reason


def test_generate_patch_with_strategy_does_not_fallback_to_free_diff_for_canonical_statement_signature(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            raise AssertionError("free diff should not be used when canonical statement signature is present")

    responder = DummyResponder()
    instance = {
        "instance_id": "demo-instance",
        "repo": "demo/repo",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["tests/test_requests.py::test_encoded_methods"]',
        "test_patch": "",
    }
    code_context = {
        "pkg/mod.py": """class Foo:
    def request(self, method=None):
        method = builtin_str(method)
        return method
""",
    }
    baseline_eval = DummyEval("no_patch", {"tests/test_requests.py::test_encoded_methods": "FAILED"})
    recorded_patches = []

    monkeypatch.setattr(
        epp,
        "generate_patch_analysis",
        lambda **kwargs: (
            {
                "root_cause": "single canonical statement conversion bug",
                "affected_components": [{"file": "pkg/mod.py", "symbol": "Foo.request", "reason": "localized"}],
                "failing_signal": "TypeError",
                "propagation_path": "single path",
                "repair_constraint": "keep behavior",
                "suggested_repair_scope": ["pkg/mod.py"],
                "suspicious_symbols": ["Foo.request"],
                "repair_obligations": [
                    {
                        "id": "encoded::method",
                        "source_symbol": "pkg.mod.Foo.request",
                        "validation_subject": "method",
                        "why_it_matters": "replace canonical conversion statement",
                    }
                ],
            },
            1,
        ),
    )
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["pkg/mod.py::Foo.request"],
                "dependency_files": [],
                "sufficiency_assessment": "enough",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(
        epp,
        "extract_required_repair_obligations",
        lambda *args, **kwargs: [
            {
                "id": "encoded::method",
                "source_symbol": "pkg.mod.Foo.request",
                "validation_subject": "method",
                "covered_original_tests": ["tests/test_requests.py::test_encoded_methods"],
                "is_active_fail_to_pass": True,
                "canonical_statement_required": True,
                "canonical_statement_text": "builtin_str(method)",
                "statement_anchor_tokens": ["builtin_str(method)"],
                "obligation_level": "primary_direct",
                "evidence_strength": "weak_evidence",
            }
        ],
    )
    monkeypatch.setattr(epp, "classify_repair_topology", lambda *args, **kwargs: "generic")
    monkeypatch.setattr(
        epp,
        "build_edit_anchor_regions",
        lambda **kwargs: [
            {
                "target_file": "pkg/mod.py",
                "target_symbol": "Foo.request",
                "class_anchor_line": "class Foo:",
                "anchor_line_before": "        method = builtin_str(method)",
                "anchor_line_after": "        return method",
                "region_snippet": "class Foo:\n    def request(self, method=None):\n        method = builtin_str(method)\n        return method",
            },
        ],
    )
    monkeypatch.setattr(
        epp,
        "generate_fragment_edit_plan",
        lambda **kwargs: (
            {
                "coverage_check": [
                    {"target_file": "pkg/mod.py", "action": "modify_region", "justification": "single"},
                ],
                "edits": [
                    {
                        "target_file": "pkg/mod.py",
                        "target_symbol": "Foo.request",
                        "anchor_line_before": "        method = builtin_str(method)",
                        "anchor_line_after": "        return method",
                        "replacement_block": "        if isinstance(method, bytes):\n            method = method.decode('utf-8')\n        else:\n            method = str(method)\n        return method",
                        "covers_obligations": ["encoded::method"],
                    },
                ],
            },
            1,
        ),
    )
    monkeypatch.setattr(
        epp,
        "run_eval",
        lambda **kwargs: recorded_patches.append(kwargs["code_patch"]) or DummyEval(
            "no_patch", {"tests/test_requests.py::test_encoded_methods": "FAILED"}
        ),
    )
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)
    monkeypatch.setattr(epp, "build_patch_feedback", lambda *args, **kwargs: "warning only")

    result = generate_patch_with_strategy(
        responder=responder,
        instance=instance,
        code_context=code_context,
        original_failure_log="TypeError",
        filtered_candidates=[],
        max_candidate_attempts=2,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus={"active_fail_to_pass_identifiers": ["tests/test_requests.py::test_encoded_methods"]},
    )

    assert responder.calls == 0
    assert recorded_patches
    assert "decode('utf-8')" in recorded_patches[0]
    assert result.patch == ""


def test_generate_patch_with_strategy_locks_multi_edit_plan_patch_even_with_prefeedback(monkeypatch):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text=""):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text

    class DummyResponder:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            bad_patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,4 +1,4 @@
 class Foo:
     x = 1
-    y = 2
+    y = 999
 """
            return "raw bad patch", bad_patch

    responder = DummyResponder()
    instance = {
        "instance_id": "demo-instance",
        "repo": "demo/repo",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["f2p"]',
        "test_patch": "",
    }
    code_context = {
        "pkg/mod.py": """class Foo:
    x = 1
    y = 2
    z = 3
""",
    }
    baseline_eval = DummyEval("no_patch", {"f2p": "FAILED", "other": "PASSED"})
    recorded_patches = []

    def fake_run_eval(**kwargs):
        recorded_patches.append(kwargs["code_patch"])
        return DummyEval("clean_apply", {"f2p": "FAILED", "other": "PASSED"})

    monkeypatch.setattr(epp, "generate_patch_analysis", lambda **kwargs: ({}, 1))
    monkeypatch.setattr(
        epp,
        "generate_patch_strategy",
        lambda **kwargs: (
            {
                "title": "t",
                "approach": "a",
                "edit_targets": ["pkg/mod.py::Foo"],
                "dependency_files": [],
                "sufficiency_assessment": "enough",
                "risks": [],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "augment_code_context_with_targets", lambda **kwargs: kwargs["code_context"])
    monkeypatch.setattr(
        epp,
        "build_edit_anchor_regions",
        lambda **kwargs: [
            {
                "target_file": "pkg/mod.py",
                "target_symbol": "Foo",
                "class_anchor_line": "class Foo:",
                "anchor_line_before": "    x = 1",
                "anchor_line_after": "    y = 2",
                "region_snippet": "class Foo:\n    x = 1\n    y = 2",
            },
            {
                "target_file": "pkg/mod.py",
                "target_symbol": "Foo",
                "class_anchor_line": "class Foo:",
                "anchor_line_before": "    y = 2",
                "anchor_line_after": "    z = 3",
                "region_snippet": "class Foo:\n    y = 2\n    z = 3",
            },
        ],
    )
    monkeypatch.setattr(
        epp,
        "generate_fragment_edit_plan",
        lambda **kwargs: (
            {
                "coverage_check": [
                    {"target_file": "pkg/mod.py", "action": "modify_region", "justification": "first"},
                    {"target_file": "pkg/mod.py", "action": "modify_region", "justification": "second"},
                ],
                "edits": [
                    {
                        "target_file": "pkg/mod.py",
                        "target_symbol": "Foo",
                        "anchor_line_before": "    x = 1",
                        "anchor_line_after": "    y = 2",
                        "replacement_block": "    x = 10\n    y = 2",
                    },
                    {
                        "target_file": "pkg/mod.py",
                        "target_symbol": "Foo",
                        "anchor_line_before": "    y = 2",
                        "anchor_line_after": "    z = 3",
                        "replacement_block": "    y = 20\n    z = 3",
                    },
                ],
            },
            1,
        ),
    )
    monkeypatch.setattr(epp, "run_eval", fake_run_eval)
    monkeypatch.setattr(epp, "run_semantic_oracle_checks", lambda **kwargs: None)
    monkeypatch.setattr(epp, "build_patch_feedback", lambda *args, **kwargs: "prefeedback warning")

    result = generate_patch_with_strategy(
        responder=responder,
        instance=instance,
        code_context=code_context,
        original_failure_log="AssertionError",
        filtered_candidates=[],
        max_candidate_attempts=2,
        baseline_eval=baseline_eval,
        timeout=1,
        max_chars_per_file=200,
        failure_focus=None,
    )

    assert responder.calls == 0
    assert recorded_patches
    assert "x = 10" in recorded_patches[0]
    assert "y = 20" in recorded_patches[0]
    assert "x = 10" in result.patch
    assert "y = 20" in result.patch


def test_run_instance_pipeline_uses_evaluated_patch_without_fallback_override(monkeypatch, tmp_path):
    class DummyResponder:
        model = "dummy-model"

    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text="", patch_applied=True, timed_out=False, error=None):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text
            self.patch_applied = patch_applied
            self.timed_out = timed_out
            self.error = error
            self.report = None

    instance = {
        "instance_id": "sympy__sympy-20590",
        "repo": "sympy/sympy",
        "problem_statement": "bug",
        "test_patch": "",
        "FAIL_TO_PASS": '["test_immutable"]',
    }
    code_context = {"sympy/core/_print_helpers.py": "class Printable:\n    pass\n"}
    bad_raw = """```diff
diff --git a/sympy/core/_print_helpers.py b/sympy/core/_print_helpers.py
--- a/sympy/core/_print_helpers.py
+++ b/sympy/core/_print_helpers.py
@@ -1,2 +1,3 @@
 class Printable:
+    __slots__ = ()
     pass
```"""
    good_patch = """diff --git a/sympy/core/_print_helpers.py b/sympy/core/_print_helpers.py
--- a/sympy/core/_print_helpers.py
+++ b/sympy/core/_print_helpers.py
@@ -1,2 +1,3 @@
 class Printable:
+    __slots__ = ()
     pass
"""
    patch_result = epp.PatchGenerationResult(
        analysis={},
        strategy={},
        edit_plan={},
        patch=good_patch,
        raw_response=bad_raw,
        attempts=1,
        accepted=False,
        acceptance_reason="rejected_no_metric_improvement",
        candidate_eval=DummyEval("clean_apply", {"test_immutable": "FAILED"}),
        semantic_oracle_passed=False,
        semantic_oracle_failed_identifiers=[],
        patch_error=None,
        final_feedback="feedback",
    )

    original_eval = DummyEval("no_patch", {"test_immutable": "FAILED"}, log_text="AssertionError")
    final_eval = DummyEval("clean_apply", {"test_immutable": "PASSED"})

    def fake_run_eval(**kwargs):
        if kwargs["code_patch"] == "":
            return original_eval
        return final_eval

    monkeypatch.setattr(epp, "build_context_file_list", lambda *args, **kwargs: [])
    monkeypatch.setattr(epp, "load_code_context", lambda *args, **kwargs: code_context)
    monkeypatch.setattr(epp, "run_eval", fake_run_eval)
    monkeypatch.setattr(epp, "extract_failure_focus", lambda *args, **kwargs: None)
    monkeypatch.setattr(epp, "recommend_generation_budget", lambda *args, **kwargs: {
        "candidate_budget": 0,
        "keep_top_k": 0,
        "attempt_budget": 1,
        "bucket_limit": 0,
        "template_limit": 0,
        "failure_mode": "attribute_error",
        "noise_ratio": 0.0,
        "difficulty": "low",
    })
    monkeypatch.setattr(epp, "generate_enhanced_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(epp, "filter_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(epp, "generate_patch_with_strategy", lambda **kwargs: patch_result)

    summary = epp.run_instance_pipeline(
        instance=instance,
        responder=DummyResponder(),
        output_dir=tmp_path,
        max_context_files=1,
        max_chars_per_file=200,
        timeout=1,
        max_candidate_attempts=1,
        baseline_only=False,
    )

    assert summary["generated_patch_nonempty"] is True
    assert summary["final_patch_apply_mode"] == "clean_apply"


def test_hard_failure_signal_and_anchor_context():
    log = "Traceback...\nIndentationError: unexpected indent"
    assert has_hard_failure_signal(log) is True
    assert get_hard_failure_signal(log) == "indentationerror"
    anchor = build_anchor_context(
        {"sympy/core/symbol.py": "class Symbol(AtomicExpr):\n    __slots__ = ('name',)\n    def __new__(self):\n        pass"},
        symbols=["Symbol", "__slots__"],
        edit_targets=["sympy/core/symbol.py"],
    )
    assert "Anchor: sympy/core/symbol.py" in anchor
    assert "__slots__" in anchor


def test_run_instance_pipeline_aligns_generated_patch_acceptance_with_final_resolution(monkeypatch, tmp_path):
    class DummyEval:
        def __init__(self, patch_apply_mode, status_map, log_text="", resolved=False):
            self.patch_apply_mode = patch_apply_mode
            self.status_map = status_map
            self.log_text = log_text
            self.log_path = ""
            self.report = None
            self.patch_applied = patch_apply_mode in {"clean_apply", "fuzzy_apply"}
            self.timed_out = False
            self.error = None
            self.resolved = resolved

    class DummyResponder:
        model = "dummy-model"

    instance = {
        "instance_id": "pallets__flask-4045",
        "repo": "pallets/flask",
        "problem_statement": "bug",
        "FAIL_TO_PASS": '["test_dotted_name_not_allowed"]',
        "test_patch": "",
        "patch": "",
    }
    code_context = {"src/flask/blueprints.py": "class Blueprint:\n    pass\n"}
    patch_result = epp.PatchGenerationResult(
        analysis={},
        strategy={},
        edit_plan={},
        patch="diff --git a/src/flask/blueprints.py b/src/flask/blueprints.py\n",
        raw_response="raw",
        attempts=1,
        accepted=False,
        acceptance_reason="rejected_f2p_not_resolved:test_dotted_name_not_allowed",
        candidate_eval=DummyEval("clean_apply", {"test_dotted_name_not_allowed": "FAILED"}),
        semantic_oracle_passed=False,
        semantic_oracle_failed_identifiers=["test_sweb_enhanced_a"],
        patch_error=None,
        final_feedback="old failure feedback",
    )
    original_eval = DummyEval("no_patch", {"test_dotted_name_not_allowed": "FAILED"}, log_text="AssertionError", resolved=False)
    final_eval = DummyEval("clean_apply", {"test_dotted_name_not_allowed": "PASSED"}, resolved=True)

    def fake_run_eval(**kwargs):
        if kwargs["code_patch"] == "":
            return original_eval
        return final_eval

    monkeypatch.setattr(epp, "build_context_file_list", lambda *args, **kwargs: [])
    monkeypatch.setattr(epp, "load_code_context", lambda *args, **kwargs: code_context)
    monkeypatch.setattr(epp, "run_eval", fake_run_eval)
    monkeypatch.setattr(epp, "extract_failure_focus", lambda *args, **kwargs: None)
    monkeypatch.setattr(epp, "recommend_generation_budget", lambda *args, **kwargs: {
        "candidate_budget": 0,
        "keep_top_k": 0,
        "attempt_budget": 1,
        "bucket_limit": 0,
        "template_limit": 0,
        "failure_mode": "value_error",
        "noise_ratio": 0.0,
        "difficulty": "low",
    })
    monkeypatch.setattr(epp, "generate_enhanced_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(epp, "filter_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(epp, "generate_patch_with_strategy", lambda **kwargs: patch_result)

    summary = epp.run_instance_pipeline(
        instance=instance,
        responder=DummyResponder(),
        output_dir=tmp_path,
        max_context_files=1,
        max_chars_per_file=200,
        timeout=1,
        max_candidate_attempts=1,
        baseline_only=False,
    )

    assert summary["generated_patch_accepted"] is True
    assert summary["generated_patch_acceptance_reason"] == "accepted_final_validation"
    assert summary["generated_patch_feedback"] is None
    assert summary["resolved"] is True
    assert summary["final_resolved"] is True


def test_aggregate_enhanced_failures_and_run_summary():
    class DummyCandidate:
        def __init__(self, failing_identifiers):
            self.failing_identifiers = failing_identifiers

    assert aggregate_enhanced_failures(
        [
            DummyCandidate(["test_sweb_enhanced_a"]),
            DummyCandidate(["test_sweb_enhanced_a", "test_sweb_enhanced_b"]),
        ]
    ) == ["test_sweb_enhanced_a", "test_sweb_enhanced_b"]

    aggregate = summarize_run(
        [
            {
                "final_resolved": True,
                "final_patch_cleanly_applied": True,
                "kept_enhanced_candidates": 1,
                "enhanced_candidates_total": 2,
                "original_failed_count": 2,
                "final_failed_count": 0,
                "original_passed_count": 3,
                "final_passed_count": 5,
                "original_passed_count_improvement": 2,
                "original_failed_count_reduction": 2,
                "enhanced_failed_count": 1,
            },
            {
                "final_resolved": False,
                "final_patch_cleanly_applied": False,
                "kept_enhanced_candidates": 0,
                "enhanced_candidates_total": 2,
                "original_failed_count": 1,
                "final_failed_count": 1,
                "original_passed_count": 4,
                "final_passed_count": 4,
                "original_passed_count_improvement": 0,
                "original_failed_count_reduction": 0,
                "enhanced_failed_count": 0,
                "adaptive_num_candidates": 5,
                "adaptive_keep_top_k": 3,
                "repair_mode": "baseline_fallback",
            },
        ]
    )
    assert aggregate["instances_total"] == 2
    assert aggregate["instances_resolved"] == 1
    assert aggregate["clean_apply_ratio"] == 0.5
    assert aggregate["enhanced_candidate_retention_rate"] == 0.25
    assert aggregate["adaptive_num_candidates_avg"] == 2.5
    assert aggregate["baseline_fallback_instances"] == 1


def test_enhanced_identifier_helpers():
    identifiers = [
        "test_immutable",
        "test_sweb_enhanced_case_one",
        "test_sweb_enhanced_case_two",
    ]
    assert get_enhanced_test_identifiers(identifiers) == [
        "test_sweb_enhanced_case_one",
        "test_sweb_enhanced_case_two",
    ]
    assert find_duplicate_test_identifiers(
        identifiers,
        ["test_immutable", "test_other"],
    ) == ["test_immutable"]


def test_get_original_test_identifiers():
    instance = {
        "test_patch": """diff --git a/tests/test_demo.py b/tests/test_demo.py
--- a/tests/test_demo.py
+++ b/tests/test_demo.py
@@ -1,1 +1,4 @@
+def test_old_case():
+    assert True
+
+def test_sweb_enhanced_new_case():
+    assert True
""",
    }
    assert get_original_test_identifiers(instance) == [
        "test_old_case",
        "test_sweb_enhanced_new_case",
    ]


def test_build_context_file_list_prioritizes_patch_files():
    instance = {
        "patch": """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1 +1 @@
-a
+b
""",
        "test_patch": """diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1 +1 @@
-a
+b
""",
    }
    assert build_context_file_list(instance, max_files=5) == [
        "pkg/core.py",
        "tests/test_core.py",
    ]


def test_get_strategy_edit_target_files_filters_non_paths():
    strategy = {
        "edit_targets": [
            "sympy/core/symbol.py",
            "Symbol",
            "__slots__",
            "sympy/core/basic.py",
        ],
        "dependency_files": ["sympy/core/expr.py"],
    }
    assert get_strategy_edit_target_files(strategy) == [
        "sympy/core/symbol.py",
        "sympy/core/basic.py",
        "sympy/core/expr.py",
    ]


def test_prompt_builders_include_key_sections():
    instance = {
        "instance_id": "demo__repo-1",
        "repo": "demo/repo",
        "problem_statement": "Bug happens when foo is None.",
        "test_patch": """diff --git a/tests/test_demo.py b/tests/test_demo.py
--- a/tests/test_demo.py
+++ b/tests/test_demo.py
@@ -1,0 +1,2 @@
+def test_old_case():
+    assert True
""",
    }
    code_context = {"pkg/core.py": "def foo(x):\n    return x"}
    failure_log = "FAILED tests/test_demo.py::test_old_case"
    failure_focus = {"failure_mode": "attribute_error", "failure_snippets": ["FAILED test_old_case"], "dominant_errors": ["AssertionError"], "failing_tests_sample": ["tests/test_demo.py::test_old_case"], "noise_ratio": 0.0}
    enhanced_prompt = build_enhanced_test_prompt(instance, code_context, failure_log, failure_focus=failure_focus)
    idea_prompt = build_test_idea_prompt(
        instance,
        code_context,
        failure_log,
        failure_mode="attribute_error",
        template_names=["attribute_absence_check", "slots_visibility_check"],
        semantic_buckets=["direct_symptom", "behavioral_consequence"],
        failure_focus=failure_focus,
    )
    analysis_prompt = build_patch_analysis_prompt(instance, code_context, failure_log, [])
    strategy_prompt = build_patch_strategy_prompt(
        instance,
        code_context,
        failure_log,
        [],
        analysis={"root_cause": "bad attribute exposure"},
    )
    patch_prompt = build_patch_prompt(instance, code_context, failure_log, [])
    enhanced_guided_patch_prompt = build_patch_prompt(
        instance,
        code_context,
        failure_log,
        [
            type(
                "Candidate",
                (),
                {
                    "identifiers": ["test_sweb_enhanced_case"],
                    "patch": "diff --git a/tests/test_demo.py b/tests/test_demo.py\n",
                },
            )()
        ],
    )
    assert "Bug happens when foo is None." in enhanced_prompt
    assert "Original test failure log" in enhanced_prompt
    assert "Original test function names to avoid: test_old_case" in enhanced_prompt
    assert "different observation angle" in enhanced_prompt
    assert "Prefer one minimal new test function over parameterized or multi-scenario tests" in enhanced_prompt
    assert "Focused failure summary" in enhanced_prompt
    assert "Detected failure mode: attribute_error" in idea_prompt
    assert "Allowed templates: attribute_absence_check, slots_visibility_check" in idea_prompt
    assert "Allowed semantic buckets: direct_symptom, behavioral_consequence" in idea_prompt
    assert "Bucket definitions:" in idea_prompt
    assert "Failure mode constraint" in idea_prompt
    assert "Avoid parameterized, multi-case, or kitchen-sink ideas" in idea_prompt
    assert "Focused failure summary" in idea_prompt
    assert "oracle" in idea_prompt
    assert "must encode the intended correct behavior" in idea_prompt
    assert "TRACEBACK-TO-OBLIGATION RULES" in idea_prompt
    assert "EXCEPTION-GAP RULE" in idea_prompt
    assert "root_cause" in analysis_prompt
    assert "affected_components" in analysis_prompt
    assert "suggested_repair_scope" in analysis_prompt
    assert "Repair strategy" not in strategy_prompt
    assert "Failure analysis" in strategy_prompt
    assert "sufficiency_assessment" in strategy_prompt
    assert "dependency_files" in strategy_prompt
    assert "Patch placement constraint" in strategy_prompt
    assert "Retained enhanced test patches" in patch_prompt
    assert "Baseline-fallback repair mode" in patch_prompt
    assert "Enhanced-guided repair mode" in enhanced_guided_patch_prompt


def test_analysis_prompt_emphasizes_flag_short_circuit_control_flow_for_runxfail_skip_location():
    instance = {
        "instance_id": "pytest-dev__pytest-7432",
        "repo": "pytest-dev/pytest",
        "problem_statement": "Skip location is wrong under --runxfail.",
        "test_patch": "",
    }
    code_context = {"src/_pytest/skipping.py": "def pytest_runtest_makereport():\n    pass\n"}
    failure_log = (
        "Failed: nomatch: 'SKIPPED [1] test_sample.py:2: unconditional skip'\n"
        "SKIPPED [1] src/_pytest/skipping.py:238: unconditional skip\n"
        "--runxfail\n"
    )
    prompt = build_patch_analysis_prompt(instance, code_context, failure_log, [])
    assert "FLAG-SHORT-CIRCUIT CONTROL-FLOW RULE" in prompt
    assert "earlier `if`/`elif`/guard branch" in prompt
    assert "later correction block becomes unreachable" in prompt


def test_analysis_prompt_lists_required_repair_obligations():
    instance = {
        "instance_id": "pallets__flask-4045",
        "repo": "pallets/flask",
        "problem_statement": "Blueprint names and endpoint/view function names with dots should raise ValueError.",
        "test_patch": "",
    }
    code_context = {
        "src/flask/blueprints.py": "class Blueprint:\n    def __init__(self, name, import_name):\n        pass\n"
    }
    failure_log = "Failed: DID NOT RAISE <class 'ValueError'>\nAssertionError: Blueprint endpoints should not contain dots\n"
    candidates = [
        epp.CandidatePatch(
            idea={
                "covers_obligations": ["test_dotted_name_not_allowed::core"],
                "target_source_symbol": "Blueprint.__init__",
                "target_validation_subject": "name",
                "trigger_shape_tokens": ["Blueprint("],
            },
            patch="",
            raw_response="",
            identifiers=[],
        ),
        epp.CandidatePatch(
            idea={
                "covers_obligations": ["test_route_decorator_custom_endpoint_with_dots::endpoint"],
                "target_source_symbol": "Blueprint.add_url_rule",
                "target_validation_subject": "endpoint",
                "trigger_shape_tokens": ["endpoint="],
            },
            patch="",
            raw_response="",
            identifiers=[],
        ),
    ]
    prompt = build_patch_analysis_prompt(
        instance,
        code_context,
        failure_log,
        candidates,
    )
    assert "Structured required repair obligations" in prompt
    assert "repair_obligations" in prompt
    assert "test_route_decorator_custom_endpoint_with_dots::endpoint" in prompt


def test_analysis_prompt_includes_symbol_cluster_rule_for_multi_subject_symbol():
    instance = {
        "instance_id": "pallets__flask-4045",
        "repo": "pallets/flask",
        "problem_statement": "Blueprint dotted names should raise ValueError.",
        "test_patch": "",
    }
    code_context = {"src/flask/blueprints.py": "class Blueprint:\n    def add_url_rule(self):\n        pass\n"}
    failure_log = "AssertionError"
    candidates = [
        epp.CandidatePatch(
            idea={
                "covers_obligations": ["test_route_decorator_custom_endpoint_with_dots::endpoint"],
                "target_source_symbol": "Blueprint.add_url_rule",
                "target_validation_subject": "endpoint",
                "trigger_shape_tokens": ["endpoint="],
            },
            patch="",
            raw_response="",
            identifiers=[],
        ),
        epp.CandidatePatch(
            idea={
                "covers_obligations": ["test_route_decorator_custom_endpoint_with_dots::view_func_name"],
                "target_source_symbol": "Blueprint.add_url_rule",
                "target_validation_subject": "view_func_name",
                "trigger_shape_tokens": ["view_func", "__name__"],
            },
            patch="",
            raw_response="",
            identifiers=[],
        ),
    ]
    prompt = build_patch_analysis_prompt(instance, code_context, failure_log, candidates)
    assert "SYMBOL-CLUSTER REPAIR RULE" in prompt
    assert "Blueprint.add_url_rule: endpoint, view_func_name" in prompt


def test_patch_analysis_feedback_requires_all_required_repair_obligations():
    feedback = build_patch_analysis_feedback(
        {
            "root_cause": "Missing validation in blueprint setup.",
            "affected_components": [],
            "failing_signal": "ValueError is not raised.",
            "propagation_path": "Bad input reaches the blueprint code.",
            "repair_constraint": "Keep valid names working.",
            "suggested_repair_scope": ["src/flask/blueprints.py"],
            "suspicious_symbols": ["Blueprint.__init__"],
            "repair_obligations": [
                {
                    "id": "test_dotted_name_not_allowed::core",
                    "source_symbol": "Blueprint.__init__",
                    "validation_subject": "name",
                    "why_it_matters": "constructor validation",
                }
            ],
        },
        required_repair_obligations=[
            {
                "id": "test_dotted_name_not_allowed::core",
                "source_symbol": "Blueprint.__init__",
                "validation_subject": "name",
            },
            {
                "id": "test_route_decorator_custom_endpoint_with_dots::view_func_name",
                "source_symbol": "Blueprint.add_url_rule",
                "validation_subject": "view_func_name",
                "trigger_shape_tokens": ["view_func", "__name__"],
            },
        ],
    )
    assert feedback is not None
    assert "test_route_decorator_custom_endpoint_with_dots::view_func_name" in feedback


def test_patch_analysis_feedback_requires_all_validation_subjects_for_same_symbol():
    feedback = build_patch_analysis_feedback(
        {
            "root_cause": "Missing validation in blueprint setup.",
            "affected_components": [],
            "failing_signal": "ValueError is not raised.",
            "propagation_path": "Bad input reaches the blueprint code.",
            "repair_constraint": "Keep valid names working.",
            "suggested_repair_scope": ["src/flask/blueprints.py"],
            "suspicious_symbols": ["Blueprint.add_url_rule"],
            "repair_obligations": [
                {
                    "id": "test_route_decorator_custom_endpoint_with_dots::endpoint",
                    "source_symbol": "Blueprint.add_url_rule",
                    "validation_subject": "endpoint",
                    "why_it_matters": "endpoint validation",
                }
            ],
        },
        required_repair_obligations=[
            {
                "id": "test_route_decorator_custom_endpoint_with_dots::endpoint",
                "source_symbol": "Blueprint.add_url_rule",
                "validation_subject": "endpoint",
            },
            {
                "id": "test_route_decorator_custom_endpoint_with_dots::view_func_name",
                "source_symbol": "Blueprint.add_url_rule",
                "validation_subject": "view_func_name",
            },
        ],
    )
    assert feedback is not None
    assert "same strong primary symbol" in feedback
    assert "view_func_name" in feedback


def test_patch_analysis_feedback_rejects_core_compression_for_multi_subject_symbol():
    feedback = build_patch_analysis_feedback(
        {
            "root_cause": "Validation is too generic.",
            "affected_components": [],
            "failing_signal": "ValueError is not raised consistently.",
            "propagation_path": "Bad input reaches the blueprint code.",
            "repair_constraint": "Keep valid names working.",
            "suggested_repair_scope": ["src/flask/blueprints.py"],
            "suspicious_symbols": ["Blueprint.add_url_rule"],
            "repair_obligations": [
                {
                    "id": "test_route_decorator_custom_endpoint_with_dots::core",
                    "source_symbol": "Blueprint.add_url_rule",
                    "validation_subject": "core",
                    "why_it_matters": "generic validation",
                }
            ],
        },
        required_repair_obligations=[
            {
                "id": "test_route_decorator_custom_endpoint_with_dots::endpoint",
                "source_symbol": "Blueprint.add_url_rule",
                "validation_subject": "endpoint",
            },
            {
                "id": "test_route_decorator_custom_endpoint_with_dots::view_func_name",
                "source_symbol": "Blueprint.add_url_rule",
                "validation_subject": "view_func_name",
            },
        ],
    )
    assert feedback is not None
    assert "generic core obligation" in feedback
    assert "Blueprint.add_url_rule" in feedback


def test_strategy_prompt_requires_explicit_elif_to_if_strategy_for_flag_short_circuit_bug():
    instance = {
        "instance_id": "pytest-dev__pytest-7432",
        "repo": "pytest-dev/pytest",
        "problem_statement": "Skip location is wrong under --runxfail.",
        "test_patch": "",
    }
    code_context = {"src/_pytest/skipping.py": "def pytest_runtest_makereport():\n    pass\n"}
    failure_log = "Failed: nomatch: 'SKIPPED [1] test_sample.py:2: unconditional skip'\n--runxfail\n"
    prompt = build_patch_strategy_prompt(
        instance,
        code_context,
        failure_log,
        [],
        analysis={"root_cause": "an elif branch in pytest_runtest_makereport becomes unreachable under runxfail"},
    )
    assert "FLAG-SHORT-CIRCUIT STRATEGY RULE" in prompt
    assert "convert that existing `elif` into a standalone `if`" in prompt
    assert "smallest structural change" in prompt
    assert "restore_unreachable_existing_branch" in prompt


def test_strategy_prompt_does_not_emit_control_flow_warning_for_non_short_circuit_validation_bug():
    instance = {
        "instance_id": "pallets__flask-4045",
        "repo": "pallets/flask",
        "problem_statement": "Blueprint names and endpoints with dots should raise ValueError.",
        "test_patch": "",
    }
    code_context = {"src/flask/blueprints.py": "class Blueprint:\n    def __init__(self, name, import_name):\n        pass\n"}
    failure_log = "Failed: DID NOT RAISE <class 'ValueError'>\nAssertionError: Blueprint endpoints should not contain dots\n"
    prompt = build_patch_strategy_prompt(
        instance,
        code_context,
        failure_log,
        [],
        analysis={"root_cause": "missing validation for blueprint names and endpoints"},
    )
    assert "CONTROL-FLOW WARNING:" not in prompt
    assert "FLAG-SHORT-CIRCUIT STRATEGY RULE" not in prompt


def test_strategy_prompt_requires_listing_all_covered_validation_subjects():
    instance = {
        "instance_id": "pallets__flask-4045",
        "repo": "pallets/flask",
        "problem_statement": "Blueprint names and endpoints with dots should raise ValueError.",
        "test_patch": "",
    }
    code_context = {"src/flask/blueprints.py": "class Blueprint:\n    def add_url_rule(self, rule, endpoint=None, view_func=None):\n        pass\n"}
    failure_log = "AssertionError: Blueprint endpoints should not contain dots\n"
    candidates = [
        epp.CandidatePatch(
            idea={
                "target_source_symbol": "Blueprint.add_url_rule",
                "target_validation_subject": "endpoint",
                "trigger_shape_tokens": ["endpoint="],
            },
            patch="",
            raw_response="",
            identifiers=[],
        ),
        epp.CandidatePatch(
            idea={
                "target_source_symbol": "Blueprint.add_url_rule",
                "target_validation_subject": "view_func_name",
                "trigger_shape_tokens": ["view_func", "__name__"],
            },
            patch="",
            raw_response="",
            identifiers=[],
        ),
    ]
    prompt = build_patch_strategy_prompt(
        instance,
        code_context,
        failure_log,
        candidates,
        analysis={"root_cause": "missing explicit validation in Blueprint.add_url_rule"},
    )
    assert "VALIDATION-SUBJECT REPAIR OBLIGATION" in prompt
    assert "endpoint" in prompt
    assert "view_func_name" in prompt
    assert "both the approach and the sufficiency_assessment" in prompt.lower()


def test_single_fail_to_pass_prompts_bias_toward_minimal_near_path_variants():
    instance = {
        "instance_id": "pytest-dev__pytest-7432",
        "repo": "pytest-dev/pytest",
        "problem_statement": "Skip location is wrong under --runxfail.",
        "FAIL_TO_PASS": '["testing/test_skipping.py::test_xfail_run_with_skip_mark[test_input1-expected1]"]',
        "test_patch": "",
    }
    code_context = {"testing/test_skipping.py": "def test_old_case():\n    pass"}
    failure_log = "AssertionError\nFAILED testing/test_skipping.py::test_xfail_run_with_skip_mark[test_input1-expected1]"
    failure_focus = {
        "failure_mode": "type_error",
        "failure_snippets": ["FAILED test_xfail_run_with_skip_mark"],
        "dominant_errors": ["AssertionError"],
        "failing_tests_sample": ["testing/test_skipping.py::test_xfail_run_with_skip_mark[test_input1-expected1]"],
        "noise_ratio": 0.0,
    }

    enhanced_prompt = build_enhanced_test_prompt(
        instance,
        code_context,
        failure_log,
        failure_focus=failure_focus,
    )
    idea_prompt = build_test_idea_prompt(
        instance,
        code_context,
        failure_log,
        failure_mode="type_error",
        template_names=["boundary_type_input"],
        semantic_buckets=["direct_symptom"],
        failure_focus=failure_focus,
    )

    assert "there is only one original FAIL_TO_PASS test" in enhanced_prompt
    assert "Stay close to that exact execution path" in enhanced_prompt
    assert "Do not expand into multiple scenarios just to create diversity" in idea_prompt


def test_multi_path_enhanced_test_prompt_requires_coverage_of_each_original_path():
    instance = {
        "instance_id": "pallets__flask-4045",
        "repo": "pallets/flask",
        "problem_statement": "dotted blueprint names and endpoints should raise ValueError",
        "FAIL_TO_PASS": '["tests/test_blueprints.py::test_dotted_name_not_allowed","tests/test_blueprints.py::test_route_decorator_custom_endpoint_with_dots"]',
        "test_patch": "",
    }
    code_context = {"tests/test_blueprints.py": "def test_a():\n    pass\n"}
    failure_log = """_________________________ test_dotted_name_not_allowed _________________________
tests/test_blueprints.py:256: Failed

________________ test_route_decorator_custom_endpoint_with_dots ________________
src/flask/scaffold.py:433: in decorator
src/flask/blueprints.py:364: AssertionError
"""
    failure_focus = {
        "failure_mode": "value_error",
        "failure_snippets": ["DID NOT RAISE", "AssertionError"],
        "dominant_errors": ["ValueError", "AssertionError"],
        "failing_tests_sample": [
            "tests/test_blueprints.py::test_dotted_name_not_allowed",
            "tests/test_blueprints.py::test_route_decorator_custom_endpoint_with_dots",
        ],
        "noise_ratio": 0.0,
    }
    prompt = build_enhanced_test_prompt(
        instance,
        code_context,
        failure_log,
        failure_focus=failure_focus,
    )
    assert "Generate at least one enhanced test per distinct code path shown above" in prompt
    assert "retained enhanced set must include at least one reproduction for each original path" in prompt


def test_test_idea_prompt_highlights_still_uncovered_original_paths():
    instance = {
        "instance_id": "pallets__flask-4045",
        "repo": "pallets/flask",
        "problem_statement": "dotted blueprint names and endpoints should raise ValueError",
        "FAIL_TO_PASS": '["test_dotted_name_not_allowed","test_route_decorator_custom_endpoint_with_dots"]',
        "test_patch": "",
    }
    code_context = {"src/flask/blueprints.py": "class Blueprint:\n    def __init__(self, name, import_name):\n        pass\n"}
    failure_log = "Failed: DID NOT RAISE ValueError\nAssertionError: Blueprint endpoints should not contain dots"
    prompt = build_test_idea_prompt(
        instance,
        code_context,
        failure_log,
        failure_mode="value_error",
        template_names=["single_exception_assert"],
        semantic_buckets=["direct_symptom"],
        failure_focus={
            "failure_mode": "value_error",
            "failure_snippets": ["DID NOT RAISE", "AssertionError"],
            "dominant_errors": ["ValueError", "AssertionError"],
            "failing_tests_sample": [
                "test_dotted_name_not_allowed",
                "test_route_decorator_custom_endpoint_with_dots",
            ],
            "noise_ratio": 0.0,
        },
        uncovered_original_tests=["test_route_decorator_custom_endpoint_with_dots"],
    )
    assert "STILL UNCOVERED" in prompt
    assert "test_route_decorator_custom_endpoint_with_dots" in prompt
    assert "MUST cover at least one of these uncovered paths" in prompt


def test_test_idea_prompt_emphasizes_traceback_anchor_exception_gap_and_sibling_validation_rules():
    instance = {
        "instance_id": "pallets__flask-4045",
        "repo": "pallets/flask",
        "problem_statement": "dotted blueprint names and endpoints should raise ValueError",
        "FAIL_TO_PASS": '["test_dotted_name_not_allowed","test_route_decorator_custom_endpoint_with_dots"]',
        "test_patch": "",
    }
    code_context = {
        "src/flask/blueprints.py": """class Blueprint:
    def add_url_rule(self, rule, endpoint=None, view_func=None):
        if endpoint:
            assert "." not in endpoint
        if view_func and hasattr(view_func, "__name__"):
            assert "." not in view_func.__name__
""",
    }
    failure_focus = {
        "target_test_tracebacks": {
            "test_route_decorator_custom_endpoint_with_dots": """with pytest.raises(ValueError):
>   bp.route("/", endpoint="a.b")(lambda: "")
src/flask/scaffold.py:433: in decorator
    self.add_url_rule(rule, endpoint, f, **options)
src/flask/blueprints.py:364: AssertionError""",
        }
    }
    prompt = build_test_idea_prompt(
        instance,
        code_context,
        "Failed: DID NOT RAISE ValueError\nAssertionError: Blueprint endpoints should not contain dots",
        failure_mode="value_error",
        template_names=["single_exception_assert"],
        semantic_buckets=["direct_symptom"],
        failure_focus=failure_focus,
        uncovered_obligations=[
            "test_route_decorator_custom_endpoint_with_dots::endpoint",
            "test_route_decorator_custom_endpoint_with_dots::view_func_name",
        ],
    )
    assert "TRACEBACK-TO-OBLIGATION REFINEMENT RULES" in prompt
    assert "LAYER 1 — PATH LOCALIZATION" in prompt
    assert "LAYER 2 — OBLIGATION REFINEMENT" in prompt
    assert "LAYER 3 — IDEA GENERATION CONSTRAINTS" in prompt
    assert "TRACEBACK-ANCHOR RULE" in prompt
    assert "SIBLING VALIDATION RULE" in prompt
    assert "TRIGGER-SHAPE RULE" in prompt
    assert "IDEA GRANULARITY RULE" in prompt
    assert "UNFINISHED-PATH RULE" in prompt
    assert "REAL-TRIGGER RULE" in prompt
    assert "covers_obligations" in prompt
    assert "target_validation_subject" in prompt
    assert "trigger_shape_tokens" in prompt
    assert "STILL UNCOVERED repair obligations" in prompt
    assert "test_route_decorator_custom_endpoint_with_dots::endpoint" in prompt


def test_extract_test_output_blocks_prefers_real_pytest_block():
    content = """+ echo '>>>>> Start Test Output'
+ echo 'placeholder'
 echo '>>>>> End Test Output'
 echo '>>>>> Start Test Output'
 pytest -rA tests/test_demo.py
 FAILED tests/test_demo.py::test_sweb_enhanced_case
 echo '>>>>> End Test Output'
"""
    blocks = extract_test_output_blocks(content)
    assert len(blocks) == 2
    assert "FAILED tests/test_demo.py::test_sweb_enhanced_case" in blocks[-1]
