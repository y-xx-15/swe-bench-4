from __future__ import annotations

import json
import re
import subprocess
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
OUTPUT_ROOTS = [
    PROJECT_ROOT / "outputs" / "enhanced_patch_pipeline",
    PROJECT_ROOT / "outputs" / "enhanced_patch_pipeline_enhanced",
    PROJECT_ROOT / "outputs" / "enhanced_patch_pipeline_baseline",
]
DEFAULT_INSTANCE_ID = "sympy__sympy-20590"
PREFERRED_RESULT_DIRS = {
    DEFAULT_INSTANCE_ID: PROJECT_ROOT
    / "outputs"
    / "enhanced_patch_pipeline"
    / "3"
    / "8"
    / DEFAULT_INSTANCE_ID
}


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def candidate_run_score(instance_dir: Path) -> tuple[int, int, int, int, float]:
    summary_path = instance_dir / "summary.json"
    summary = read_json(summary_path, {})
    resolved_bonus = 1 if summary.get("final_resolved") or summary.get("resolved") else 0
    normal_run_bonus = 0 if "ablation" in instance_dir.parts else 1
    kept_tests = int(summary.get("kept_enhanced_candidates") or 0)
    has_filtered_tests = 1 if read_json(instance_dir / "enhanced_candidates_filtered.json", []) else 0
    mtime = summary_path.stat().st_mtime if summary_path.exists() else 0.0
    return resolved_bonus, normal_run_bonus, has_filtered_tests, kept_tests, mtime


def find_instance_dir(instance_id: str) -> Path | None:
    preferred_dir = PREFERRED_RESULT_DIRS.get(instance_id)
    if preferred_dir and preferred_dir.exists():
        return preferred_dir

    candidates: list[Path] = []
    for output_root in OUTPUT_ROOTS:
        if output_root.exists():
            candidates.extend(path for path in output_root.rglob(instance_id) if path.is_dir())
    if not candidates:
        return None
    return max(candidates, key=candidate_run_score)


def first_source_context(code_context: dict, summary: dict) -> str:
    analysis = summary.get("patch_analysis") or {}
    source_components = [
        item
        for item in analysis.get("affected_components", [])
        if item.get("file") and "test" not in item.get("file", "").lower()
    ]
    for component in source_components:
        content = code_context.get(component.get("file"))
        symbol = component.get("symbol")
        if content and symbol:
            symbol_context = extract_symbol_context(content, symbol)
            if symbol_context:
                return symbol_context

    target_files = [item.get("file") for item in source_components]
    target_files.extend(
        path
        for path in analysis.get("suggested_repair_scope", [])
        if "test" not in path.lower()
    )
    target_files.extend(path for path in code_context if not path.startswith("__tail__"))

    for target_file in target_files:
        content = code_context.get(target_file)
        if content:
            return trim_context(content)
    return ""


def extract_symbol_context(content: str, symbol: str) -> str:
    pattern = re.compile(rf"^(class|def)\s+{re.escape(symbol)}\b.*$", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return ""
    start = match.start()
    next_block = re.search(r"^(class|def)\s+\w+\b", content[match.end() :], re.MULTILINE)
    end = match.end() + next_block.start() if next_block else len(content)
    return trim_context(content[start:end])


def trim_context(content: str, max_lines: int = 80) -> str:
    lines = content.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + ["..."])


def build_bug_description(summary: dict, failure_focus: dict) -> str:
    analysis = summary.get("patch_analysis") or {}
    parts = [
        analysis.get("root_cause"),
        analysis.get("failing_signal"),
        analysis.get("repair_constraint"),
    ]
    if not any(parts):
        failing_tests = failure_focus.get("failing_tests_sample") or []
        dominant_errors = failure_focus.get("dominant_errors") or []
        parts = [
            f"失败测试：{', '.join(failing_tests)}" if failing_tests else "",
            f"主要错误：{', '.join(dominant_errors)}" if dominant_errors else "",
        ]
    return "\n\n".join(part for part in parts if part)


def build_result(instance_id: str, bug_code: str = "", bug_description: str = "") -> dict:
    instance_dir = find_instance_dir(instance_id)
    if instance_dir is None:
        raise FileNotFoundError(f"未找到实例 {instance_id} 的运行结果")

    summary = read_json(instance_dir / "summary.json", {})
    code_context = read_json(instance_dir / "code_context.json", {})
    failure_focus = read_json(instance_dir / "failure_focus.json", {})
    enhanced_tests = read_json(instance_dir / "enhanced_candidates_filtered.json", [])
    rejected_tests = read_json(instance_dir / "enhanced_candidates_rejected.json", [])
    default_bug_code = first_source_context(code_context, summary)
    default_bug_description = build_bug_description(summary, failure_focus)

    return {
        "instance_id": instance_id,
        "source": str(instance_dir.relative_to(PROJECT_ROOT)),
        "bug_code": bug_code or default_bug_code,
        "bug_description": bug_description or default_bug_description,
        "summary": summary,
        "enhanced_tests": enhanced_tests,
        "rejected_tests": rejected_tests,
        "patch": read_text(instance_dir / "model_patch.diff"),
        "patch_response": read_text(instance_dir / "patch_response.txt"),
        "original_failure": read_json(instance_dir / "original_failure.json", {}),
        "failure_focus": failure_focus,
    }


def run_real_pipeline(instance_id: str) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "swebench.experiments.enhanced_patch_pipeline",
        "--instance_ids",
        instance_id,
        "--output_dir",
        str(PROJECT_ROOT / "outputs" / "enhanced_patch_pipeline_ui"),
    ]
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


class RepairRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_request_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw_body = self.rfile.read(length).decode("utf-8")
        return json.loads(raw_body or "{}")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/results":
            query = parse_qs(parsed.query)
            instance_id = query.get("instance_id", [DEFAULT_INSTANCE_ID])[0] or DEFAULT_INSTANCE_ID
            try:
                self.write_json(build_result(instance_id))
            except FileNotFoundError as exc:
                self.write_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/repair":
            self.write_json({"error": "未知接口"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self.read_request_json()
            instance_id = payload.get("instance_id") or DEFAULT_INSTANCE_ID
            if payload.get("run_real_pipeline"):
                process = run_real_pipeline(instance_id)
                self.write_json(
                    {
                        "instance_id": instance_id,
                        "pid": process.pid,
                        "message": "真实流水线已在后台启动，完成后可再次读取结果。",
                    },
                    HTTPStatus.ACCEPTED,
                )
                return
            self.write_json(
                build_result(
                    instance_id=instance_id,
                    bug_code=payload.get("bug_code", ""),
                    bug_description=payload.get("bug_description", ""),
                )
            )
        except FileNotFoundError as exc:
            self.write_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except json.JSONDecodeError:
            self.write_json({"error": "请求体不是合法 JSON"}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.write_json({"error": f"服务端处理失败：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    host = "127.0.0.1"
    port = 8001
    server = ThreadingHTTPServer((host, port), RepairRequestHandler)
    print(f"缺陷修复界面已启动：http://{host}:{port}")
    print("按 Ctrl+C 停止服务")
    server.serve_forever()


if __name__ == "__main__":
    main()
