#!/usr/bin/env python3
"""CADGenBench input/export adapter around the pinned official IterCAD-Draw loop.

The upstream executor, prompt, four-view renderer and turn loop are imported
unchanged. No ground-truth geometry enters the generation process.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3] if len(HERE.parents) > 3 else Path("/repo")
DEFAULT_UPSTREAM = Path("/opt/IterCAD") if Path("/opt/IterCAD").is_dir() else REPO_ROOT / "external_baselines/itercad"
NO_GT_SELECTION_POLICY = {
    "id": "terminal-completion-before-official-comparator-v1",
    "condition": "No GT shape/CD is available",
    "priority": "An attempt with successful executable code and a final exec_success or done_by_model turn precedes a last-successful-code fallback",
    "within_same_completion_class": "Unchanged official compare_results; its equal-score/missing-CD tie retains the earlier attempt",
    "extra_model_calls": 0,
    "GT_used": False,
    "export_completion_gate_relaxed": False,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_id(value):
    import re
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"Invalid fixture id: {value!r}")
    return value


def fixture_record(inputs, cid):
    import yaml
    folder = (Path(inputs) / safe_id(cid)).resolve()
    root = Path(inputs).resolve()
    if not folder.is_relative_to(root):
        raise ValueError("Fixture path escapes input root")
    data = yaml.safe_load((folder / "description.yaml").read_text())
    if data.get("task_type", "generation") != "generation" or data.get("input_type") != "text+image":
        return None
    paths = []
    names = data.get("input_files")
    if not isinstance(names, list) or not names:
        raise ValueError(f"Fixture {cid} has no declared drawing inputs")
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Unsafe input file in fixture {cid}")
        path = (folder / name).resolve()
        if not path.is_relative_to(folder):
            raise ValueError(f"Input escapes fixture {cid}")
        if not path.is_file():
            raise FileNotFoundError(f"Fixture {cid} declared input missing: {name}")
        paths.append(path)
    return {"id": cid, "description": str(data.get("description", "")), "paths": paths}


def prepare(inputs, destination, cases=None, all_generation=False, skip_unavailable=False):
    """Preserve each page at original pixels; combine multiple pages without scaling."""
    from PIL import Image, ImageDraw
    inputs, destination = Path(inputs), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    ids = sorted((p.name for p in inputs.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)) if all_generation else (cases or ["102", "103", "145", "146"])
    records, excluded = [], []
    for cid in ids:
        try:
            record = fixture_record(inputs, cid)
        except FileNotFoundError as error:
            if not skip_unavailable:
                raise
            excluded.append({"fixture_id": cid, "reason": str(error)})
            continue
        if record is not None:
            records.append(record)
    if not records:
        raise ValueError("No generation fixtures selected")
    destination.mkdir(parents=True)
    samples, provenance = [], []
    for record in records:
        cid = record["id"]
        folder = destination / "cases" / cid
        folder.mkdir(parents=True)
        pages = []
        for index, source in enumerate(record["paths"]):
            copied = folder / f"page-{index + 1}{source.suffix.lower()}"
            shutil.copy2(source, copied)
            with Image.open(source) as image:
                pages.append(image.convert("RGB"))
        if len(pages) == 1:
            # Convert losslessly to PNG so downstream MIME and pixels agree.
            pages[0].save(folder / "reference.png")
            offsets = [[0, 0]]
        else:
            header, gap = 28, 16
            width = max(page.width for page in pages)
            height = sum(page.height + header for page in pages) + gap * (len(pages) - 1)
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            offsets, y = [], 0
            for index, page in enumerate(pages):
                draw.text((4, y + 6), f"Reference drawing page {index + 1}", fill="black")
                y += header
                canvas.paste(page, (0, y))
                offsets.append([0, y])
                y += page.height + gap
            canvas.save(folder / "reference.png")
        samples.append({"uid": cid, "qid": cid, "_task_type": "IterCAD-Draw", "gt_view_path": f"cases/{cid}/reference.png", "gt_stl_normalized_path": "", "cadgenbench_description": record["description"]})
        provenance.append({"fixture_id": cid, "input_sha256": [digest(p) for p in record["paths"]], "reference_sha256": digest(folder / "reference.png"), "page_sizes": [[p.width, p.height] for p in pages], "page_offsets": offsets, "rescaled": False})
    (destination / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    write_json(destination / "input_manifest.json", {"task": "engineering-drawing-to-CAD", "fixture_ids": [r["id"] for r in records], "reference_only": True, "ground_truth_geometry_used": False, "multi_page_policy": "vertical contact sheet, no page resizing", "description_policy": "append fixture description to the native initial request once", "fixtures": provenance, "excluded": excluded})
    return {"samples": len(samples), "jsonl": str(destination / "samples.jsonl"), "excluded": excluded}


def verify_upstream(root):
    expected = read_json(HERE / "source.json")
    for name, sha in expected["eval_file_sha256"].items():
        path = Path(root) / "eval" / name
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f"Official IterCAD source missing or changed: {path}; run setup.sh")
    return expected


def load_upstream(root):
    verify_upstream(root)
    eval_dir = Path(root) / "eval"
    sys.path.insert(0, str(eval_dir))
    spec = importlib.util.spec_from_file_location("cadgenbench_official_itercad", eval_dir / "evalution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attempt_completed(result):
    """Operational counterpart of upstream's terminal-success criterion without CD."""
    turns = result.get("all_turns_data", [])
    if not turns or not any(turn.get("exec_success") and isinstance(turn.get("code"), str) for turn in turns):
        return False
    return bool(turns[-1].get("exec_success")) or turns[-1].get("status") == "done_by_model"


def attempt_selection_policy(sample):
    if sample.get("gt_stl_normalized_path") or sample.get("gt_stl_path"):
        return {"id": "official-comparator-with-GT", "summary": "Unchanged official compare_results"}
    return dict(NO_GT_SELECTION_POLICY)


def run_attempts(upstream, sample, config):
    """Retain all native attempts; preserve terminal success when GT CD is absent."""
    attempts, selected = [], None
    without_gt = not (sample.get("gt_stl_normalized_path") or sample.get("gt_stl_path"))
    for index in range(config["pass_k"]):
        attempt_config = dict(config, temp_dir=str(Path(config["temp_dir"]) / f"attempt{index}"))
        result = upstream.process_itercad_draw(sample, attempt_config)
        attempts.append(result)
        if selected is None:
            selected = result
        elif without_gt and attempt_completed(selected) != attempt_completed(result):
            if attempt_completed(result):
                selected = result
        elif upstream.compare_results(selected, result) is result:
            selected = result
    return selected, next(i for i, value in enumerate(attempts) if value is selected), attempts


def response_input(messages):
    """Translate native text/image history without changing its wording or images."""
    converted = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"Unsupported native message role: {role}")
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise ValueError("Native content must be text or text/image blocks")
        blocks = []
        for block in content:
            if block.get("type") == "text":
                blocks.append({"type": "input_text", "text": block["text"]})
            elif block.get("type") == "image_url":
                image = block["image_url"]
                if not isinstance(image, dict) or not isinstance(image.get("url"), str):
                    raise ValueError("Native image block needs image_url.url")
                blocks.append({"type": "input_image", "image_url": image["url"], "detail": image.get("detail", "auto")})
            else:
                raise ValueError(f"Unsupported native content type: {block.get('type')}")
        converted.append({"role": role, "content": blocks})
    return converted


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def response_usage(usage):
    """Keep reported tokens separate; absent provider fields remain null."""
    return {"input_tokens": field(usage, "input_tokens"), "output_tokens": field(usage, "output_tokens"),
            "total_tokens": field(usage, "total_tokens"),
            "cached_input_tokens": field(field(usage, "input_tokens_details"), "cached_tokens"),
            "reasoning_output_tokens": field(field(usage, "output_tokens_details"), "reasoning_tokens")}


def response_text(response):
    text = field(response, "output_text")
    if isinstance(text, str) and text.strip():
        return text
    pieces = []
    for item in field(response, "output", []) or []:
        if field(item, "type") == "message":
            for block in field(item, "content", []) or []:
                if field(block, "type") == "output_text":
                    pieces.append(field(block, "text", ""))
    return "".join(pieces)


def install_responses_transport(upstream):
    """Opt-in transport bridge; the native prompt, executor and loop stay intact."""
    original = upstream.GeneratorClient

    class ResponsesClient(original):
        def __init__(self, config):
            super().__init__(config)
            self.effort = config.get("reasoning_effort")
            self.api_temperature = config.get("api_temperature")
            self.request_timeout = config.get("request_timeout_seconds", 120)
            self.usage_path = Path(config["temp_dir"]) / "api_usage.json"
            self.calls = []

        def chat(self, messages, temperature=0.7, timeout=120):
            # Replay the same complete visible history as the official chat client.
            # Do not mix previous_response_id with the loop's full-history replay.
            request = {"model": self.model_name, "input": response_input(messages),
                       "max_output_tokens": self.max_tokens, "store": False,
                       "timeout": self.request_timeout}
            if self.effort:
                request["reasoning"] = {"effort": self.effort}
            if self.api_temperature is not None:
                request["temperature"] = self.api_temperature
            record = {"call": len(self.calls), "protocol": "responses", "model": self.model_name}
            try:
                response = self.client.responses.create(**request)
                status = field(response, "status")
                record.update(response_id=field(response, "id"), status=status,
                              usage=response_usage(field(response, "usage")))
                content = response_text(response)
                if status in {"incomplete", "failed", "cancelled"}:
                    reason = field(field(response, "incomplete_details"), "reason")
                    record["incomplete_reason"] = reason
                    return None, f"responses_{status}: {reason or 'no complete response'}"
                if not content.strip():
                    return None, "responses_empty_assistant_message"
                return content, None
            except Exception as error:
                # Avoid persisting request bodies, credentials or unredacted gateway errors.
                record.update(status="request_error", error_type=type(error).__name__)
                return None, f"{type(error).__name__}: Responses request failed"
            finally:
                self.calls.append(record)
                write_json(self.usage_path, {"protocol": "responses", "calls": self.calls})

    upstream.GeneratorClient = ResponsesClient


def install_task_context(upstream):
    """Keep the official loop while preserving CADGenBench's textual input."""
    original = upstream.GeneratorClient

    class DrawingTaskClient(original):
        def __init__(self, config):
            super().__init__(config)
            self.task_description = config.get("cadgenbench_description", "").strip()
            self.task_context_added = False

        def chat(self, messages, *args, **kwargs):
            if self.task_description and not self.task_context_added:
                for message in messages:
                    if message.get("role") == "user" and isinstance(message.get("content"), list):
                        for block in message["content"]:
                            if block.get("type") == "text":
                                block["text"] += "\nCADGenBench task description: " + self.task_description
                                self.task_context_added = True
                                break
                        if self.task_context_added:
                            break
            return super().chat(messages, *args, **kwargs)

    upstream.GeneratorClient = DrawingTaskClient


def install_step_capture(upstream, exporter):
    """Save the same object the official loop just executed, without rerunning code."""
    original = upstream.try_execute_and_eval

    def captured(code, holder, temp_dir, uid, turn, gt_stl_path, sample_points, need_render=False):
        data, obj = original(code, holder, temp_dir, uid, turn, gt_stl_path, sample_points, need_render=need_render)
        if data.get("exec_success") and obj is not None:
            folder = Path(temp_dir) / f"{safe_id(uid)}_turn{turn}_formal"
            folder.mkdir(parents=True, exist_ok=True)
            source, step = folder / "solution.py", folder / "output.step"
            source.write_text(code)
            try:
                exporter(obj, step)
                data = dict(data, cadgenbench_source_path=str(source), cadgenbench_step_path=str(step),
                            cadgenbench_source_sha256=digest(source), cadgenbench_step_sha256=digest(step))
            except Exception as error:
                data = dict(data, cadgenbench_export_error=f"{type(error).__name__}: {error}")
        return data, obj

    upstream.try_execute_and_eval = captured


def export_candidate(result, folder):
    """Export an actually executed candidate; do not claim earlier code is final."""
    folder = Path(folder)
    turns = result.get("all_turns_data", [])
    last = turns[-1] if turns else {}
    valid_turns = [t for t in turns if t.get("exec_success") and isinstance(t.get("code"), str)]
    if not valid_turns:
        return {"status": "failed", "reason": "No successful CAD execution", "has_step": False}
    chosen = valid_turns[-1]
    code = chosen["code"]
    saved_source = Path(chosen.get("cadgenbench_source_path", "/nonexistent"))
    saved_step = Path(chosen.get("cadgenbench_step_path", "/nonexistent"))
    if not saved_source.resolve().is_relative_to(folder.resolve()) or not saved_step.resolve().is_relative_to(folder.resolve()):
        return {"status": "failed", "reason": "Missing captured formal execution artifact", "has_step": False}
    if not saved_source.is_file() or not saved_step.is_file() or digest(saved_source) != chosen.get("cadgenbench_source_sha256") or digest(saved_step) != chosen.get("cadgenbench_step_sha256"):
        return {"status": "failed", "reason": "Captured source/STEP missing or changed", "has_step": False}
    if hashlib.sha256(code.encode()).hexdigest() != digest(saved_source):
        return {"status": "failed", "reason": "Captured source does not match executed code", "has_step": False}
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "solution.py"
    shutil.copy2(saved_source, source)
    shutil.copy2(saved_step, folder / "output.step")
    closed = bool(last.get("exec_success")) or last.get("status") == "done_by_model"
    return {"status": "completed" if closed else "incomplete", "has_step": True, "artifact_stage": "last_successfully_executed_code", "last_turn_status": last.get("status"), "exported_turn": chosen.get("turn"), "source_sha256": digest(source), "step_sha256": digest(folder / "output.step"), "native_cd_scored": False}


def run_one(sample, config, upstream_root, folder):
    upstream = load_upstream(upstream_root)
    import cadquery as cq
    if config.get("api_protocol") == "responses":
        install_responses_transport(upstream)
    install_task_context(upstream)
    install_step_capture(upstream, lambda obj, path: cq.exporters.export(obj, str(path), exportType="STEP"))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(config, temp_dir=str(folder / "native-work"), cadgenbench_description=sample.get("cadgenbench_description", ""))
    selected, index, attempts = run_attempts(upstream, sample, config)
    write_json(folder / "attempts.json", {"selected_attempt": index, "pass_k": len(attempts),
               "selection_policy": attempt_selection_policy(sample),
               "attempt_terminal_completion": [{"attempt": i, "completed": attempt_completed(item)} for i, item in enumerate(attempts)],
               "attempts": attempts})
    status = export_candidate(selected, folder)
    status.update(fixture_id=sample["uid"], selected_attempt=index, pass_k=len(attempts), model=config["gen_model_name"], protocol="official IterCAD-Draw", checkpoint_identity="user_supplied_endpoint", geometry_accuracy_scored=False)
    write_json(folder / "result.json", status)
    return status


def run(args):
    root = Path(args.upstream_root).resolve()
    source = verify_upstream(root)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not 1 <= args.workers <= 8 or min(args.max_turns, args.pass_k, args.max_output_tokens, args.request_timeout_seconds) < 1:
        raise ValueError("workers must be 1–8; turns, attempts, tokens and timeout must be positive")
    if args.api_temperature is not None and not 0 <= args.api_temperature <= 2:
        raise ValueError("api-temperature must be between 0 and 2")
    if args.api_temperature is not None and args.api_protocol != "responses":
        raise ValueError("api-temperature is only an opt-in Responses transport option; official chat keeps 0.7")
    data_path = Path(args.data_jsonl).resolve()
    samples = [json.loads(line) for line in data_path.open() if line.strip()]
    seen = set()
    for sample in samples:
        cid = safe_id(sample.get("uid"))
        if cid in seen:
            raise ValueError("Duplicate fixture id")
        seen.add(cid)
        if sample.get("gt_stl_normalized_path") or sample.get("gt_stl_path"):
            raise ValueError("Adaptation runner refuses GT geometry; use separate offline evaluation")
        reference = (data_path.parent / sample["gt_view_path"]).resolve()
        if not reference.is_file():
            raise FileNotFoundError(reference)
        sample["gt_view_path"] = str(reference)
        sample["gt_stl_normalized_path"] = ""
    model = args.model or os.environ.get("GEN_MODEL")
    api = args.api_url or os.environ.get("GENERATOR_API")
    if not model or not api:
        raise ValueError("Set GEN_MODEL and GENERATOR_API, or pass --model and --api-url")
    key = os.environ.get(args.api_key_env, "EMPTY")
    output.mkdir(parents=True)
    # No API key is written to the configuration artifact.
    extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {"chat_template_kwargs": {"enable_thinking": False}}
    config = {"generator_api": api, "generator_api_key": key, "gen_model_name": model, "max_turns": args.max_turns, "pass_k": args.pass_k, "sample_points": 8192, "extra_body": extra, "max_tokens": args.max_output_tokens,
              "api_protocol": args.api_protocol, "reasoning_effort": args.reasoning_effort, "api_temperature": args.api_temperature, "request_timeout_seconds": args.request_timeout_seconds}
    # The official native system prompt is imported in each isolated worker.
    module = load_upstream(root)
    config["system_prompt"] = module.SYSTEM_PROMPT
    write_json(output / "experiment_contract.json", {"method": "itercad-native-drawing-adaptation", "source": source, "model": model, "api_key_environment_variable": args.api_key_env, "api_protocol": args.api_protocol, "reasoning_effort": args.reasoning_effort, "temperature": args.api_temperature if args.api_protocol == "responses" else 0.7, "temperature_policy": "service default unless explicitly supplied" if args.api_protocol == "responses" else "official client default", "request_timeout_seconds": args.request_timeout_seconds if args.api_protocol == "responses" else 120, "max_output_tokens": args.max_output_tokens, "max_turns": args.max_turns, "pass_k": args.pass_k, "workers": args.workers, "fixture_ids": sorted(seen), "gt_in_generation_or_selection": False, "attempt_selection_policy": dict(NO_GT_SELECTION_POLICY), "requested_official_checkpoint": model == "KnowledgeXLab/IterCAD-RL", "checkpoint_bytes_verified": False, "renderer": "official TOP/FRONT/RIGHT/ISO, overall extents and radii"})
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, sample, config, str(root), str(output / "cases" / sample["uid"])): sample["uid"] for sample in samples}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"fixture_id": cid, "status": "error", "error": f"{type(error).__name__}: {error}", "has_step": False}
                write_json(output / "cases" / cid / "result.json", result)
            results.append(result)
            write_json(output / "summary.json", {"status": "running" if len(results) < len(samples) else "completed" if all(r["status"] == "completed" for r in results) else "partial", "total_cases": len(samples), "processed_cases": len(results), "geometry_accuracy_scored": False, "cases": sorted(results, key=lambda x: x["fixture_id"])})
    return {"output": str(output), "processed": len(results), "completed": sum(r["status"] == "completed" for r in results)}


def doctor(root, output_dir=None):
    module = load_upstream(root)
    import cadquery as cq
    from PIL import Image
    def check(folder):
        folder = Path(folder)
        path = folder / "views.png"
        obj = cq.Workplane("XY").box(20, 10, 3).faces(">Z").workplane().hole(4)
        if not obj.val().isValid():
            raise RuntimeError("Synthetic native CAD is invalid")
        if not module.render_views(obj, str(path)):
            raise RuntimeError("Official IterCAD four-view renderer failed")
        with Image.open(path) as image:
            size = list(image.size)
        cq.exporters.export(obj, str(folder / "output.step"), exportType="STEP")
        record = {"ok": True, "cadquery": cq.__version__, "official_four_view_render": True, "image_size": size, "model_calls": 0, "shape_valid": True, "volume": obj.val().Volume(), "render_sha256": digest(path), "step_sha256": digest(folder / "output.step")}
        # Exercise the unchanged official control flow with deterministic local
        # replies. This is an integration test, not a model evaluation.
        calls = []
        encoder = module.GeneratorClient.encode_image
        code = 'import cadquery as cq\nr = cq.Workplane("XY").box(20, 10, 3).faces(">Z").workplane().hole(4)'
        class ScriptedClient:
            encode_image = staticmethod(encoder)
            def __init__(self, config):
                self.system_prompt = config["system_prompt"]
                self.turn = 0
            def chat(self, messages):
                calls.append({"messages": len(messages), "turn": self.turn})
                self.turn += 1
                return (f"```python\n{code}\n```" if self.turn == 1 else "<DONE>"), None
        module.GeneratorClient = ScriptedClient
        install_task_context(module)
        install_step_capture(module, lambda shape, dest: cq.exporters.export(shape, str(dest), exportType="STEP"))
        mock_dir = folder / "mock-loop"
        selected, selected_index, attempts = run_attempts(module,
            {"uid": "synthetic-doctor", "gt_view_path": str(path), "gt_stl_normalized_path": ""},
            {"max_turns": 5, "pass_k": 2, "sample_points": 8192,
             "system_prompt": module.SYSTEM_PROMPT, "cadgenbench_description": "Synthetic diagnostic only.",
             "temp_dir": str(mock_dir / "native-work")})
        exported = export_candidate(selected, mock_dir)
        if exported["status"] != "completed" or len(calls) != 4 or len(attempts) != 2:
            raise RuntimeError(f"Official native loop integration test failed: {exported}")
        write_json(mock_dir / "attempts.json", {"selected_attempt": selected_index, "attempts": attempts})
        write_json(mock_dir / "result.json", exported)
        record.update(official_loop_mock={"ok": True, "scripted_replies": len(calls), "pass_k": 2,
                      "max_turns": 5, "actual_turns_per_attempt": [r["turns"] for r in attempts],
                      "selected_attempt": selected_index, "exported_status": exported["status"],
                      "step_sha256": exported["step_sha256"]})
        write_json(folder / "doctor.json", record)
        freeze = Path("/opt/runtime-requirements.freeze.txt")
        if freeze.is_file():
            shutil.copy2(freeze, folder / freeze.name)
        return record
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=False)
        return check(folder)
    with tempfile.TemporaryDirectory(prefix="itercad-native-doctor-") as temporary:
        return check(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Convert CADGenBench generation drawings into native Draw JSONL")
    p.add_argument("--data-inputs-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--cases", nargs="+")
    group.add_argument("--all-generation", action="store_true")
    p.add_argument("--skip-unavailable-inputs", action="store_true")
    p = sub.add_parser("run", help="Run the pinned official agent loop and export code/STEP")
    p.add_argument("--data-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    p.add_argument("--model")
    p.add_argument("--api-url")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--api-protocol", choices=["chat", "responses"], default="chat")
    p.add_argument("--api-temperature", type=float, help="Responses only; omitted by default for reasoning-model compatibility")
    p.add_argument("--request-timeout-seconds", type=int, default=120, help="Responses request timeout; chat retains official 120s")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    p.add_argument("--max-output-tokens", type=int, default=40960)
    p.add_argument("--max-turns", type=int, default=5)
    p.add_argument("--pass-k", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("doctor", help="Check official code, CadQuery and annotated renderer without a model")
    p.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    p.add_argument("--output-dir", type=Path)
    p = sub.add_parser("source-check", help="Verify pinned official files without importing CAD libraries")
    p.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.data_inputs_dir, args.output_dir, args.cases, args.all_generation, args.skip_unavailable_inputs)
    elif args.command == "doctor":
        result = doctor(args.upstream_root, args.output_dir)
    elif args.command == "source-check":
        result = verify_upstream(args.upstream_root)
    else:
        result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # The official loop may return normally after exhausting failed attempts.
    # Preserve its complete report, but make one-command/scheduler failure real.
    if args.command == "run" and result["completed"] != result["processed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
