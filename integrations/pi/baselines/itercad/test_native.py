from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location("itercad_native_adapter", Path(__file__).with_name("native.py"))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def make_fixture(root, cid, pages, task="generation"):
    folder = root / cid
    folder.mkdir(parents=True)
    names = []
    for i, (size, color) in enumerate(pages):
        name = f"input{i}.png"
        Image.new("RGB", size, color).save(folder / name)
        names.append(name)
    (folder / "description.yaml").write_text("description: Reconstruct the part.\ninput_type: text+image\ntask_type: " + task + "\ninput_files:\n" + "".join(f"  - {name}\n" for name in names))
    return folder


def test_prepare_preserves_single_page_pixels_and_excludes_shape_gt(tmp_path):
    inputs = tmp_path / "inputs"
    make_fixture(inputs, "102", [((17, 23), "red")])
    out = tmp_path / "prepared"
    native.prepare(inputs, out, ["102"])
    sample = json.loads((out / "samples.jsonl").read_text())
    assert sample["gt_stl_normalized_path"] == ""
    assert sample["uid"] == "102"
    with Image.open(out / sample["gt_view_path"]) as image:
        assert image.size == (17, 23)
        assert image.getpixel((16, 22)) == (255, 0, 0)


def test_multiple_drawing_pages_keep_original_scale_and_all_pixels(tmp_path):
    inputs = tmp_path / "inputs"
    make_fixture(inputs, "145", [((10, 20), "red"), ((20, 10), "blue")])
    out = tmp_path / "prepared"
    native.prepare(inputs, out, ["145"])
    manifest = json.loads((out / "input_manifest.json").read_text())
    rec = manifest["fixtures"][0]
    assert rec["page_sizes"] == [[10, 20], [20, 10]]
    assert rec["rescaled"] is False
    with Image.open(out / "cases/145/reference.png") as image:
        for (width, height), (x, y), expected in zip(rec["page_sizes"], rec["page_offsets"], [(255, 0, 0), (0, 0, 255)]):
            assert image.getpixel((x + width - 1, y + height - 1)) == expected


def test_unavailable_inputs_need_explicit_skip_and_are_recorded(tmp_path):
    inputs = tmp_path / "inputs"
    make_fixture(inputs, "102", [((10, 10), "white")])
    broken = make_fixture(inputs, "101", [((10, 10), "white")])
    (broken / "input0.png").unlink()
    make_fixture(inputs, "201", [((10, 10), "white")], "editing")
    with pytest.raises(FileNotFoundError):
        native.prepare(inputs, tmp_path / "strict", all_generation=True)
    result = native.prepare(inputs, tmp_path / "available", all_generation=True, skip_unavailable=True)
    assert result["samples"] == 1
    assert result["excluded"][0]["fixture_id"] == "101"


def test_input_path_cannot_leave_fixture(tmp_path):
    inputs = tmp_path / "inputs"
    folder = make_fixture(inputs, "102", [((10, 10), "white")])
    (folder / "description.yaml").write_text("input_type: text+image\ninput_files: ['../private.png']\n")
    with pytest.raises(ValueError, match="Unsafe"):
        native.fixture_record(inputs, "102")


def test_pinned_official_source_hashes_verify_without_cadquery():
    root = Path(__file__).resolve().parents[4] / "external_baselines/itercad"
    # This test is a source-integrity check, not a live model or CAD dependency test.
    assert native.verify_upstream(root)["commit"] == "55998c4a4537573f67ca241dedc52ee2840652ca"


def test_pass_k_preserves_both_native_attempts_and_settings():
    seen = []
    def draw(sample, config):
        seen.append((sample, config))
        return {"status": "success", "attempt": len(seen)}
    module = SimpleNamespace(process_itercad_draw=draw, compare_results=lambda a, b: a)
    sample = {"uid": "146", "gt_stl_normalized_path": ""}
    result, index, attempts = native.run_attempts(module, sample, {"pass_k": 2, "max_turns": 5, "temp_dir": "/tmp/native"})
    assert index == 0
    assert result["attempt"] == 1
    assert len(attempts) == 2
    assert [c[1]["max_turns"] for c in seen] == [5, 5]
    assert [c[1]["temp_dir"] for c in seen] == ["/tmp/native/attempt0", "/tmp/native/attempt1"]


def complete_and_incomplete_attempts():
    successful = {"turn": 0, "status": "exec_success", "exec_success": True, "code": "r = model"}
    incomplete = {"status": "success", "final_cd": -1, "all_turns_data": [dict(successful),
        {"turn": 1, "status": "exec_error", "exec_success": False}]}
    complete = {"status": "success", "final_cd": -1, "all_turns_data": [dict(successful),
        {"turn": 1, "status": "done_by_model"}]}
    return incomplete, complete


def select_fixture(results, compare, sample=None):
    remaining = iter(results)
    seen = []
    def draw(sample, config):
        seen.append(config)
        return next(remaining)
    module = SimpleNamespace(process_itercad_draw=draw, compare_results=compare)
    output = native.run_attempts(module, sample or {"uid": "140", "gt_stl_normalized_path": ""},
        {"pass_k": 2, "max_turns": 5, "temp_dir": "/tmp/selection-test"})
    assert len(seen) == 2 and all(config["max_turns"] == 5 for config in seen)
    return output


def test_no_gt_complete_later_attempt_precedes_failed_last_turn_fallback():
    # Regression for v3: native outer status is success for both attempts, and
    # missing CD made the original comparator choose the failed-last-turn one.
    incomplete, complete = complete_and_incomplete_attempts()
    def should_not_compare_different_completion_classes(a, b):
        pytest.fail("Different completion classes must not collapse into a missing-CD tie")
    selected, index, attempts = select_fixture([incomplete, complete], should_not_compare_different_completion_classes)
    assert selected is complete and index == 1
    assert attempts == [incomplete, complete]
    assert incomplete["all_turns_data"][-1]["status"] == "exec_error"


@pytest.mark.parametrize("choose_later", [False, True])
def test_same_completion_class_still_uses_the_official_comparator(choose_later):
    _, a = complete_and_incomplete_attempts()
    _, b = complete_and_incomplete_attempts()
    calls = []
    def compare(left, right):
        calls.append((left, right))
        return right if choose_later else left
    selected, index, _ = select_fixture([a, b], compare)
    assert calls == [(a, b)]
    assert selected is (b if choose_later else a)
    assert index == int(choose_later)


def test_all_incomplete_attempts_remain_incomplete_and_keep_earlier_tie():
    a, _ = complete_and_incomplete_attempts()
    b, _ = complete_and_incomplete_attempts()
    selected, index, attempts = select_fixture([a, b], lambda left, right: left)
    assert selected is a and index == 0 and len(attempts) == 2
    assert not native.attempt_completed(selected)
    assert selected["all_turns_data"][-1]["status"] == "exec_error"


def test_GT_mode_preserves_the_original_comparator_without_new_priority():
    incomplete, complete = complete_and_incomplete_attempts()
    selected, index, _ = select_fixture([incomplete, complete], lambda left, right: left,
        {"uid": "140", "gt_stl_normalized_path": "/official-test-gt.stl"})
    assert selected is incomplete and index == 0


def test_DONE_without_preceding_successful_code_is_not_complete():
    assert not native.attempt_completed({"all_turns_data": [{"status": "done_by_model"}]})
    assert not native.attempt_completed({"all_turns_data": [
        {"status": "exec_error", "exec_success": False, "code": "r = broken"}, {"status": "done_by_model"}]})
    assert native.attempt_completed({"all_turns_data": [{"exec_success": True, "code": "r = model"}]})


def test_native_initial_request_preserves_fixture_text_once():
    class Client:
        def __init__(self, config):
            pass
        def chat(self, messages):
            return messages
    module = SimpleNamespace(GeneratorClient=Client)
    native.install_task_context(module)
    client = module.GeneratorClient({"cadgenbench_description": "Use millimetres."})
    messages = [{"role": "user", "content": [{"type": "text", "text": "Official initial request"}]}]
    client.chat(messages)
    client.chat(messages)
    text = messages[0]["content"][0]["text"]
    assert text.startswith("Official initial request")
    assert text.count("Use millimetres.") == 1


def captured_turn(tmp_path):
    obj = object()
    calls = []
    def execute(code, holder, temp, uid, turn, gt, points, need_render=False):
        calls.append(code)
        return {"exec_success": True, "status": "exec_success"}, obj
    module = SimpleNamespace(try_execute_and_eval=execute)
    def export(actual, path):
        assert actual is obj
        path.write_text("STEP of the actual executed object")
    native.install_step_capture(module, export)
    data, returned_obj = module.try_execute_and_eval("r = model()", None, str(tmp_path / "native-work"), "146", 0, "", 8192, need_render=True)
    assert returned_obj is obj
    assert calls == ["r = model()"]
    return dict(data, code="r = model()", turn=0), calls


def test_step_export_captures_actual_execution_without_reexecuting_code(tmp_path):
    turn, calls = captured_turn(tmp_path)
    result = native.export_candidate({"all_turns_data": [turn, {"status": "done_by_model"}]}, tmp_path)
    assert result["status"] == "completed"
    assert calls == ["r = model()"]
    assert native.digest(tmp_path / "output.step") == result["step_sha256"]
    assert (tmp_path / "solution.py").read_text() == "r = model()"


def test_prior_valid_code_does_not_make_failed_last_turn_complete(tmp_path):
    turn, _ = captured_turn(tmp_path)
    result = native.export_candidate({"all_turns_data": [turn, {"status": "exec_error", "exec_success": False}]}, tmp_path)
    assert result["status"] == "incomplete"
    assert result["has_step"] is True


def test_changed_captured_step_is_rejected(tmp_path):
    turn, _ = captured_turn(tmp_path)
    Path(turn["cadgenbench_step_path"]).write_text("different geometry")
    result = native.export_candidate({"all_turns_data": [turn]}, tmp_path)
    assert result["status"] == "failed"
    assert result["has_step"] is False


def test_responses_translation_preserves_full_history_and_image_bytes():
    url = "data:image/png;base64,c29tZWltYWdl"
    history = [{"role": "system", "content": "Official system"},
               {"role": "user", "content": [{"type": "text", "text": "Original request"},
                {"type": "image_url", "image_url": {"url": url, "detail": "high"}}]},
               {"role": "assistant", "content": "```python\nr = model\n```"},
               {"role": "user", "content": "Official feedback"}]
    converted = native.response_input(history)
    assert [m["role"] for m in converted] == ["system", "user", "assistant", "user"]
    assert converted[1]["content"][0] == {"type": "input_text", "text": "Original request"}
    assert converted[1]["content"][1] == {"type": "input_image", "image_url": url, "detail": "high"}
    assert converted[2]["content"] == history[2]["content"]
    assert history[1]["content"][1]["type"] == "image_url"
    with pytest.raises(ValueError, match="Unsupported native content"):
        native.response_input([{"role": "user", "content": [{"type": "audio"}]}])


def test_responses_usage_maps_cache_reasoning_and_unknown_values():
    result = native.response_usage(SimpleNamespace(input_tokens=100, output_tokens=40, total_tokens=140,
        input_tokens_details=SimpleNamespace(cached_tokens=30), output_tokens_details=SimpleNamespace(reasoning_tokens=12)))
    assert result == {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140,
                      "cached_input_tokens": 30, "reasoning_output_tokens": 12}
    assert native.response_usage(None)["input_tokens"] is None
    assert native.response_usage({"input_tokens": 1})["output_tokens"] is None


def responses_fixture(tmp_path, reply):
    requests = []
    def create(**request):
        requests.append(request)
        return reply
    class Client:
        def __init__(self, config):
            self.client = SimpleNamespace(responses=SimpleNamespace(create=create))
            self.model_name, self.max_tokens = "gpt-5.6-sol", 40960
    module = SimpleNamespace(GeneratorClient=Client)
    native.install_responses_transport(module)
    client = module.GeneratorClient({"temp_dir": str(tmp_path), "reasoning_effort": "low", "request_timeout_seconds": 600})
    return client, requests


def test_responses_transport_opt_in_and_completed_text_usage(tmp_path):
    response = {"id": "resp_mock", "status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": "<DONE>"}]}], "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}}
    client, requests = responses_fixture(tmp_path, response)
    result, error = client.chat([{"role": "user", "content": "Original native prompt"}])
    assert (result, error) == ("<DONE>", None)
    assert requests[0]["reasoning"] == {"effort": "low"}
    assert requests[0]["max_output_tokens"] == 40960
    assert requests[0]["timeout"] == 600
    assert requests[0]["store"] is False
    assert "temperature" not in requests[0]
    assert "chat_template_kwargs" not in requests[0]
    assert "previous_response_id" not in requests[0]
    record = json.loads((tmp_path / "api_usage.json").read_text())
    assert record["calls"][0]["usage"]["total_tokens"] == 25
    assert record["calls"][0]["response_id"] == "resp_mock"


def test_incomplete_responses_never_become_executable_final_code(tmp_path):
    client, _ = responses_fixture(tmp_path, {"status": "incomplete", "output_text": "```python\nr =", "incomplete_details": {"reason": "max_output_tokens"}, "usage": {"output_tokens": 40960}})
    result, error = client.chat([{"role": "user", "content": "Original native prompt"}])
    assert result is None
    assert error == "responses_incomplete: max_output_tokens"
    record = json.loads((tmp_path / "api_usage.json").read_text())
    assert record["calls"][0]["usage"]["output_tokens"] == 40960


@pytest.mark.parametrize("completed,expected", [(0, 2), (1, 0)])
def test_cli_run_reports_incomplete_attempts_as_failure(monkeypatch, capsys, completed, expected):
    monkeypatch.setattr(native.sys, "argv", ["native.py", "run", "--data-jsonl", "/unused/samples.jsonl", "--output-dir", "/unused/output"])
    monkeypatch.setattr(native, "run", lambda args: {"processed": 1, "completed": completed})
    assert native.main() == expected
    assert json.loads(capsys.readouterr().out)["completed"] == completed
