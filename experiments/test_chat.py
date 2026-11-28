import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from vlm_compression_agent import chat, controller


PLAN = {"action": "propose", "max_vram_gib": 4, "max_score_drop_pp": 2, "max_rounds": 5}


class ChatTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        workspace = stack.enter_context(tempfile.TemporaryDirectory())
        self.project = Path(workspace)
        self.output = self.project / "local-report"
        for name, quantization in ((controller.MODEL, {}), (f"{controller.MODEL}-AWQ",
                                                            {"quant_method": "awq", "bits": 4})):
            model = self.project / "models" / name
            model.mkdir(parents=True)
            (model / "config.json").write_text(json.dumps(dict(
                model_type="qwen2_5_vl", hidden_size=2048, num_hidden_layers=36,
                quantization_config=quantization)), encoding="utf-8")
            (model / "model.safetensors").touch()
        for task, split in (("pope", "test"), ("textvqa", "validation")):
            data = self.project / "datasets" / task
            data.mkdir(parents=True)
            (data / f"{split}.parquet").touch()
        stack.enter_context(patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}))
        self.doctor = stack.enter_context(patch.object(chat, "check_gpu", return_value={
            "status": "ready", "cuda_device": "NVIDIA test GPU", "cuda_memory": {"free_mib": 9700, "total_mib": 12288},
            "issues": [],
        }))
        self.complete = stack.enter_context(patch.object(chat.DeepSeekClient, "complete"))
        self.evaluator = stack.enter_context(patch.object(controller, "LocalEvaluator", return_value=self.measure))

    @staticmethod
    def measure(candidate, split, number):
        bf16 = candidate == controller.BF16
        return {task: {"score": 0.8, "median_ttft_ms": 430 if bf16 else 350,
                       "peak_allocated_mib": 7300 if bf16 else 3500,
                       "report": "private/round.json", "gpu": "private_gpu_name",
                       "unexpected": "private prediction"} for task in controller.TASKS}

    def cli(self, *inputs):
        argv = ["vlm-compression-agent", "chat", "--project-dir", str(self.project), "--output-dir", str(self.output),
                "--search-samples", "6", "--validation-samples", "6"]
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", argv), patch("builtins.input", side_effect=inputs), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                controller.main()
            except SystemExit as error:
                return error.code, stdout.getvalue() + stderr.getvalue()
        return 0, stdout.getvalue() + stderr.getvalue()

    def test_chat_confirms_and_runs_a_local_search_with_validated_constraints(self):
        replies = iter([{"action": "question", "message": "显存上限是多少？"}, PLAN, {"message": "通过复测"}])
        requests = []

        def complete(messages):
            requests.append(list(messages))
            return next(replies)

        self.complete.side_effect = complete
        code, text = self.cli("想省显存", "4 GiB，最多掉 2 个点，跑 5 轮", "运行")

        self.assertEqual(code, 0, text)
        self.assertIn("显存上限是多少？", text)
        self.assertIn("通过复测", text)
        report = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["goal"], {key: PLAN[key] for key in report["goal"]})
        self.assertEqual(report["search_samples_per_task"], 6)
        self.assertEqual(report["status"], "recommended")
        self.assertEqual(report["rounds_used"], 5)
        second_request = requests[1]
        self.assertEqual([message["role"] for message in second_request], ["system", "user", "assistant", "user"])
        self.assertIn("想省显存", str(second_request))
        summary_request = str(requests[-1])
        self.assertIn("median_ttft_ms", summary_request)
        for local_only in (str(self.project), "private", "test-secret", "想省显存"):
            self.assertNotIn(local_only, summary_request)

    def test_declining_plan_does_not_start_a_gpu_worker(self):
        self.complete.return_value = PLAN
        code, text = self.cli("帮我测 4 GiB", "quit")

        self.assertEqual(code, 0, text)
        self.evaluator.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_revising_the_plan_replaces_its_constraints(self):
        self.complete.side_effect = [PLAN, {**PLAN, "max_vram_gib": 5}, {"message": "通过复测"}]
        code, text = self.cli("帮我测 4 GiB", "显存放宽到 5 GiB", "run")

        self.assertEqual(code, 0, text)
        report = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["goal"]["max_vram_gib"], 5)

    def test_api_failure_keeps_the_conversation_but_clears_pending_confirmation(self):
        self.complete.side_effect = [PLAN, chat.DeepSeekError("API unavailable"),
                                     {"action": "question", "message": "请重述要修改的目标。"}]
        code, text = self.cli("帮我测 4 GiB", "把显存放宽到 5 GiB", "run", "quit")

        self.assertEqual(code, 0, text)
        self.assertIn("API unavailable", text)
        self.assertIn("请重述要修改的目标。", text)
        self.assertIn("帮我测 4 GiB", str(self.complete.call_args.args[0]))
        self.evaluator.assert_not_called()

    def test_preflight_prints_issues_without_contacting_api(self):
        self.doctor.return_value = {"status": "needs_setup", "issues": ["CUDA device is unavailable"]}
        code, text = self.cli()

        self.assertEqual(code, 1)
        self.assertIn("CUDA device", text)
        self.complete.assert_not_called()
        self.evaluator.assert_not_called()

    def test_chat_rejects_a_checkpoint_from_another_model_size(self):
        model = self.project / "models" / controller.MODEL / "config.json"
        model.write_text(json.dumps(dict(model_type="qwen2_5_vl", hidden_size=3584, num_hidden_layers=28)))
        code, text = self.cli()

        self.assertEqual(code, 1)
        self.assertIn("3B", text)
        self.complete.assert_not_called()
        self.evaluator.assert_not_called()

    def test_out_of_range_or_untyped_plans_do_not_run(self):
        for overrides in ({"max_rounds": 4}, {"max_rounds": True}, {"max_vram_gib": float("inf")},
                          {"max_score_drop_pp": "2"}, {"command": "untrusted"}):
            with self.subTest(overrides=overrides):
                self.complete.return_value = {**PLAN, **overrides}
                code, text = self.cli("生成计划")
                self.assertEqual(code, 1, text)
        self.evaluator.assert_not_called()

    def test_missing_api_key_is_reported_before_preflight(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "  "}):
            code, text = self.cli()

        self.assertEqual(code, 1)
        self.assertIn("DEEPSEEK_API_KEY", text)
        self.doctor.assert_not_called()

    def test_explanation_api_failure_preserves_results_and_success_status(self):
        self.complete.side_effect = [PLAN, chat.DeepSeekError("API unavailable")]
        code, text = self.cli("帮我测", "run")

        self.assertEqual(code, 0, text)
        self.assertIn("API unavailable", text)
        self.assertIn(str(self.output / "report.json"), text)
        self.assertIn("median_ttft_ms", text)
        self.assertTrue((self.output / "report.json").is_file())


class DeepSeekTests(unittest.TestCase):
    def test_flash_request_has_expected_json_contract(self):
        response = io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop",
                                                       "message": {"content": json.dumps(PLAN)}}]}).encode())
        with patch.object(chat, "urlopen", return_value=response) as open_url:
            self.assertEqual(chat.DeepSeekClient("test-secret").complete([{"role": "user", "content": "json"}]), PLAN)

        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_http_error_does_not_reveal_response_or_credential(self):
        error = HTTPError("https://api.deepseek.com/chat/completions", 401, "test-secret", {}, io.BytesIO(b"private"))
        with patch.object(chat, "urlopen", side_effect=error), self.assertRaises(chat.DeepSeekError) as caught:
            chat.DeepSeekClient("test-secret").complete([])

        self.assertIn("401", str(caught.exception))
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertNotIn("private", str(caught.exception))

    def test_truncated_or_empty_json_is_rejected(self):
        for finish, content in (("length", json.dumps(PLAN)), ("stop", "   "), ("stop", "[]")):
            response = io.BytesIO(json.dumps({"choices": [{"finish_reason": finish,
                                                           "message": {"content": content}}]}).encode())
            with self.subTest(finish=finish, content=content), patch.object(chat, "urlopen", return_value=response), \
                    self.assertRaises(chat.DeepSeekError):
                chat.DeepSeekClient("test-secret").complete([])


class LocalPreflightTests(unittest.TestCase):
    def test_gpu_preflight_releases_gpu_context_before_search(self):
        from argparse import Namespace

        args = Namespace(bf16_model=Path("bf16 checkpoint"), awq_model=Path("awq checkpoint"),
                         datasets_dir=Path("data"), cuda_device=1, awq_backend="gemm_triton")
        result = {"status": "ready", "cuda_device": "NVIDIA test GPU", "cuda_memory": {"free_mib": 9700}}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}), \
                patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(result))) as run:
            self.assertEqual(chat.check_gpu(args), result)

        command = run.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "vlm_compression_agent", "doctor"])
        self.assertIn("bf16 checkpoint", command)
        self.assertEqual(command[command.index("--cuda-device") + 1], "1")
        self.assertEqual(run.call_args.kwargs["cwd"], Path(chat.__file__).resolve().parents[1])
        self.assertNotIn("DEEPSEEK_API_KEY", run.call_args.kwargs["env"])
