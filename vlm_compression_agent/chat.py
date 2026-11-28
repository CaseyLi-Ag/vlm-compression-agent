"""Translate a terminal conversation into a validated local evaluation goal."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .controller import BASELINE_FAILURES, MODEL, TASKS, Goal


SYSTEM_PROMPT = """你是 Qwen2.5-VL-3B 本地压缩评测的对话助手。使用用户的语言对话。
现有搜索在本机 NVIDIA CUDA 上评测 BF16、INT4 AWQ 和 VisionZip，目标是在满足显存和得分约束后
降低首 token 延迟（TTFT）。POPE 和 TextVQA 始终一起评测，同一个最大掉分值分别约束两项任务。
TextVQA 适合检查读图中文字的能力。区分目标机器上的候选显存上限与评测机器加载 BF16 所需的显存。
如果缺少关键约束、指标有歧义或约束冲突，先追问。如果用户授权用默认值或请你给出试跑方案，
可以采用当前默认值。max_vram_gib 为正数，max_score_drop_pp 使用百分点（0 到 100），
max_rounds 是至少为 5 的整数，包括独立复测所用轮次。原有目标不随意变更。
其他模型或超出当前评测能力的目标需要说明限制。不要要求用户提供密钥、文件路径或评测数据。
每次仅返回以下三种 json 对象之一，字段名保持原样。
{"action":"question","message":"部署显存上限是多少？"}
{"action":"propose","max_vram_gib":4,"max_score_drop_pp":2,"max_rounds":8}
{"action":"unsupported","message":"当前对话评测支持 Qwen2.5-VL-3B。"}"""

EXPLAIN_PROMPT = """用简体中文解读给出的本地评测摘要。说明是否通过独立复测、采用的配置及
测得的显存、首 token 延迟和各项分数变化。如果没有配置通过，根据记录说明限制或运行失败。
解释只依据给出的摘要，禁止补造指标或声称找到未经复测的推荐。返回 json，例如 {"message":"评测完成。"}。"""


class DeepSeekError(Exception):
    """A public-facing API or response validation error."""


class DeepSeekClient:
    def __init__(self, key):
        self.key = key

    def complete(self, messages):
        payload = {
            "model": "deepseek-flash",
            "messages": messages,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "max_tokens": 1024,
        }
        request = Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        request.add_unredirected_header("Authorization", f"Bearer {self.key}")
        try:
            with urlopen(request, timeout=45) as response:
                body = response.read(1024 * 1024)
        except HTTPError as error:
            error.close()
            raise DeepSeekError(f"DeepSeek 请求失败（HTTP {error.code}）。请检查 API Key、余额或限流。") from None
        except (OSError, HTTPException, ValueError):
            raise DeepSeekError("无法连接 DeepSeek。请检查网络和 API Key 后重试。") from None
        try:
            choice = json.loads(body)["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("Incomplete completion")
            result = json.loads(choice["message"]["content"])
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
        except (ValueError, KeyError, IndexError, TypeError):
            raise DeepSeekError("DeepSeek 返回的 JSON 不完整或无法解析，请重试。") from None
        return result


def message_text(reply):
    message = reply.get("message")
    if isinstance(message, str):
        clean = "".join(char for char in message if char.isprintable() or char == "\n").strip()
        if clean:
            return clean
    raise DeepSeekError("DeepSeek 没有返回可显示的消息，请重试。")


def proposed_goal(reply):
    try:
        vram, drop, rounds = (reply[field] for field in ("max_vram_gib", "max_score_drop_pp", "max_rounds"))
        if (set(reply) != {"action", "max_vram_gib", "max_score_drop_pp", "max_rounds"}
                or type(vram) not in (int, float) or type(drop) not in (int, float) or type(rounds) is not int):
            raise ValueError("Invalid goal fields")
        return Goal(vram, drop, rounds)
    except (ValueError, KeyError, OverflowError):
        raise DeepSeekError("DeepSeek 给出的评测计划无效。请明确显存上限、允许掉分和评测轮数后重试。") from None


def check_model_size(args):
    for model in (args.bf16_model, args.awq_model):
        try:
            config = json.loads((model / "config.json").read_text(encoding="utf-8"))
            text = config.get("text_config", config)
            structure = text.get("hidden_size"), text.get("num_hidden_layers")
        except (OSError, ValueError, AttributeError):
            raise DeepSeekError(f"无法读取 {model / 'config.json'}。请运行 doctor 检查本地模型。") from None
        if config.get("model_type") != "qwen2_5_vl" or structure != (2048, 36):
            raise DeepSeekError(f"chat 支持 {MODEL}，请检查模型目录 {model}。")


def check_gpu(args):
    # End CUDA preflight before the benchmark worker loads the BF16 model.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("DEEPSEEK_API_KEY", None)
    command = [sys.executable, "-m", "vlm_compression_agent", "doctor",
               "--bf16-model", str(args.bf16_model), "--awq-model", str(args.awq_model),
               "--datasets-dir", str(args.datasets_dir), "--cuda-device", str(args.cuda_device),
               "--awq-backend", args.awq_backend]
    try:
        result = subprocess.run(command, env=env, cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, encoding="utf-8", errors="replace")
        return json.loads(result.stdout)
    except (OSError, ValueError):
        raise DeepSeekError("本地预检未完成。请单独运行 doctor 查看诊断信息。") from None


def report_summary(report):
    """Select the aggregate metrics and decisions that leave the local machine."""
    result = {field: report[field] for field in ("goal", "status", "rounds_used", "stop_reason")}
    best = report["recommendation"]
    result["recommendation"] = None
    if best:
        metrics = {
            task: {field: best["metrics"][task][field] for field in ("score", "median_ttft_ms", "peak_allocated_mib")}
            for task in TASKS
        }
        result["recommendation"] = dict(candidate=best["candidate"], metrics=metrics,
                                        mean_ttft_ms=best["mean_ttft_ms"], score_delta_pp=best["score_delta_pp"])
    result["rounds"] = [
        {field: entry[field] for field in ("split", "candidate", "status", "violations") if field in entry}
        for entry in report["rounds"]
    ]
    return result


def read_goal(client, args, defaults):
    prompt = f"{SYSTEM_PROMPT}\n当前默认值 {json.dumps(asdict(defaults))}。"
    history = [{"role": "system", "content": prompt}]
    goal = None
    while True:
        label = "输入 run 或 运行开始评测，也可以继续修改需求" if goal else "描述你的压缩需求"
        try:
            text = input(f"{label}> ").strip()
        except EOFError:
            return None
        if text.casefold() in {"quit", "exit", "退出", "取消"}:
            return None
        if goal and text.casefold() in {"run", "运行"}:
            return goal
        if not text:
            continue
        goal = None
        user_message = {"role": "user", "content": text}
        try:
            reply = client.complete([*history, user_message])
        except DeepSeekError as error:
            print(f"{error} 可以重新输入这一轮需求。", file=sys.stderr)
            continue
        history.append(user_message)
        history.append({"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)})
        action = reply.get("action")
        if action in ("question", "unsupported"):
            print(message_text(reply))
        elif action == "propose":
            goal = proposed_goal(reply)
            print(f"评测计划 {MODEL} · 单卡 CUDA {args.cuda_device} · {args.policy} · 最多 {goal.max_rounds} 轮")
            print(f"每项任务峰值显存低于 {goal.max_vram_gib:g} GiB，相对 BF16 最多下降 "
                  f"{goal.max_score_drop_pp:g} 个百分点。在约束内优先降低 TTFT。")
            print(f"POPE 和 TextVQA 各用 {args.search_samples} 道搜索题、{args.validation_samples} 道独立复测题。")
        else:
            raise DeepSeekError("DeepSeek 没有返回有效的追问或计划，请重试。")


def run_chat(args, defaults, evaluate):
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        print("请先在当前终端设置 DEEPSEEK_API_KEY。", file=sys.stderr)
        return 1
    client = DeepSeekClient(key)
    try:
        check_model_size(args)
        info = check_gpu(args)
        if info["status"] != "ready":
            for issue in info["issues"]:
                print(issue, file=sys.stderr)
            return 1
        memory = info["cuda_memory"]
        print(f"本地 GPU {info['cuda_device']} · 可用显存 {memory['free_mib'] / 1024:.2f} GiB / "
              f"{memory['total_mib'] / 1024:.2f} GiB")
        print("评测会先加载 BF16 基线，原有 3B 实测峰值约 7.18 GiB，运行时还需要额外显存。")
        print("DeepSeek 接收需求文本和评测摘要。模型、评测数据和报告保存在本机。输入 quit 可退出。")
        goal = read_goal(client, args, defaults)
    except DeepSeekError as error:
        print(error, file=sys.stderr)
        return 1
    if goal is None:
        return 0

    report, path = evaluate(goal)
    summary = json.dumps(report_summary(report), ensure_ascii=False)
    print(f"本地报告 {path}", flush=True)
    print(summary, flush=True)
    messages = [{"role": "system", "content": EXPLAIN_PROMPT}, {"role": "user", "content": summary}]
    try:
        print(message_text(client.complete(messages)))
    except DeepSeekError as error:
        print(f"报告解读失败。{error}", file=sys.stderr)
    return int(report["status"] in BASELINE_FAILURES)
