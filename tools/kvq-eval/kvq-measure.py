#!/usr/bin/env python3
"""KV cache quality (full-vocab teacher-forced KL) and VRAM measurement helper for llama-server.

Subcommands:
  vram  - measure idle baseline before launch, load, prefill peak, and decode peak VRAM
  kl    - single-request teacher-forced full-vocab KL via force_tokens + dump_logprobs_path
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
import numpy as np


def query_gpu_mem_mib(gpu_index=0):
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used,memory.total,memory.reserved",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [int(x.strip()) for x in out.split(",")]
    return parts[0], parts[1], parts[2]


def wait_ready(base_url, timeout_s=300):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server at {base_url} did not become ready within {timeout_s}s")


def post_json(url, payload, timeout_s=1800):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


class VramSampler:
    def __init__(self, gpu_index=0, interval_s=0.25):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                used, _, _ = query_gpu_mem_mib(self.gpu_index)
                self.samples.append((time.time(), used))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def peak(self, t_start=0.0, t_end=None):
        vals = [u for t, u in self.samples if t >= t_start and (t_end is None or t <= t_end)]
        return max(vals) if vals else 0


def start_server(exe, model, port, extra_args, log_path, env_overrides=None):
    cmd = [exe, "-m", model, "--port", str(port), "--host", "127.0.0.1"]
    if extra_args:
        cmd.extend(shlex.split(extra_args, posix=False))
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    lf = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return proc, lf


def stop_server(proc, lf):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if lf is not None:
        lf.close()


def cmd_vram(args):
    idle_used, total_mib, reserved_mib = query_gpu_mem_mib(args.gpu)
    print(f"GPU {args.gpu}: total={total_mib} MiB, reserved={reserved_mib} MiB, idle_baseline={idle_used} MiB", flush=True)

    env_overrides = {}
    for kv in args.env or []:
        k, v = kv.split("=", 1)
        env_overrides[k] = v

    url = f"http://127.0.0.1:{args.port}"
    proc, lf = start_server(args.exe, args.model, args.port, args.server_args, args.log, env_overrides)
    sampler = VramSampler(args.gpu, args.poll_interval)
    try:
        wait_ready(url)
        time.sleep(1.0)
        loaded_used, _, _ = query_gpu_mem_mib(args.gpu)
        print(f"Loaded: total={loaded_used} MiB (delta=+{loaded_used - idle_used} MiB)", flush=True)

        if args.prompt_tokens_json:
            prompt = json.load(open(args.prompt_tokens_json, encoding="utf-8"))["prompt_tokens"]
        else:
            with open(args.text_file, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            tok_res = post_json(f"{url}/tokenize", {"content": text, "add_special": True})
            prompt = tok_res["tokens"]
            if args.n_prompt > 0 and len(prompt) > args.n_prompt:
                prompt = prompt[: args.n_prompt]

        print(f"Prompt tokens: {len(prompt)}, n_predict: {args.n_predict}", flush=True)
        sampler.start()

        # Phase 1: prefill only (n_predict=1) to isolate PP peak if requested, or single request
        t_req0 = time.time()
        res = post_json(
            f"{url}/completion",
            {
                "prompt": prompt,
                "n_predict": args.n_predict,
                "temperature": 0.0,
                "top_k": 1,
                "cache_prompt": False,
                "ignore_eos": True,
            },
        )
        time.sleep(1.5)
        t_req1 = time.time()
        sampler.stop()

        peak_total = sampler.peak(t_req0, t_req1)
        timings = res.get("timings", {})
        pp_ms = timings.get("prompt_ms", 0.0)
        pp_tps = timings.get("prompt_per_second", 0.0)
        tg_tps = timings.get("predicted_per_second", 0.0)
        t_pp_end = t_req0 + pp_ms / 1000.0
        pp_peak = sampler.peak(t_req0, t_pp_end) if pp_ms > 0 else 0
        tg_peak = sampler.peak(t_pp_end, t_req1) if pp_ms > 0 else peak_total

        print(
            f"PP peak: {pp_peak} MiB (delta=+{pp_peak - idle_used} MiB) | "
            f"TG peak: {tg_peak} MiB (delta=+{tg_peak - idle_used} MiB) | "
            f"Overall peak: {peak_total} MiB (delta=+{peak_total - idle_used} MiB) | "
            f"PP={pp_tps:.1f} t/s, TG={tg_tps:.1f} t/s",
            flush=True,
        )
    finally:
        sampler.stop()
        stop_server(proc, lf)


def eval_kl_arrays(ref_lp_raw, cand_lp_raw, teacher_tokens):
    ref_lp = ref_lp_raw.astype(np.float64)
    m = np.max(ref_lp, axis=1, keepdims=True)
    ref_lp = ref_lp - (m + np.log(np.sum(np.exp(ref_lp - m), axis=1, keepdims=True)))

    cand_lp = cand_lp_raw.astype(np.float64)
    mc = np.max(cand_lp, axis=1, keepdims=True)
    cand_lp = cand_lp - (mc + np.log(np.sum(np.exp(cand_lp - mc), axis=1, keepdims=True)))

    ref_p = np.exp(ref_lp)
    kls = np.maximum(np.sum(ref_p * (ref_lp - cand_lp), axis=1), 0.0)
    kl_sorted = np.sort(kls)
    n_tokens = ref_lp.shape[0]
    top1_match = int(np.sum(np.argmax(ref_lp, axis=1) == np.argmax(cand_lp, axis=1)))

    idx = np.arange(n_tokens)
    t_arr = np.array(teacher_tokens[:n_tokens], dtype=np.int64)
    nll_ref = float(-np.mean(ref_lp[idx, t_arr]))
    nll_cand = float(-np.mean(cand_lp[idx, t_arr]))
    dnll = nll_cand - nll_ref
    return {
        "n_tokens": n_tokens,
        "top1_match": top1_match,
        "kl_mean": float(np.mean(kls)),
        "kl_p50": float(kl_sorted[n_tokens // 2]),
        "kl_p95": float(kl_sorted[int(n_tokens * 0.95)]),
        "kl_max": float(kl_sorted[-1]),
        "dnll": dnll,
        "ppl_ratio": math.exp(dnll),
    }


def cmd_kl(args):
    tdata = json.load(open(args.teacher_json, encoding="utf-8"))
    prompt_tokens = tdata["prompt_tokens"]
    teacher_tokens = tdata["teacher_tokens"]
    if args.n_tokens > 0:
        teacher_tokens = teacher_tokens[: args.n_tokens]

    env_overrides = {}
    for kv in args.env or []:
        k, v = kv.split("=", 1)
        env_overrides[k] = v

    out_lp = os.path.abspath(args.out_lp)
    os.makedirs(os.path.dirname(out_lp), exist_ok=True)

    url = f"http://127.0.0.1:{args.port}"
    proc, lf = start_server(args.exe, args.model, args.port, args.server_args, args.log, env_overrides)
    try:
        wait_ready(url)
        res = post_json(
            f"{url}/completion",
            {
                "prompt": prompt_tokens,
                "n_predict": len(teacher_tokens),
                "temperature": 0.0,
                "top_k": 1,
                "cache_prompt": False,
                "ignore_eos": True,
                "return_tokens": True,
                "force_tokens": teacher_tokens,
                "dump_logprobs_path": out_lp,
            },
        )
        got = res.get("tokens", [])
        if got != teacher_tokens:
            raise RuntimeError(f"force_tokens mismatch: sent {len(teacher_tokens)}, got {len(got)}")
        timings = res.get("timings", {})
        print(
            f"Teacher-forced {len(teacher_tokens)} tokens "
            f"(PP={timings.get('prompt_per_second', 0):.1f} t/s, TG={timings.get('predicted_per_second', 0):.1f} t/s)",
            flush=True,
        )
    finally:
        stop_server(proc, lf)

    if args.ref_lp:
        n_tokens = len(teacher_tokens)
        ref_raw = np.fromfile(args.ref_lp, dtype=np.float32)
        n_vocab = ref_raw.size // n_tokens
        ref_raw = ref_raw.reshape(n_tokens, n_vocab)
        cand_raw = np.fromfile(out_lp, dtype=np.float32).reshape(n_tokens, n_vocab)
        m = eval_kl_arrays(ref_raw, cand_raw, teacher_tokens)
        print(
            f"KL vs {os.path.basename(args.ref_lp)} (N={m['n_tokens']}, V={n_vocab}): "
            f"top-1={m['top1_match']}/{m['n_tokens']}, "
            f"KL mean={m['kl_mean']:.6f} p50={m['kl_p50']:.6f} p95={m['kl_p95']:.6f} max={m['kl_max']:.6f}, "
            f"dNLL={m['dnll']:+.5f}, ppl_ratio={m['ppl_ratio']:.6f}",
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(description="KV quality (KL) and VRAM measurement helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("vram", help="Measure idle, load, prefill, and decode peak VRAM")
    pv.add_argument("--exe", required=True)
    pv.add_argument("--model", required=True)
    pv.add_argument("--server-args", default="")
    pv.add_argument("--env", action="append", default=[])
    pv.add_argument("--text-file", default="")
    pv.add_argument("--prompt-tokens-json", default="")
    pv.add_argument("--n-prompt", type=int, default=478000)
    pv.add_argument("--n-predict", type=int, default=64)
    pv.add_argument("--port", type=int, default=8776)
    pv.add_argument("--gpu", type=int, default=0)
    pv.add_argument("--poll-interval", type=float, default=0.25)
    pv.add_argument("--log", default="tmp/vram_measure.log")
    pv.set_defaults(func=cmd_vram)

    pk = sub.add_parser("kl", help="Measure full-vocab teacher-forced KL via force_tokens + dump_logprobs_path")
    pk.add_argument("--exe", required=True)
    pk.add_argument("--model", required=True)
    pk.add_argument("--server-args", default="")
    pk.add_argument("--env", action="append", default=[])
    pk.add_argument("--teacher-json", required=True)
    pk.add_argument("--ref-lp", default="")
    pk.add_argument("--out-lp", required=True)
    pk.add_argument("--n-tokens", type=int, default=0)
    pk.add_argument("--port", type=int, default=8777)
    pk.add_argument("--log", default="tmp/kl_measure.log")
    pk.set_defaults(func=cmd_kl)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
