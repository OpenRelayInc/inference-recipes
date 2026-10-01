#!/usr/bin/env python3
"""Closed-loop TPM ladder for an OpenAI-compatible streaming endpoint.
The API key comes only from the OR_KEY environment variable.
Local-engine copy: TPM_URL and TPM_MODEL override the endpoint and model; adds tpot_mean_ms (mean of per-request
(e2e - ttft) / (completion_tokens - 1)) and lit_tpm (Little's-law output tokens/min)."""
import argparse, asyncio, datetime, json, os, random, resource, statistics, sys, time
import aiohttp

URL = os.environ.get("TPM_URL", "https://inference.openrelay.inc/v1/chat/completions")
MODEL = os.environ.get("TPM_MODEL", "openrelay/qwen3.8-27b")
STAGGER = 0.0
SYL = ["ka", "lo", "mi", "ren", "tas", "vo", "pel", "dun", "sar", "ith", "or", "bel", "cra", "min",
       "tor", "ges", "phi", "lan", "qu", "ex", "nor", "wid", "sta", "mo", "rel", "fan", "gri", "hol"]


def rand_words(n, rng):
    return " ".join("".join(rng.choice(SYL) for _ in range(rng.randint(1, 3))) for _ in range(n))


def make_prompt(words, rng):
    nonce = "%016x" % rng.getrandbits(64)
    return (f"{nonce} Read the following notes and then write a long, detailed essay about them.\n\n"
            + rand_words(words, rng))


def body(prompt, out_tokens, stream):
    b = {"model": MODEL, "max_tokens": out_tokens, "ignore_eos": True, "temperature": 0.7,
         "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "user", "content": prompt}]}
    if stream:
        b["stream"] = True
        b["stream_options"] = {"include_usage": True}
    return b


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


async def one_stream(sess, hdr, words, out_tokens, rng):
    r = {"t0": time.monotonic(), "status": None, "ttft": None, "end": None, "pt": 0, "ct": 0,
         "done": False, "usage": False, "err": None}
    try:
        async with sess.post(URL, headers=hdr, json=body(make_prompt(words, rng), out_tokens, True)) as resp:
            r["status"] = resp.status
            if resp.status != 200:
                await resp.read()
                r["end"] = time.monotonic()
                return r
            buf = b""
            async for chunk in resp.content.iter_any():
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        r["done"] = True
                        continue
                    try:
                        j = json.loads(data)
                    except Exception:
                        continue
                    ch = j.get("choices") or []
                    if ch and r["ttft"] is None:
                        d = ch[0].get("delta") or {}
                        if d.get("content") or d.get("reasoning_content"):
                            r["ttft"] = time.monotonic() - r["t0"]
                    u = j.get("usage")
                    if u:
                        r["usage"] = True
                        r["pt"] = u.get("prompt_tokens", 0)
                        r["ct"] = u.get("completion_tokens", 0)
            r["end"] = time.monotonic()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        r["err"] = type(e).__name__
        r["end"] = time.monotonic()
    return r


async def calibrate(hdr, out_tokens, target):
    rng = random.Random()
    words = int(target / 2.5)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as sess:
        for i in range(4):
            async with sess.post(URL, headers=hdr, json=body(make_prompt(words, rng), out_tokens, False)) as resp:
                txt = await resp.text()
                if resp.status != 200:
                    print(f"probe status={resp.status} body={txt[:300]}", flush=True)
                    return None
                u = json.loads(txt)["usage"]
            print(f"probe {i}: words={words} status=200 prompt_tokens={u['prompt_tokens']} "
                  f"completion_tokens={u['completion_tokens']}", flush=True)
            if 2400 <= u["prompt_tokens"] <= 2600 and abs(u["prompt_tokens"] - target) < 40:
                break
            words = int(words * target / u["prompt_tokens"])
        # one streaming probe to confirm TTFT detection + usage chunk
        r = await one_stream(sess, hdr, words, out_tokens, rng)
        print(f"stream probe: status={r['status']} ttft_ms={None if r['ttft'] is None else round(r['ttft']*1000)} "
              f"pt={r['pt']} ct={r['ct']} done={r['done']} usage={r['usage']} err={r['err']}", flush=True)
    print(json.dumps({"calibrated_words": words}), flush=True)
    return words


async def run_step(hdr, n, seconds, warmup, words, out_tokens):
    results = []
    stop_at = time.monotonic() + seconds
    t_start = time.monotonic()
    wall0 = time.time() - (time.monotonic() - t_start)
    conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=seconds + 60, sock_read=60)) as sess:
        inflight = [0]

        async def worker(i):
            rng = random.Random()
            # small stagger by default; --stagger S spreads stream starts over S seconds so a closed loop with a
            # fixed output length does not run in lockstep (desync measurement only).
            await asyncio.sleep(i * (STAGGER / n if STAGGER else 0.005))
            while time.monotonic() < stop_at:
                inflight[0] += 1
                r = await one_stream(sess, hdr, words, out_tokens, rng)
                inflight[0] -= 1
                results.append(r)
                if r["status"] == 429 or r["err"]:
                    await asyncio.sleep(0.2)
        tasks = [asyncio.create_task(worker(i)) for i in range(n)]
        await asyncio.sleep(max(0, stop_at - time.monotonic()))
        inflight_at_stop = inflight[0]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    w0, w1 = t_start + warmup, stop_at
    win = [r for r in results if r["end"] is not None and w0 <= r["end"] <= w1]
    ok = [r for r in win if r["status"] == 200 and r["err"] is None and r["done"] and r["usage"] and r["ttft"] is not None]
    n429 = sum(1 for r in win if r["status"] == 429)
    trunc = sum(1 for r in win if r["status"] == 200 and r["err"] is None and not (r["done"] and r["usage"] and r["ttft"] is not None))
    other = sum(1 for r in win if r["err"] is not None or (r["status"] not in (200, 429)))
    mw = (w1 - w0) / 60.0
    in_tok = sum(r["pt"] for r in ok)
    out_tok = sum(r["ct"] for r in ok)
    ttfts = [r["ttft"] * 1000 for r in ok]
    e2e = [r["end"] - r["t0"] for r in ok]
    dec = [(r["ct"] - 1) / (r["end"] - r["t0"] - r["ttft"]) for r in ok if r["ct"] > 1 and (r["end"] - r["t0"] - r["ttft"]) > 0]
    total = len(win)
    errs = n429 + trunc + other
    statuses = {}
    for r in win:
        k = str(r["status"]) if r["err"] is None else r["err"]
        statuses[k] = statuses.get(k, 0) + 1
    def iso(m):
        return datetime.datetime.fromtimestamp(wall0 + (m - t_start), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    allok = [r for r in results if r["status"] == 200 and r["err"] is None and r["usage"]]
    run_all = {"run_start_utc": iso(t_start), "run_stop_utc": iso(stop_at), "completed_with_usage": len(allok),
               "attempts": len(results), "in_tokens": sum(r["pt"] for r in allok),
               "out_tokens": sum(r["ct"] for r in allok), "inflight_cancelled_at_stop": inflight_at_stop}
    window = {"start_utc": iso(w0), "end_utc": iso(w1), "start_epoch": round(wall0 + (w0 - t_start), 3),
              "end_epoch": round(wall0 + (w1 - t_start), 3), "seconds": round(w1 - w0, 1),
              "in_tokens": in_tok, "out_tokens": out_tok, "sum_tokens": in_tok + out_tok}
    minutes = []
    nmin = int(round(w1 - w0) // 60)
    for m in range(nmin):
        a0, a1 = w0 + 60 * m, w0 + 60 * (m + 1)
        mo = [r for r in ok if a0 <= r["end"] < a1]
        ma = [r for r in win if a0 <= r["end"] < a1]
        mt = [r["ttft"] * 1000 for r in mo]
        minutes.append({"min": m + 1, "start_utc": iso(a0), "completed": len(mo),
                        "in_tok": sum(r["pt"] for r in mo), "out_tok": sum(r["ct"] for r in mo),
                        "ttft_p50": round(pct(mt, 50)) if mt else None, "ttft_p95": round(pct(mt, 95)) if mt else None,
                        "n429": sum(1 for r in ma if r["status"] == 429),
                        "errs": sum(1 for r in ma if r["err"] is not None or r["status"] not in (200, 429))})
    return {
        "window": window, "minutes": minutes, "run_all": run_all,
        "conc": n, "completed": len(ok), "attempts": total, "rps": round(len(ok) / (w1 - w0), 2),
        "in_tpm": round(in_tok / mw), "out_tpm": round(out_tok / mw), "tot_tpm": round((in_tok + out_tok) / mw),
        "ttft_p50": round(pct(ttfts, 50)), "ttft_p95": round(pct(ttfts, 95)), "ttft_p99": round(pct(ttfts, 99)),
        "e2e_p50": round(pct(e2e, 50), 2), "e2e_p95": round(pct(e2e, 95), 2),
        "decode_tps": round(statistics.mean(dec), 1) if dec else None,
        "tpot_mean_ms": round(1000 * statistics.mean([1 / x for x in dec]), 2) if dec else None,
        # Little's law: n streams x mean completion tokens / mean e2e. Free of the wave quantization that a
        # closed loop puts on "requests completed inside the window".
        "lit_tpm": round(n * statistics.mean([r["ct"] for r in ok]) / statistics.mean(e2e) * 60) if ok else None,
        "mean_pt": round(statistics.mean([r["pt"] for r in ok])) if ok else None,
        "mean_ct": round(statistics.mean([r["ct"] for r in ok])) if ok else None,
        "n429": n429, "other_err": other, "truncated": trunc,
        "err_rate": round(errs / total, 4) if total else 1.0, "statuses": statuses,
    }


HDR = ("conc", "completed", "rps", "in_tpm", "out_tpm", "tot_tpm", "ttft_p50", "ttft_p95", "ttft_p99",
       "e2e_p50", "e2e_p95", "decode_tps", "mean_pt", "mean_ct", "n429", "other_err", "truncated", "err_rate", "tpot_mean_ms", "lit_tpm")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="32,64,128,256,384,512,704")
    ap.add_argument("--seconds", type=int, default=90)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--input-tokens", type=int, default=2500)
    ap.add_argument("--output-tokens", type=int, default=500)
    ap.add_argument("--words", type=int, default=0, help="skip calibration, use this word count")
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--prev-429", action="store_true", help="previous invocation's last step had 429s")
    ap.add_argument("--pause", type=int, default=8)
    ap.add_argument("--stagger", type=float, default=0.0, help="spread stream starts over this many seconds")
    a = ap.parse_args()
    global STAGGER
    STAGGER = a.stagger
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    key = os.environ.get("OR_KEY", "").strip()
    if not key:
        print("OR_KEY missing", flush=True)
        sys.exit(2)
    hdr = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    words = a.words or await calibrate(hdr, a.output_tokens, a.input_tokens)
    if not words or a.calibrate_only:
        return
    print(" | ".join(HDR), flush=True)
    rows, prev429 = [], a.prev_429
    for s in [int(x) for x in a.steps.split(",")]:
        row = await run_step(hdr, s, a.seconds, a.warmup, words, a.output_tokens)
        rows.append(row)
        print(" | ".join(str(row[k]) for k in HDR) + "  statuses=" + json.dumps(row["statuses"]), flush=True)
        print("WINDOW " + json.dumps(row["window"]), flush=True)
        print("RUNALL " + json.dumps(row["run_all"]), flush=True)
        print("minute | start_utc | completed | in_tok | out_tok | ttft_p50 | ttft_p95 | n429 | errs", flush=True)
        for mrow in row["minutes"]:
            print(" | ".join(str(mrow[k]) for k in ("min", "start_utc", "completed", "in_tok", "out_tok",
                                                     "ttft_p50", "ttft_p95", "n429", "errs")), flush=True)
        reason = None
        if row["err_rate"] >= 0.05:
            reason = "error rate >= 5%"
        elif row["n429"] and prev429:
            reason = "429s at two consecutive steps"
        elif row["ttft_p95"] == row["ttft_p95"] and row["ttft_p95"] > 15000:
            reason = "TTFT p95 > 15s"
        prev429 = row["n429"] > 0
        if reason:
            print(f"STOP: {reason} at conc={s}", flush=True)
            break
        await asyncio.sleep(a.pause)
    print("SUMMARY " + json.dumps({"words": words, "rows": rows}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
