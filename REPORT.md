# LLM Inference + Observability — Assignment Report
**Inna Bichikova**

---

## 1. Serving Configuration (Phase 1)

**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`  
**Hardware:** 1× NVIDIA H100 NVLink 80GB

**Final vLLM flags:**

| Flag | Value | Justification |
|------|-------|---------------|
| `--max-model-len` | 8192 | Model's full context exceeds H100 memory; 8192 covers our 1.5-3K prompt workload with headroom |
| `--host` | 0.0.0.0 | Binds to all interfaces so the agent can reach vLLM from localhost |
| `--port` | 8000 | Standard port, matches agent config and port forwarding setup |

**Tuning attempt:** Also tested `--max-num-seqs 32 --enable-chunked-prefill` to increase concurrency. This did not improve P95 latency (went from 7.99s to 10.72s) because the bottleneck is sequential LLM calls inside the agent, not vLLM's batching capacity.

---

## 2. Baseline Eval Results (Phase 5)

**Eval set:** 30 curated BIRD-bench questions across 11 SQLite databases.

| Metric | Value |
|--------|-------|
| Total questions | 30 |
| Skipped | 0 |
| Final correct | 9 |
| **Final pass rate** | **30%** |

**Per-iteration pass rate:**

| Iteration | Pass rate |
|-----------|-----------|
| iter_1 | 33.3% |
| iter_2 | 30.0% |
| iter_3 | 30.0% |

**Commentary:** The pass rate at iteration 1 (33.3%) is slightly higher than at iterations 2-3 (30%). This means the revise loop occasionally takes a correct answer and makes it worse. The verifier is too aggressive — it sometimes rejects correct results, for example flagging a 0-row result as wrong when the real answer is indeed 0 rows.

---

## 3. SLO Diagnosis and Iteration (Phase 6)

**Target SLO:** P95 end-to-end agent latency under 5s at 10+ RPS over 5 minutes.

### Baseline load test (2 RPS, 60s)

| Metric | Value |
|--------|-------|
| Achieved RPS | 1.89 |
| P50 latency | 0.74s |
| P95 latency | 3.98s |
| HTTP errors | 16 |

### Higher load test (5 RPS, 60s) — before tuning

| Metric | Value |
|--------|-------|
| Achieved RPS | 2.89 |
| P50 latency | 1.30s |
| P95 latency | 7.99s |
| HTTP errors | 38 |

### Iteration log

**Iteration 1:** Saw high P95 (7.99s) and achieved RPS well below target (2.89 vs 5.0) → hypothesized vLLM was not batching concurrent requests efficiently → added `--max-num-seqs 32` and `--enable-chunked-prefill` → P95 got worse (10.72s), achieved RPS unchanged.

**Root cause:** The bottleneck is not vLLM — it is the agent architecture. Each agent run makes 2-3 sequential LLM calls (generate → verify → optional revise). With each call taking 1-2s, a single agent run takes 2-6s minimum. This makes 10 RPS structurally impossible without parallelizing agent calls or reducing LLM calls per request. An additional issue: ~13% of requests fail with HTTP 400 (context length exceeded) because some BIRD databases have very large schemas that exceed the 8192 token limit.

### Final numbers (5 RPS, 60s) — after tuning

| Metric | Value |
|--------|-------|
| Achieved RPS | 2.50 |
| P50 latency | 1.90s |
| P95 latency | 10.73s |

**Verdict: SLO missed.** Achieved ~2.5 RPS vs target 10 RPS. P95 latency 10.73s vs target 5s. The gap is 4× on RPS and 2× on latency. The fundamental constraint is the sequential multi-step agent design, not the inference server configuration.

---

## 4. Agent Value

The verify→revise loop adds real value in specific cases. During testing, it caught a clearly wrong result where the average superhero weight came back as 33 million kg — the verifier correctly flagged this as implausible and triggered a revision. However, the per-iteration pass rate tells a more nuanced story: iter_1 pass rate is 33.3% but drops to 30% at iter_2 and iter_3. This means the loop is slightly hurting overall accuracy — it occasionally takes a correct first answer and revises it into a wrong one. The most likely cause is an overly aggressive verifier: it rejects results that look surprising but are actually correct, for example a 0-row result when the real answer genuinely is 0 rows. The loop earns its keep on obvious failures like the superhero weight case, but needs a more precise verifier prompt before it reliably improves accuracy across the full eval set.

---

## 5. What I Would Do With More Time

1. **Fix the verifier prompt** — add explicit instructions not to reject 0-row results unless the question clearly implies rows should exist. The current false-rejection rate is the main reason the loop hurts accuracy.

2. **Increase `--max-model-len` to 16384** — this would eliminate the ~13% of requests that fail due to context length, which is the biggest source of HTTP errors under load.

3. **Cache database schemas** — currently the schema is re-rendered on every request. Caching it would reduce prompt size and save tokens on every call.

4. **Run the full 5-minute load test** — we only ran 60-second tests due to time constraints. A proper 300-second run would give more stable P95 numbers.

5. **Parallelize agent calls** — the fundamental latency bottleneck is sequential LLM calls. Running generate and a speculative verify in parallel would cut latency roughly in half.