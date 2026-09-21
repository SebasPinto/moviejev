# Offline evaluation: Jev vs LLM judge vs random

Run 2026-09-21 · 50 MovieLens users · seed 42 · protocol in [ADR 0004](adr/0004-eval-protocol.md) ·
raw results in [eval/results-20260921.json](eval/results-20260921.json).

```bash
uv run python scripts/eval_movielens.py --users 50
```

Setup: per user, the last 5 movies rated ≥ 4 are held out and mixed with 15 popular movies the
user never rated. A profile is extracted with the production prompt from the earlier ratings.
Each reranker orders the same ~20 candidates against the same profile. `none` keeps the shuffled
order (random baseline). LLM: `claude-sonnet-4-6`. Jev: `jev-latest`.

## Results

| reranker | ndcg@10 | hit@1 | precision@5 | mrr | cost (USD, 50 users) | cost / user | judge() p50 | judge() p95 |
|---|---|---|---|---|---|---|---|---|
| none (random) | 0.343 | 0.24 | 0.204 | 0.437 | 0 | 0 | – | – |
| llm_judge | **0.625** | 0.48 | **0.440** | 0.666 | 1.92 | 0.038 | 11.4 s | 22.9 s |
| jev | 0.601 | **0.56** | 0.404 | **0.712** | 0.033 | 0.0007 | 1.05 s | 1.30 s |

HitRate@10 was 0.96–0.98 for all three, including random, as ADR 0004 predicts; it is omitted.
Tokens: llm_judge 525k in / 23k out over 994 calls; Jev 777k in / 83k out over 994 calls.
Profile extraction (shared by all rerankers, 54 LLM calls) cost a further $0.30.

## Paired comparison, Jev minus LLM judge (bootstrap 95 % CI, 5000 resamples)

| metric | mean diff | 95 % CI | Jev wins / ties / losses |
|---|---|---|---|
| ndcg@10 | −0.024 | [−0.065, +0.020] | 15 / 3 / 32 |
| precision@5 | −0.036 | [−0.080, +0.008] | 8 / 27 / 15 |
| hit@1 | +0.080 | [−0.080, +0.240] | 10 / 34 / 6 |
| mrr | +0.045 | [−0.054, +0.143] | 13 / 26 / 11 |

Jev minus random: ndcg@10 +0.258 [+0.174, +0.337], hit@1 +0.32 [+0.16, +0.48]. Both rerankers
are far above random with no overlap in the intervals.

## Reading

- **Quality: statistically indistinguishable at n = 50.** Every Jev-vs-LLM-judge interval
  crosses zero. The LLM judge wins more head-to-heads on NDCG@10 (32 vs 15), so if there is a
  real gap it favours the LLM judge on list depth. Jev leads on whether the top-1 item is a hit
  and on MRR, which is what a "show me one movie" product cares about. Neither lead survives
  the interval.
- **Cost: Jev is ~59× cheaper** ($0.033 vs $1.92 for the same 994 judgements), and that is
  with Jev's output tokens priced at $0 because TypeSafe does not publish an output price;
  at any plausible output rate the ratio stays above 50×.
- **Latency: Jev is ~11× faster per user** at the pipeline's settings (Jev concurrency 8,
  LLM judge 4). Per call that is ~0.4 s vs ~2.3 s, so ~6× faster per judgement; the rest is
  parallelism the LLM provider's rate limits make expensive to match.
- **Calibration is barely exercised.** `low_confidence_rate` was 0.3 % at the 0.35 threshold,
  and the violation (`noul`) question rarely fires because ratings-derived profiles almost
  never contain exclusions. The expected-value scoring is in use, but this protocol does not
  test whether the calibration is *correct*; a reliability diagram over held-out labels is the
  next experiment.

## Caveats

- 50 users, one seed, one LLM. Differences under ~0.05 NDCG are inside the noise.
- Negatives are popular, not hard. A same-genre negative set would separate the rerankers
  more and might change the ordering.
- Profiles come from ratings summaries, not real free-text requests.
- 8 sampled users were skipped because the LLM overran a profile field bound; the bound is
  now clipped instead of rejected (see CHANGELOG), so a rerun will include them.
- The LLM judge returns a point estimate with confidence 1.0 by construction, so tie-breaking
  on confidence only helps Jev. That is the design (ADR 0001), not a bug, but it is a small
  structural advantage in MRR.

## What this answers

"Is Jev actually better or cheaper here?" — **not measurably better, not worse, and roughly two
orders of magnitude cheaper and one order faster** for the reranking step. Whether that trade
is worth it depends on whether the product needs the LLM judge's marginal list depth at 59×
the price.
