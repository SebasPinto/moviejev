# ADR 0004: Offline eval isolates the reranker with held-out positives plus sampled negatives

**Status:** accepted · 2026-09-21

## Context
The question the project exists to answer is whether Jev reranks better, cheaper or faster than
an LLM judge under the same rubric. MovieLens gives us real taste histories and what each user
actually liked next, but no free-text requests and no LLM-proposed candidates.

Two protocols were considered:
(a) Mirror production: LLM proposes candidates from the profile, measure overlap with held-out
    titles. Overlap between 15 proposals and 5 specific held-out titles is near zero for every
    reranker, so the metric would measure the candidate generator, not the reranker.
(b) Fix the candidate set: the user's last 5 positive ratings (rating ≥ 4) plus 15 popular
    movies the user never rated, shuffled. Every reranker orders the same 20 candidates
    against the same profile. `none` becomes a random baseline.

## Decision
(b). Per user: split by time, build a free-text request from training ratings only (top liked and
disliked titles), extract the profile with the production prompt, fetch TMDB cards for all 20
candidates, and score each reranker with NDCG@10, Hit@1, Precision@5, MRR, plus tokens, cost and
`judge()` latency. Users are sampled with a fixed seed; LLM outputs are cached on disk so reruns
only pay for Jev.

## Consequences
- Measures exactly the component Jev replaces. Candidate quality is out of scope here.
- HitRate@10 is uninformative with 5 positives in 20 (near 1.0 for random); it is kept for the
  roadmap's sake but Hit@1, Precision@5 and NDCG@10 carry the signal.
- Negatives are popular, not hard: a reranker that learns "popular = good" is not penalised.
  Hard negatives (same genre as the positives) are a follow-up if results look too easy.
- Profiles come from ratings summaries, not real prompts; exclusions are rarely populated, so
  the `noul` violation question is under-exercised by this eval.
