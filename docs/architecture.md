# Architecture

## Components

| Module | Responsibility |
|---|---|
| `moviejev.config` | Typed settings from env (`pydantic-settings`). Single source of configuration. |
| `moviejev.models` | Domain types: `TasteProfile`, `Candidate`, `Movie`, `Verdict`, `Recommendation`. |
| `moviejev.llm` | `LLM` abstraction (`complete`, `complete_json`) with Anthropic and OpenAI adapters. |
| `moviejev.catalog` | `Catalog.resolve(Candidate) -> Movie | None`. TMDB implementation. |
| `moviejev.reranker` | `Reranker.judge(profile, movies) -> list[Verdict]`. Jev, LLM-judge, passthrough. |
| `moviejev.pipeline` | Orchestration: profile → candidates → verify → judge → rank → explain. |
| `moviejev.api` / `moviejev.cli` | Thin delivery layers. No business logic. |
| `moviejev.wiring` | Builds the object graph from settings. Only place that reads secrets. |

## Why one Jev call per candidate

TypeSafe's guidance is that accuracy drops as the state fills with content unrelated to the
decision. A single state with all 15 candidates and 15 Score questions would be cheaper (one
request) but each question would see 14 irrelevant movie cards. Per-candidate states are ~300
tokens each; at $0.042 / 1M input tokens, 15 calls cost ~$0.0002. Latency is bounded by the
slowest call, not the sum, thanks to `asyncio.gather` with a concurrency semaphore.

## Scoring

`Score` returns a probability over the 5 ordered fit levels. We use the **expected value** of
that distribution (normalised to [0, 1]) rather than the argmax: two movies both "most likely
level 3" are distinguished by how much mass sits on level 4 vs level 2.

`Noul` returns P(violates an exclusion). Final score = `expected_fit × (1 − P(violation))`.
A movie the model is 90 % sure breaks a "no horror" rule is effectively removed without a hard
threshold.

`confidence` (from the distribution's shape) is not used for ordering beyond tie-breaking. It is
surfaced as `low_confidence` so a UI can label or hide those items. Thresholding is a product
decision, not a pipeline one.

## The LLM judge exists to make a fair comparison

`LLMJudgeReranker` answers the *same* rubric (`FIT_LEVELS`, `REASON_CHOICES`,
`VIOLATION_INSTRUCTION` are shared constants). It cannot produce a calibrated distribution, so it
returns a point estimate with confidence 1.0. Any quality difference in the offline eval is then
attributable to the model, not the prompt.
