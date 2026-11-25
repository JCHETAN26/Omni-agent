# Answer quality on TechQA: plain RAG vs the agent

How often does the system give an engineer a correct, actionable answer, and how often does it know to say
"I don't know"? Measured end-to-end on real IBM technical-support questions.

## Setup

| | |
| --- | --- |
| Dataset | [`nvidia/TechQA-RAG-Eval`](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval) (Apache-2.0), sha256-pinned in `benchmarks/answer_quality/techqa.py` |
| Questions | all 910: 610 answerable (one gold technote each, reference answer written by IBM support), 300 not answerable from the corpus |
| Corpus | 28,481 IBM technotes → 176,710 chunks, indexed in-process in 889 s (Apple M2) |
| Retrieval | hybrid BM25 + Faiss HNSW (MiniLM-L6-v2), RRF, top-4 passages; no learning-to-rank, no cross-encoder |
| Answer model | `gpt-4.1-mini-2025-04-14`, same prompt for both variants |
| Judge | `gpt-4.1-2025-04-14` (different, stronger model), temperature 0, rubric in `benchmarks/answer_quality/judge.py` |
| Code | answers produced at `9b2db11`, judged at `41ba999` |

- **rag:** retrieve the top 4 for the question and generate once.
- **agent:** the full LangGraph agent. It routes the question, grades the passages, rewrites and re-retrieves,
  generates, checks the answer is grounded, then retries or abstains.

Abstentions are scored by rule, not by the judge:

- an abstention on an answerable question is a **miss**;
- an abstention on an unanswerable question is **correct**;
- any answer to an unanswerable question is **wrong**.

An agent turn in which any step silently fell back to the offline heuristic was rejected and retried, so every
scored answer came from the LLM. There were 0 errors in either variant.

## Results

| Metric (n) | rag | agent | agent − rag (95% CI, McNemar p) |
| --- | --- | --- | --- |
| **Correct**, answerable (610) | **48.0%** | **48.0%** | 0.0 pts (−3.4, +3.4), p = 1.0 |
| Correct or partial, answerable (610) | 83.1% | 72.0% | |
| Incorrect, answerable (610) | 15.4% | **8.0%** | |
| Missed (abstained), answerable (610) | 1.5% | 20.0% | |
| **Abstained correctly**, unanswerable (300) | 2.7% | **31.7%** | **+29.0 pts** (+24.0, +34.3), p < 0.001 |
| **End-to-end accuracy** (correct + correct abstentions, 910) | 33.1% | **42.6%** | **+9.6 pts** (+6.5, +12.5), p < 0.001 |
| Precision when it answers | 32.8% (893) | **42.3%** (693) | |
| Faithful to passages, when answering | 76.0% (893) | **82.4%** (693) | +3.5 pts on the 688 both answered, p = 0.058 |
| Actionable, answered answerable | 48.2% (601) | 55.7% (488) | |
| Gold technote in top 4, answerable (610) | 57.1% | 57.5% | |
| Correct when the gold technote was retrieved | 69.3% (348) | 69.0% (351) | |
| LLM tokens per question (prompt + completion) | 1,338 | 4,757 (3.6×) | |
| Answer cost, 910 questions (list price) | $0.72 | $2.30 | |

95% CIs are from 2,000 bootstrap resamples. Paired deltas use the same questions, with an exact McNemar test. Latency
(p50 22.9 s rag, 20.5 s agent) was measured with 8 questions in flight on one laptop, CPU embedding included, so it is
not a single-user latency figure.

## What this says

- **The agent is not more often right; it is less often wrong.**
  - On answerable questions both variants are correct 48.0% of the time. Each gets 57 questions the other misses.
  - The agent halves the wrong answers (15.4% → 8.0%) and recognises 32% of unanswerable questions, against 3% for
    plain RAG.
  - Net effect: +9.6 points end-to-end accuracy and +9.5 points precision when it answers.
- **The price of that caution** is that the agent abstains on 20% of answerable questions, against 1.5% for RAG, and
  it uses 3.6× the tokens.
- **Retrieval is the ceiling.**
  - The gold technote reaches the top 4 for only 57% of answerable questions.
  - When it does, about 69% of answers are correct.
  - The ranking improvements measured on BEIR (learning-to-rank, the cross-encoder) were not enabled here and are
    the obvious next experiment.

## Judge reliability

- **Self-consistency:** 200 judged answers (100 per variant, seed 11) were re-graded. The 3-way correctness verdict
  was identical 99.0% of the time (Cohen's κ 0.98), and the faithfulness verdict 96.5% of the time
  (`judge-consistency.json`).
- **Human agreement:** a blind 40-item sample is exported by `run.py human-sample`. Agreement is not measured yet.
- **Rubric change:** after the pilot, the rubric was tightened before any of these answers were judged. A "partial"
  verdict now has to contain an element of the reference's fix; agreeing with background facts is not enough.

## Known issues found by this run

- **Fixed before this run (`9b2db11`).** In the 20-question pilot, the agent's scoped-conflict check read
  `No.=0 … Site No.=251` in a Netcool filter string as a setting conflict, and the agent abstained on an
  answerable question. Pilot answers were discarded.
- **Still present.** 5 of 910 agent answers (0.5%) are still false "sources disagree" abstentions. They are triggered
  by different SQL error codes (for example `sqlcode -911` vs `-952`) in unrelated error messages. The check was
  designed for runbook settings and is too eager on vendor technotes.

## Files

| File | Contents |
| --- | --- |
| `summary.json` | every metric above, with CIs and paired tests |
| `answers-{rag,agent}.jsonl.gz` | question, reference, answer, abstention, passages, gold hit, latency per question |
| `judgments-{rag,agent}.jsonl.gz` | judge verdict and reason per answered question |
| `judgments-repeat-*.jsonl.gz`, `judge-consistency.json` | re-graded sample and agreement |
| `meta-*.json`, `judge-meta-*.json` | token counts, wall time, index stats, models |

Reproduce (needs `LLM_API_KEY` or `OPENAI_API_KEY`):

```bash
python -m benchmarks.answer_quality.run fetch  --data ~/omni-eval/techqa
python -m benchmarks.answer_quality.run answer --data ~/omni-eval/techqa --run <dir>
python -m benchmarks.answer_quality.run judge  --run <dir>
python -m benchmarks.answer_quality.run consistency --run <dir>
python -m benchmarks.answer_quality.run report --run <dir>
```
