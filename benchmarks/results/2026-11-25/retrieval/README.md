# Retrieval recall on TechQA: getting the gold technote into the answer's context

The answer-quality run (`../../2026-11-24/answer-quality/`) found that retrieval is the ceiling. The gold technote
reached the 4 passages the answer is written from for only 57% of answerable questions, and when it did, about 69%
of answers were correct. This run measures what lifts that number.

## Method

| | |
| --- | --- |
| Questions | the 610 answerable TechQA questions (TRAIN 450, DEV 160), each with one gold technote |
| Corpus | 28,481 technotes → 176,710 chunks; hybrid BM25 + Faiss HNSW (MiniLM-L6-v2), RRF k=60 |
| Pool | the top 100 fused chunks per question, the same pool production fetches (`pools-meta.json`) |
| Metric | hit@k: at least one of the k returned chunks is from the gold technote. The answer uses k=4. |
| Code | `benchmarks/answer_quality/retrieval.py` |

- **Same code as production.** Indexing takes about 15 minutes, so each question's pool is collected once. Every
  configuration is then replayed through the production `search.pipeline.retrieve_with_rerank`, with stored
  cross-encoder scores standing in for live inference.
  - Nothing about ranking is re-implemented.
  - A live check on 40 DEV questions with the in-process reranker reproduced the stored scores to within 3×10⁻⁸ and
    gave the same hit@4 on all 40.
- **No tuning on reported questions.**
  - Learning-to-rank (LTR) models are trained on TRAIN only. A 20% slice of TRAIN is used for early stopping and to
    choose between models.
  - The DEV table reports what that choice gives on questions no model saw.
  - The headline is 5-fold cross-validation over all 610 questions. Each question is ranked by a model trained on
    the other four folds.
- **Reading the results.** 95% CIs are from 2,000 bootstrap resamples. Paired comparisons with the baseline use an
  exact McNemar test.

## Results

**Headline: out-of-fold, all 610 answerable questions.** The configuration is the bge-reranker-base cross-encoder
(512 tokens) over the top 50 fused chunks, an LTR model on its scores, and at most 1 chunk per technote.

| | Baseline (RRF, as in the answer-quality run) | LTR + bge-reranker | Δ (gained / lost questions, McNemar p) |
| --- | --- | --- | --- |
| **hit@4** | 57.4% | **65.4%** (61.6–69.2) | **+8.0 pts** (+66 / −17, p < 0.001) |
| hit@10 | 66.1% | 73.3% (69.7–76.7) | +7.2 pts (+49 / −5, p < 0.001) |
| hit@20 | 73.6% | 77.2% (73.8–80.5) | +3.6 pts (+27 / −5, p < 0.001) |

- **Second fold seed** (29): hit@4 66.1%, hit@10 73.0%, hit@20 77.5%.
- **20 candidates instead of 50:** hit@4 64.6%; hit@20 does not change.
  - **50 was chosen after seeing both cross-validation results,** so the 65% line is cleared narrowly and with that
    caveat.
- **Ceiling:** the gold technote is somewhere in the 100-chunk pool for 86.4% of questions.

**All configurations, DEV (n = 160), LTR trained on TRAIN only:**

| Configuration | TRAIN hit@4 | DEV hit@4 | DEV hit@10 | DEV hit@20 | DEV hit@4 vs baseline |
| --- | --- | --- | --- | --- | --- |
| baseline (RRF) | 55.1% | 63.7% | 72.5% | 78.1% | |
| RRF, 1 chunk/doc | 56.0% | 63.7% | 74.4% | 78.1% | +0.0 (p=1) |
| bge-reranker @20 | 56.4% | 60.6% | 75.0% | 78.1% | −3.1 (p=0.42) |
| bge-reranker @50 | 57.1% | **54.4%** | 71.2% | 80.0% | **−9.4 (p=0.006)** |
| bge-reranker @100 | 56.7% | 54.4% | 68.1% | 76.9% | −9.4 (p=0.008) |
| bge-reranker @50, 1 chunk/doc | 57.3% | 55.6% | 72.5% | 81.9% | −8.1 (p=0.024) |
| **LTR + bge-reranker @20, 1 chunk/doc** | 63.6%* | **69.4%** | 77.5% | 78.1% | +5.6 (p=0.064) |
| **LTR + bge-reranker @50, 1 chunk/doc** | 65.8%* | **69.4%** | 77.5% | 81.9% | +5.6 (p=0.093) |

\* in-sample: the LTR model was trained on these questions.

The DEV pool ceiling is 90.0%. Earlier measurements, made before the identifier change below (baseline DEV hit@4 was
also 63.7% then; `configs-minilm-*.json`):

- **ms-marco-MiniLM-L-6 cross-encoder, alone:** DEV hit@4 57.5–60.6% at 256 or 512 tokens. It is worse than the
  baseline at every depth.
- **LTR trained on BEIR, no TechQA training:** DEV hit@4 61.3–65.0%, not significantly different from the baseline.

## What this says

- **An off-the-shelf cross-encoder hurts on this data.**
  - Both rerankers are trained on short web-search queries.
  - TechQA questions are long support posts that quote error messages, and the rerankers push the gold technote
    down. bge-reranker over 50 candidates loses 9 points of hit@4 on DEV.
- **The cross-encoder score is still a useful feature.**
  - An LTR model trained on TechQA learns how much to trust the cross-encoder against BM25, the dense ranks and RRF.
  - Combined, they gain +8 points of hit@4 out-of-sample.
  - The same LTR without the cross-encoder feature gains +3.9 points (`crossval` without `--uses-ce`, before the
    identifier change).
- **One chunk per technote** has little effect on hit@4 here. It helps hit@10 a little.
- **Exact-identifier anchoring** used to miss IBM message IDs that end in a severity letter (`SQL0911N`,
  `DSNT408I`, `CRIMA1217E`) and SQLCODEs (`-911`), and treated `www-01` in `www-01.ibm.com` as an identifier.
  - The fix (`search/evidence.py`) adds +0.4 pts of TRAIN hit@4 for RRF and +0.7 pts with LTR. DEV is unchanged.
  - Most exact IDs are already caught by BM25.
  - Turning anchoring off costs 1.8–2.5 points for RRF.
- **Cost:** on an Apple M2 (MPS), the in-process reranker scores 50 passages in 1.8 s at p50 and 4.4 s at p95, or
  20 passages in 0.75 s.
  - That is above the default 1.5 s `rerank_timeout_s`. Past the timeout, the pipeline falls back to RRF order.
  - On a GPU or with Triton this would be much shorter. That has not been measured for bge here.

## Use it

```bash
INFERENCE_BACKEND=local LOCAL_RERANK_MODEL=BAAI/bge-reranker-base RERANK_ENABLED=true \
RERANK_CANDIDATES=50 KB_MAX_CHUNKS_PER_DOC=1 RERANK_TIMEOUT_S=5 LTR_MODEL_PATH=<model dir>
```

The LTR model is domain-specific. A model trained on TechQA is for IBM support content. Train one on your own
questions with `retrieval.py train-ltr`.

## Files

| File | Contents |
| --- | --- |
| `crossval-c50-seed{13,29}.json`, `crossval-c20-seed13.json` | out-of-fold hit@4/10/20 with CIs, per fold, vs baseline |
| `configs-bge-reranker.json` | every configuration above on DEV, TRAIN and ALL |
| `configs-minilm-L{256,512}.json` | the same with the MiniLM cross-encoder, plus the BEIR LTR models (before the identifier change) |
| `ltr-train-only-training.json` | training curve of the TRAIN-only LTR model |
| `pools-meta.json` | index and pool size |

Reproduce:

```bash
R="python -m benchmarks.answer_quality.retrieval"
$R collect  --data ~/omni-eval/techqa-raw --work <w>          # ~15 min indexing
$R score-ce --work <w> --model BAAI/bge-reranker-base --max-length 512 --batch 32   # ~55 min on M2
$R train-ltr --work <w> --out <w>/ltr --uses-ce --ce-model BAAI/bge-reranker-base --ce-max-length 512
$R evaluate --work <w> --ce-model BAAI/bge-reranker-base --ce-max-length 512 --ltr techqa-ce=<w>/ltr
$R crossval --work <w> --uses-ce --candidates 50
```
