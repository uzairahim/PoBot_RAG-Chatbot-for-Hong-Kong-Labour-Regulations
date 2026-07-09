# Evaluation

## Approach summary

This system answers questions about Hong Kong labour and employment regulations
using a Retrieval-Augmented Generation (RAG) pipeline over **11 official HK
Government documents** (Labour Department, Immigration Department, MPFA — 968
text chunks). A query is embedded with a local multilingual model
(`multilingual-e5-small`), the most similar chunks are fetched from a **FAISS**
index, and a **cross-encoder re-ranker** re-orders them so genuinely relevant
passages beat keyword-matching boilerplate. The top chunks are passed to a
swappable LLM (default: Groq Llama-3.3-70B) with a prompt that forces answers to
be **grounded in the retrieved context, cited inline, and language-matched**
(English or Tagalog). A calibrated **confidence gate** derived from the
cross-encoder score makes the bot say *"I don't know"* rather than guess when the
corpus can't answer. Retrieval quality is measured quantitatively (below), and
the main limitation is that answer completeness depends on how the source
documents fragment a topic across chunks.

## How to reproduce

```bash
python -m src.fetch_sources     # download the 11 source documents
python -m src.ingest            # clean, chunk, embed, build FAISS index
python -m src.evaluate --k 5    # quantitative retrieval metrics (below)
pytest -q                       # unit + integration tests
python chatbot.py               # interactive chatbot
```

## 1. Quantitative retrieval evaluation

We hand-labeled a gold set of **20 questions** (`data/eval/gold_questions.json`),
each tagged with the source document(s) that should answer it. A retrieved chunk
counts as relevant if it comes from a labeled source. We report standard IR
metrics at k=5, comparing the bi-encoder alone vs. bi-encoder + cross-encoder
re-ranking (`python -m src.evaluate`):

| Metric | Bi-encoder only | + Cross-encoder rerank | Δ |
|---|--:|--:|--:|
| Hit@5  | 0.950 | **1.000** | +0.050 |
| MRR    | 0.850 | 0.846 | −0.004 |
| nDCG@5 | 0.869 | **0.889** | +0.020 |

- **Hit@5 → 1.000**: after re-ranking, *every* gold question has a correct-source
  chunk in the top-5. This is the answerability ceiling — retrieval is no longer
  the bottleneck for any test question.
- **nDCG@5 +0.020**: re-ranking produces better-ordered results (correct chunks
  sit higher).
- **MRR −0.004 (flat)**: the bi-encoder already placed a correct chunk near the
  top for most questions, so the re-ranker's gain shows up in coverage/ordering
  rather than the top-1 position. This is an honest, small effect — re-ranking
  helps here, modestly.

*Granularity caveat:* relevance is **document-level** (right document, not
hand-labeled right sentence). This is a deliberate, transparent simplification;
true chunk-level labels would be stronger but require far more annotation.

## 2. Sample queries and responses

All outputs below are verbatim from `python chatbot.py`.

### Q1 — Recruitment agency rules (English)
> **Q:** What is the maximum commission an employment agency can charge?
>
> **A:** The maximum commission an employment agency can charge is 10% of the
> first month's wages received by the job-seeker after they have been placed in
> employment by the agency [1], [2], [3], [4].
> *(confidence: 97%)*
>
> Sources: Code of Practice for Employment Agencies (labour.gov.hk) — pp. 10, 16, 42

Correct under HK law. High confidence, well-cited.

### Q2 — Work-injury compensation (English)
> **Q:** What should I do and what compensation can I get if I am injured at work?
>
> **A:** If you are injured at work, you should notify your
> employer/supervisor/human resources department [1]. Your employer should then
> notify the insurer … If the injury results in permanent incapacity, you may be
> entitled to compensation … proportionate to the loss of earning capacity [3], [4].
>
> Sources: A Concise Guide to the Employees' Compensation Ordinance (labour.gov.hk) — pp. 8, 15, 49, 52

Accurate, multi-chunk synthesis with page-level citations.

### Q3 — Minimum wage (Tagalog → Tagalog)
> **Q:** Ano ang minimum wage para sa isang domestic helper sa Hong Kong?
>
> **A:** Ang minimum wage para sa isang domestic helper sa Hong Kong ay
> HK$5,100 per month [1], [2], [3].
>
> Sources: Hiring Foreign Domestic Helpers (gov.hk); FDH FAQ (labour.gov.hk)

The bot **retrieved from English sources and answered in Tagalog** — multilingual
at both retrieval and generation. (MAW figure matches the corpus snapshot.)

### Q4 — Out-of-domain (fallback)
> **Q:** What is the capital of France?
>
> **A:** I'm sorry, I don't have enough information in my Hong Kong labour
> regulation sources to answer that confidently. For accurate guidance, please
> contact the Hong Kong Labour Department …
>
> *(no sources; LLM not called)*

The confidence gate fired (confidence 0.03) — no hallucination, no misleading
citations.

### Q5 — Domestic worker rights (English) — see failure case below
> **Q:** What are the rights of domestic workers in Hong Kong?
>
> **A:** … refers the user to the *"Foreign domestic helpers' rights and
> protection under the Employment Ordinance"* publication rather than
> enumerating the concrete rights.

## 3. Limitations and failure cases

### Failure case A — "meta-answers" when a topic is fragmented across sources
For the broad question *"What are the rights of domestic workers?"*, the top
retrieved chunks are **pamphlet reference lists** in the Code of Practice that
literally contain the phrase "domestic helpers' rights," rather than the passages
that actually enumerate rights (rest days, MAW, medical, accommodation) — which
are scattered across the FDH guide, standard contract, and FAQ. The LLM therefore
produces a correct-but-unsatisfying *meta-answer* ("refer to this publication").
This is a genuine RAG weakness: **broad queries whose answer is distributed across
many chunks retrieve worse than narrow, factual queries.**
*Mitigations (future work):* query expansion/decomposition, larger `top_k` with
map-reduce summarization, or a curated FAQ document.

### Failure case B — bi-encoder cosine is not a usable confidence signal
The initial design gated the fallback on the e5 **cosine** score. We discovered
empirically that `multilingual-e5` has a **compressed cosine range**: an
out-of-domain query ("capital of France") still scored **0.75**, versus ~0.85 for
real answers — too close for any fixed threshold. We switched the confidence gate
to the **cross-encoder score** (via a sigmoid → probability), which separates
cleanly: **0.03** for out-of-domain vs **0.94+** for real answers. This is why
the fallback now works; it's documented as both a limitation we hit and the fix.

### Other limitations
- **No hybrid (lexical) search** — exact terms like statute section numbers rely on
  the embedding model; BM25 + dense hybrid would help precise lookups.
- **Corpus recency** — figures like the MAW/SMW are as of the downloaded document
  snapshot; there is no live update mechanism.
- **Answer quality is not auto-scored** — we measure retrieval quantitatively but
  judge answer quality manually; an LLM-as-judge (faithfulness) pass is future work.
- **Chunking is size-based**, not structure-aware; section-aware chunking could keep
  related provisions together and reduce Failure case A.
