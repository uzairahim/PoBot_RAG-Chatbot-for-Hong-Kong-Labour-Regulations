# Retrieval Evaluation

Gold set: **20 questions** · cut-off **k=5** · document-level relevance.
Embedding: `intfloat/multilingual-e5-small` · Re-ranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

| Metric | Bi-encoder only | + Cross-encoder rerank | Δ |
|---|--:|--:|--:|
| Hit@5 | 0.950 | 1.000 | +0.050 |
| MRR | 0.850 | 0.846 | -0.004 |
| nDCG@5 | 0.869 | 0.889 | +0.020 |

**Hit@k** = fraction of questions with a correct-source chunk in the top-k.  
**MRR** = mean reciprocal rank of the first correct-source chunk.  
**nDCG@k** = ranking quality (rewards correct chunks appearing higher).