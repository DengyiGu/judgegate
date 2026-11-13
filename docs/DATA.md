# Data and model attribution

The experiment uses the test splits from the [BEIR project](https://github.com/beir-cellar/beir).
Archive URLs, MD5 checks, SHA256 hashes, model commits and counts are recorded in the config,
summary and provenance files. Accessed 2026-11-13. No raw text, embeddings, training examples
or model weights are distributed in this repository.

- [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/): Boteva et al.'s medical
  retrieval collection, 3,633 documents, 323 test queries, 12,334 judgments. The original
  collection is for academic use and points to NutritionFacts terms for underlying content.
  Obtain data from the source and follow its terms; the code's MIT license does not license it.
- [TREC-COVID](https://ir.nist.gov/trec-covid/): biomedical retrieval judgments associated with
  CORD-19. The BEIR snapshot has 171,332 documents, 50 queries and 66,336 judgments. Two
  negative-grade rows are clamped to zero in this experiment. Underlying article rights vary;
  obtain and use them under their source terms.
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507): Apache-2.0;
  local relevance judge, unchanged pretrained weights, bfloat16 inference. No fine-tuning.
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5): MIT;
  dense encoder with its retrieval query prefix.
- [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2): MIT;
  dense encoder with query/passage prefixes.
- [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2):
  Apache-2.0; reranks the top 50 BM25 candidates.

Retrieval uses title plus text, 256-token model limits, cosine similarity for dense models,
BM25 k1=0.9/b=0.4 with sklearn English tokenization, and RRF constant 60. Two deliberately
degraded bge variants use sign vectors and 32-token passages. They are labeled ablations,
not independently trained systems. The judge sees at most 250 whitespace-separated words.
This rubric and truncation need not match the original human judging instructions.
