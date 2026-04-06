# [Principled and Scalable Diversity-Aware Retrieval via Cardinality-Constrained Binary Quadratic Programming](https://arxiv.org/abs/2604.02554)

## Overview

This repository provides PyTorch implementations of the diversity-aware retrieval algorithms from the paper, including the proposed Frank-Wolfe algorithm (FW-QP) and two baselines: Maximal Marginal Relevance (MMR) and Greedy Determinantal Point Process (DPP).

## Files

- `diversity_retriever.py` — Core retrieval algorithms (`fw_qp_rank`, `mmr_rank`, `greedy_dpp_rank`)
- `encode_qampari.py` — QAMPARI corpus embedding encoding 
- `eval_retrieval_qampari.py` — Evaluation on QAMPARI

## Requirements

```bash
pip install torch sentence-transformers pyserini datasets tqdm numpy
```

## Data Preparation

### QAMPARI

```bash
wget https://aggreg-qa.s3.amazonaws.com/qampari.zip
unzip qampari.zip -d datasets/
```

### Embedding Model

Download [BGE-M3](https://huggingface.co/BAAI/bge-m3) or any SentenceTransformer-compatible model.

## Usage

### 1. Encode Corpus

```bash
python encode_qampari.py \
    --dataset ./datasets/qampari_data/dev_data.jsonl \
    --embedder ./models/bge-m3 \
    --output ./data/wiki_qampari_embeddings.pt \
    --device cuda:0
```

### 2. Run Evaluation

```bash
python eval_retrieval_qampari.py \
    --dataset ./datasets/qampari_data/dev_data.jsonl \
    --embedder ./models/bge-m3 \
    --embeddings ./data/wiki_qampari_embeddings.pt \
    --output ./output/qampari \
    --device cuda:0 \
    --k 100
```

## Citation

```bibtex
@article{lu2026principled,
  title={Principled and Scalable Diversity-Aware Retrieval via Cardinality-Constrained Binary Quadratic Programming},
  author={Lu, Qiheng and Sidiropoulos, Nicholas D.},
  journal={arXiv preprint arXiv:2604.02554},
  year={2026}
}
```