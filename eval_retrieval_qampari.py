import torch
import numpy as np
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from diversity_retriever import DiversityRetriever
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./datasets/qampari_data/dev_data.jsonl")
    parser.add_argument("--embedder", type=str, default="./models/bge-m3")
    parser.add_argument("--embeddings", type=str, default="./data/wiki_qampari_embeddings.pt")
    parser.add_argument("--output", type=str, default="./output/qampari")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--k", type=int, default=100)
    return parser.parse_args()


def answer_recall(selected_texts, answer_list):
    total = 0
    hit = 0
    combined = " ".join(selected_texts).lower()
    for answer in answer_list:
        total += 1
        aliases = set([answer["answer_text"]] + answer["aliases"])
        if any(alias.lower() in combined for alias in aliases):
            hit += 1
    return hit / total if total > 0 else 0.0

def ilad(E_selected):
    k = E_selected.shape[0]
    if k < 2:
        return 0.0
    sim = (E_selected @ E_selected.t()).cpu().float()
    idx = torch.triu_indices(k, k, offset=1)
    mean_sim = sim[idx[0], idx[1]].mean().item()
    return 1.0 - mean_sim

def evaluate_retrieval_qampari(pipeline, embedder, k, dataset_path):
    thetas = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]

    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))

    methods = ["mmr", "greedy_dpp", "fw_qp"]

    results = {
        theta: {m: {"recall": [], "ilad": [], "time": []} for m in methods}
        for theta in thetas
    }

    records = []

    for entry in tqdm(dataset, desc="Evaluating"):
        question = entry["question_text"]
        answer_list = entry["answer_list"]

        query_emb = embedder.encode(
            question,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=pipeline.E.device
        )
        c = pipeline.E @ query_emb

        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)

        record = {
            "question": question,
            "mmr": {},
            "greedy_dpp": {},
            "fw_qp": {}
        }

        for theta in thetas:
            t_start.record()
            mmr_idx = pipeline.mmr_rank(c, k, theta)
            t_end.record()
            torch.cuda.synchronize()

            mmr_texts = [pipeline.chunks[i] for i in mmr_idx]
            results[theta]["mmr"]["recall"].append(answer_recall(mmr_texts, answer_list))
            results[theta]["mmr"]["ilad"].append(ilad(pipeline.E[mmr_idx]))
            results[theta]["mmr"]["time"].append(t_start.elapsed_time(t_end))
            record["mmr"][str(theta)] = mmr_texts

            t_start.record()
            dpp_idx = pipeline.greedy_dpp_rank(c, k, theta)
            t_end.record()
            torch.cuda.synchronize()

            dpp_texts = [pipeline.chunks[i] for i in dpp_idx]
            results[theta]["greedy_dpp"]["recall"].append(answer_recall(dpp_texts, answer_list))
            results[theta]["greedy_dpp"]["ilad"].append(ilad(pipeline.E[dpp_idx]))
            results[theta]["greedy_dpp"]["time"].append(t_start.elapsed_time(t_end))
            record["greedy_dpp"][str(theta)] = dpp_texts

            t_start.record()
            fw_idx = pipeline.fw_qp_rank(c, k, theta)
            t_end.record()
            torch.cuda.synchronize()

            fw_texts = [pipeline.chunks[i] for i in fw_idx]
            results[theta]["fw_qp"]["recall"].append(answer_recall(fw_texts, answer_list))
            results[theta]["fw_qp"]["ilad"].append(ilad(pipeline.E[fw_idx]))
            results[theta]["fw_qp"]["time"].append(t_start.elapsed_time(t_end))
            record["fw_qp"][str(theta)] = fw_texts

        records.append(record)

    summary = {}
    for theta in thetas:
        summary[theta] = {}
        for m in methods:
            r = results[theta][m]
            summary[theta][m] = {
                "recall": float(np.mean(r["recall"])),
                "ilad":   float(np.mean(r["ilad"])),
                "time_ms": float(np.mean(r["time"])),
            }

    return summary, records

def print_summary(summary):
    methods = ["mmr", "greedy_dpp", "fw_qp"]
    header  = f"{'theta':>6}  {'method':>12}  {'recall':>8}  {'ilad':>8}  {'time':>8}"
    print(header)
    print("-" * len(header))
    for theta in sorted(summary.keys()):
        for m in methods:
            s = summary[theta][m]
            print(
                f"{theta:>6.1f}  {m:>12}  "
                f"{s['recall']:>8.4f}  {s['ilad']:>8.4f}  {s['time_ms']:>8.2f}"
            )
        print()

if __name__=="__main__":
    args = parse_args()

    embedder = SentenceTransformer(args.embedder, device=args.device)

    pipeline = DiversityRetriever(args.device)
    pipeline.load_path(args.embeddings)

    summary, records = evaluate_retrieval_qampari(pipeline, embedder, args.k, args.dataset)
    with open(f"{args.output}_{args.k}.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print_summary(summary)