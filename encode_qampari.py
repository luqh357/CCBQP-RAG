from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./datasets/qampari_data/dev_data.jsonl")
    parser.add_argument("--index", type=str, default="wikipedia-dpr-100w")
    parser.add_argument("--embedder", type=str, default="./models/bge-m3")
    parser.add_argument("--output", type=str, default="./data/wiki_qampari_embeddings.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--k", type=int, default=3000)
    return parser.parse_args()

def main():
    args = parse_args()

    searcher = LuceneSearcher.from_prebuilt_index(args.index)

    dataset = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))


    corpus = {}

    for item in tqdm(dataset):
        question = item["question_text"]

        hits = searcher.search(question, k=args.k)
        for hit in hits:
            entry = json.loads(hit.lucene_document.get("raw"))
            doc_id = entry["id"]
            contents = entry["contents"]
            
            if doc_id not in corpus:
                corpus[doc_id] = contents
        
    ids = list(corpus.keys())
    chunks = list(corpus.values())

    embedder = SentenceTransformer(args.embedder, device=args.device)

    E = embedder.encode(
        chunks, 
        normalize_embeddings=True, 
        convert_to_tensor=True, 
        device=args.device, 
        batch_size=args.batch_size, 
        show_progress_bar=True
    )

    torch.save({"ids": ids, "chunks": chunks, "embeddings": E.cpu()}, args.output)

if __name__ == "__main__":
    main()
