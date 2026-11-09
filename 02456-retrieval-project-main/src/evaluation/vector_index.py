"""
Retrieval pipeline skeleton: metrics + interchangeable retrievers.

    retriever.fit(doc_ids, texts)
    retriever.search(queries, k)   # queries: {qid: text} -> {qid: [doc_id, ...]}
                                   # best first, length <= k

Data conventions
----------------
    doc_ids : list of document ids
    texts   : list of passage texts, same order as doc_ids
    queries : {qid: query text}
    qrels   : {qid: set of relevant doc_ids}   (from FiQA / SQuAD annotations)
"""

import torch
 
 
class VectorIndex:
    def __init__(self, embeddings, doc_ids, device="cpu"):
        """
        embeddings: tensor [N, d], row i is the embedding of doc_ids[i]
        doc_ids:    list of N document ids
        """
        if embeddings.shape[0] != len(doc_ids):
            raise ValueError(
                f"{embeddings.shape[0]} embeddings but {len(doc_ids)} doc_ids"
            )
        self.device = device
        self.embeddings = embeddings.detach().float().to(device)
        self.doc_ids = list(doc_ids)
 
    def __len__(self):
        return len(self.doc_ids)
 
    @classmethod
    def from_corpus(cls, encoder, texts, doc_ids, batch_size=64, device="cpu"):
        """Encode the whole corpus in batches (no gradients) and build the index."""
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                emb = encoder.encode(texts[i : i + batch_size])
                chunks.append(emb.detach().float().cpu())
        return cls(torch.cat(chunks, dim=0), doc_ids, device=device)
 
    def search(self, query_emb, k=10, query_batch_size=256):
        """
        query_emb: tensor [Q, d] (or [d] for a single query)
        Returns a list of Q lists with the top-k doc_ids, best first.
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        k = min(k, len(self))
 
        ranked = []
        with torch.no_grad():
            for i in range(0, query_emb.shape[0], query_batch_size):
                q = query_emb[i : i + query_batch_size].float().to(self.device)
                scores = q @ self.embeddings.T  # [q, N] inner product
                top_idx = scores.topk(k, dim=1).indices
                ranked.extend(
                    [self.doc_ids[j] for j in row] for row in top_idx.tolist()
                )
        return ranked
 
    def search_texts(self, encoder, queries, k=10, **kwargs):
        """Encode raw query texts with the query encoder, then search."""
        with torch.no_grad():
            q = encoder.encode(queries)
        return self.search(q, k=k, **kwargs)
 
    def save(self, path):
        torch.save(
            {"embeddings": self.embeddings.cpu(), "doc_ids": self.doc_ids}, path
        )
 
    @classmethod
    def load(cls, path, device="cpu"):
        data = torch.load(path, map_location="cpu")
        return cls(data["embeddings"], data["doc_ids"], device=device)
 