"""
Code for: Principled and Scalable Diversity-Aware Retrieval via Cardinality-Constrained Binary Quadratic Programming

This file provides the minimal, highly optimized PyTorch implementations of the retrieval algorithms used in this paper. 

Contents:
1. `fw_qp_rank`: The proposed Frank-Wolfe algorithm with exact line search (Algorithm 1).
2. `mmr_rank`: Vectorized maximal marginal relevance with O(n) cache updates.
3. `greedy_dpp_rank`: Greedy determinantal point process with Cholesky decomposition.
"""
import torch

class DiversityRetriever:
    def __init__(self, device):
        self.device = device
        self.E = None
        self.chunks = None
        self.ids = None

    def load_path(self, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.ids = data["ids"]
        self.chunks = data["chunks"]
        E = data["embeddings"]
        self.E = E.to(self.device)

    def mmr_rank(self, c, k, theta):
        n = c.shape[0]

        selected = []
        remaining_mask = torch.ones(n, dtype=torch.bool, device=self.device)

        first_idx = c.argmax().item()
        selected.append(first_idx)
        remaining_mask[first_idx] = False

        max_sim_to_selected = self.E @ self.E[first_idx]

        for t in range(k - 1):
            scores = theta * c - (1 - theta) * max_sim_to_selected
            scores[~remaining_mask] = -torch.inf

            best_idx = scores.argmax().item()
            selected.append(best_idx)
            remaining_mask[best_idx] = False

            sim_to_new = self.E @ self.E[best_idx]

            max_sim_to_selected = torch.maximum(max_sim_to_selected, sim_to_new)

        return selected
    
    def greedy_dpp_rank(self, c, k, theta):
        eps = 1e-10
        n = c.shape[0]

        d2 = torch.ones(n, dtype=torch.float32, device=self.device)

        C = torch.zeros((n, k), dtype=torch.float32, device=self.device)

        selected = []
        remaining_mask = torch.ones(n, dtype=torch.bool, device=self.device)

        for t in range(k):
            scores = theta * c + (1 - theta) * torch.log(d2 + eps)
            scores[~remaining_mask] = -torch.inf

            j = scores.argmax().item()
            selected.append(j)
            remaining_mask[j] = False

            if t == k - 1:
                break

            d_j = d2[j].sqrt().clamp(min=eps)

            s_j = self.E @ self.E[j]
            
            if t == 0:
                dot_j = torch.zeros(n, device=self.device)
            else:
                dot_j = C[:, :t] @ C[j, :t]

            e = (s_j - dot_j) / d_j
            C[remaining_mask, t] = e[remaining_mask]
            d2[remaining_mask] = (d2[remaining_mask] - e[remaining_mask] ** 2).clamp(min=eps)

        return selected
    
    def fw_qp_rank(self, c, k, theta, max_iter=200):
        # max theta * (k - 1) * c^T x + (1 - theta) * x^T (I - EE^T) x
        # s.t. sum(x) = k, x in {0,1}^n
        n = c.shape[0]

        x = torch.full((n,), k / n, dtype=torch.float32, device=self.device)
        Et_x = (k / n) * self.E.sum(dim=0)

        for t in range(max_iter):
            grad = theta * (k - 1) * c + 2 * (1 - theta) * (2 * x - self.E @ Et_x)

            s = torch.zeros(n, dtype=torch.float32, device=self.device)
            _, idx = torch.topk(grad, k)
            s[idx] = 1

            d = s - x
            fw_gap = grad @ d
            if fw_gap == 0:
                break

            Et_s = self.E[idx].sum(dim=0)
            Et_d = Et_s - Et_x

            denom = (2 * (1 - theta) * (2 * (d @ d) - (Et_d @ Et_d))).item()

            if denom >= 0:
                gamma = 1.0
            else:
                gamma = min(1.0, (fw_gap / (-denom)))
            
            x += gamma * d
            Et_x += gamma * Et_d

        _, idx = torch.topk(x, k)

        return idx.tolist()