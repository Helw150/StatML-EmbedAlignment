import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, BatchSampler


class Align:
    def __init__(self, src_emb, tgt_emb, lr, sinkhorn_steps=1000, reg=0.05):
        # center and normalize embeddings
        self.src_emb = src_emb.float()
        self.tgt_emb = tgt_emb.float()

        # initialize mapping layer
        self.mapping = nn.Linear(src_emb.size(1), tgt_emb.size(1), bias=False)
        self.sinkhorn_steps = sinkhorn_steps
        self.reg = reg
        self.lr = lr

    def center_and_normalize(self, emb):
        emb = emb - torch.mean(emb, dim=1, keepdim=True)
        emb = F.normalize(emb, dim=1, p=2)
        return emb

    def frank_wolfe(self, alpha, max_iter=200, tol=1e-8):
        # .. initial estimate, could be any feasible point ..
        x_t = sparse.dok_matrix((n_features, 1))
        trace = []  # to keep track of the gap

        # .. some quantities can be precomputed ..
        Atb = A.T.dot(b)
        for it in range(max_iter):
            # .. compute gradient. Slightly more involved than usual because ..
            # .. of the use of sparse matrices ..
            Ax = x_t.T.dot(A.T).ravel()
            grad = (A.T.dot(Ax) - Atb)

            # .. the LMO results in a vector that is zero everywhere except for ..
            # .. a single index. Of this vector we only store its index and magnitude ..
            idx_oracle = np.argmax(np.abs(grad))
            mag_oracle = alpha * np.sign(-grad[idx_oracle])
            g_t = x_t.T.dot(grad).ravel() - grad[idx_oracle] * mag_oracle
            trace.append(g_t)
            if g_t <= tol:
                break
            q_t = A[:, idx_oracle] * mag_oracle - Ax
            step_size = min(q_t.dot(b - Ax) / q_t.dot(q_t), 1.)
            x_t = (1. - step_size) * x_t
            x_t[idx_oracle] = x_t[idx_oracle] + step_size * mag_oracle
        return x_t, np.array(trace)

    def initialize(self):
        # todo
        if False:
            print('Initializing Q.')
            # solve OT
            # currently hardcoded for src = ES, and tgt = EN
            src_top_k = self.src_emb[1009:3509]
            tgt_top_k = self.tgt_emb[1997:4497]

            K_src = torch.matmul(src_top_k, src_top_k.T)
            K_tgt = torch.matmul(tgt_top_k, tgt_top_k.T)

            P = ...
            a = torch.linalg.multi_dot([src_top_k.T, P, tgt_top_k])
            u, s, v = torch.svd(a)
            self.mapping.weight.data = torch.matmul(u, v)
    @torch.no_grad()
    def sinkhorn(self, X, Y):
        # einsum
        C = torch.cdist(Y, X, p=2)
        # C = torch.sum(torch.pow(Y.view(Y.shape[0], 1, Y.shape[1]) - X.view(1, X.shape[0], X.shape[1]), 2), -1)
        K = torch.exp(-C / self.reg)

        a = torch.full([Y.shape[0]], 1. / Y.shape[0], requires_grad=True).to('cpu')
        b = torch.full([X.shape[0]], 1. / X.shape[0], requires_grad=True).to('cpu')
        lefts = [torch.ones_like(a)]
        rights = []

        for i in range(self.sinkhorn_steps):
            rights += [b / torch.matmul(lefts[i - 1], K)]
            lefts += [a / torch.matmul(K, rights[i])]

        P = lefts[-1].view(-1, 1) * K * rights[-1].view(1, -1)
        return P.detach()

    @torch.no_grad()
    def orthogonalize(self, a):
        # svd
        u, s, v = torch.svd(a)
        return torch.matmul(u, v)

    def train_step(self, batch_size, num_steps):
        src_sampler = SubsetRandomSampler(torch.arange(self.src_emb.size(0)))
        src_sampler = BatchSampler(src_sampler, batch_size, True)
        tgt_sampler = SubsetRandomSampler(torch.arange(self.tgt_emb.size(0)))
        tgt_sampler = BatchSampler(tgt_sampler, batch_size, True)

        src_iter = iter(src_sampler)
        tgt_iter = iter(tgt_sampler)
        for n_iter in tqdm(range(num_steps)):
            # sample a batch from each domain
            src_indices = next(src_iter, None)
            if src_indices is None:
                src_iter = iter(src_sampler)
                src_indices = next(src_iter, None)

            tgt_indices = next(tgt_iter, None)
            if tgt_indices is None:
                tgt_iter = iter(tgt_sampler)
                tgt_indices = next(tgt_iter, None)

            src_batch = self.src_emb[src_indices]
            tgt_batch = self.tgt_emb[tgt_indices]

            # apply mapping to src embeddings
            mapped_src_batch = self.mapping(src_batch)

            # apply sinkhorn to get transport plan
            P = self.sinkhorn(mapped_src_batch, tgt_batch)

            # perform a gradient step
            G = - 2 * torch.linalg.multi_dot([src_batch.T, P, tgt_batch])
            M = self.mapping.weight.data - self.lr * G

            # project on the set of orthogonal matrices
            M = self.orthogonalize(M)

            # update mapping weights
            self.mapping.weight.data = M

    def train(self):
        # initialize Q by using the convex relaxation
        self.initialize()

        batch_size = 500
        num_steps = 100
        for n_iter in range(5):
            print(f'Training epoch {n_iter}')
            self.train_step(batch_size, num_steps)

            # double batch size for next step
            batch_size = 2 * batch_size
            num_steps = num_steps // 2

    def retrieve(self):
        mapped_src_emb = self.mapping(self.src_emb)
        mapped_src_emb = F.normalize(mapped_src_emb, dim=1, p=2)
        tgt_emb = F.normalize(self.tgt_emb, dim=1, p=2)
        retreived_pairs = get_candidates(mapped_src_emb, tgt_emb, knn=10, max_rank=15000)
        return retreived_pairs
    def procrustes(self, pairs):
        A = self.src_emb[pairs[:, 0]]
        B = self.tgt_emb[pairs[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu()
        u, s, v = torch.svd(M)
        self.mapping.weight.data = torch.matmul(u, v)

    def refine(self, num_steps=5):
        # training loop
        for n_iter in range(num_steps):
            print(f'Refinement epoch {n_iter}')
            # build a dictionary from aligned embeddings
            retreived_pairs = self.retrieve()

            # apply the Procrustes solution
            self.procrustes(retreived_pairs)


def get_candidates(emb1, emb2, knn, max_rank=0):
    """
    Get best translation pairs candidates.
    Function taken from MUSE repo.
    """

    def get_nn_avg_dist(emb, query, knn):
        """
        Compute the average distance of the `knn` nearest neighbors
        for a given set of embeddings and queries.
        """
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.detach().numpy()

    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    if max_rank > 0:
        n_src = min(max_rank, n_src)

    # contextual dissimilarity measure
    # average distances to k nearest neighbors
    average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
    average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
    average_dist1 = average_dist1.type_as(emb1)
    average_dist2 = average_dist2.type_as(emb2)

    # for every source word
    for i in range(0, n_src, bs):

        # compute target words scores
        scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
        best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

        # update scores / potential targets
        all_scores.append(best_scores.cpu())
        all_targets.append(best_targets.cpu())

    all_scores = torch.cat(all_scores, 0)
    all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
    ], 1)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if max_rank > 0:
        selected = all_pairs.max(1)[0] <= max_rank
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    return all_pairs


def load_embeddings(filename):
    df = pd.read_csv(filename, sep=",")
    # Remove Column Numbers
    np_array = df.values[:, 1:]
    return torch.tensor(np_array)


def align_embeddings(src_emb, tgt_emb):
    trainer = Align(src_emb, tgt_emb, lr=1e-3)
    trainer.train()
    trainer.refine()
    aligned_src_emb = trainer.mapping(trainer.src_emb)
    return aligned_src_emb


en_embed = load_embeddings("embedding_files/en_embeddings.txt")
print(en_embed.shape)
es_embed = load_embeddings("embedding_files/es_embeddings.txt")
print(es_embed.shape)

aligned_es_embed = align_embeddings(es_embed, en_embed)
# Sanity Check on the Embedding Matrix
print(aligned_es_embed.shape)
out_np = aligned_es_embed.numpy()
out_df = pd.DataFrame(out_np)
out_df.to_csv("embedding_files/aligned_es_embeddings.txt")
