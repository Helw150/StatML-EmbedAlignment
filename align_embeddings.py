import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, BatchSampler

import cvxpy as cvx


class Align:
    def __init__(self, src_emb, tgt_emb, lr, sinkhorn_steps=1000, reg=0.05):
        # center and normalize embeddings
        self.src_emb = src_emb.float().cuda()
        self.tgt_emb = tgt_emb.float().cuda()

        # initialize mapping layer
        self.mapping = nn.Linear(src_emb.size(1), tgt_emb.size(1), bias=False).cuda()
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
            grad = A.T.dot(Ax) - Atb

            # .. the LMO results in a vector that is zero everywhere except for ..
            # .. a single index. Of this vector we only store its index and magnitude ..
            idx_oracle = np.argmax(np.abs(grad))
            mag_oracle = alpha * np.sign(-grad[idx_oracle])
            g_t = x_t.T.dot(grad).ravel() - grad[idx_oracle] * mag_oracle
            trace.append(g_t)
            if g_t <= tol:
                break
            q_t = A[:, idx_oracle] * mag_oracle - Ax
            step_size = min(q_t.dot(b - Ax) / q_t.dot(q_t), 1.0)
            x_t = (1.0 - step_size) * x_t
            x_t[idx_oracle] = x_t[idx_oracle] + step_size * mag_oracle
        return x_t, np.array(trace)

    def convex_assignment(self, K_x, K_y):
        P = cvx.Variable(shape=K_x.shape)
        obj = cvx.Minimize(cvx.norm(cvx.matmul(K_x, P) - cvx.matmul(P, K_y), 2))
        consts = [cvx.sum(P, axis=0) == 1, cvx.sum(P, axis=1) == 1]
        prob = cvx.Problem(obj, consts)
        prob.solve(verbose=True)
        return torch.tensor(P.value).float().cuda()

    def initialize(self):
        self.mapping.weight.data = self.orthogonalize(self.mapping.weight.data)

        # todo
        if True:
            print("Initializing Q.")
            # solve OT
            # currently hardcoded for src = ES, and tgt = EN
            size = 100
            src_top_k = self.src_emb[
                [
                    int(index)
                    for index in open(
                        "most_common_words/spanish_index.txt", "r"
                    ).readlines()[:size]
                ]
            ]
            tgt_top_k = self.tgt_emb[
                [
                    int(index)
                    for index in open(
                        "most_common_words/english_index.txt", "r"
                    ).readlines()[:size]
                ]
            ]

            K_src = torch.matmul(src_top_k, src_top_k.T)
            K_tgt = torch.matmul(tgt_top_k, tgt_top_k.T)

            P = self.convex_assignment(K_src.cpu().numpy(), K_tgt.cpu().numpy())
            a = torch.linalg.multi_dot([src_top_k.T, P, tgt_top_k])
            u, s, v = torch.svd(a)
            self.mapping.weight.data = torch.matmul(u, v)

    @torch.no_grad()
    def sinkhorn(self, X, Y):
        # einsum
        C = torch.cdist(Y, X, p=2)
        # C = torch.sum(torch.pow(Y.view(Y.shape[0], 1, Y.shape[1]) - X.view(1, X.shape[0], X.shape[1]), 2), -1)
        K = torch.exp(-C / self.reg)

        a = torch.full(
            [Y.shape[0]], 1.0 / Y.shape[0], requires_grad=True
        ).cuda()  # .to("cpu")
        b = torch.full(
            [X.shape[0]], 1.0 / X.shape[0], requires_grad=True
        ).cuda()  # .to("cpu")
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
            G = -2 * torch.linalg.multi_dot([src_batch.T, P, tgt_batch])
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
            print(f"Training epoch {n_iter}")
            self.train_step(batch_size, num_steps)

            # double batch size for next step
            batch_size = min(2 * batch_size, min(len(self.src_emb), len(self.tgt_emb)))
            num_steps = num_steps // 2

    def retrieve(self):
        mapped_src_emb = self.mapping(self.src_emb)
        mapped_src_emb = F.normalize(mapped_src_emb, dim=1, p=2)
        tgt_emb = F.normalize(self.tgt_emb, dim=1, p=2)
        retreived_pairs = get_candidates(
            mapped_src_emb, tgt_emb, knn=10, max_rank=15000
        )
        return retreived_pairs

    def procrustes(self, pairs):
        A = self.src_emb[pairs[:, 0]]
        B = self.tgt_emb[pairs[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cuda()
        u, s, v = torch.svd(M)
        self.mapping.weight.data = torch.matmul(u, v)

    def refine(self, num_steps=5):
        # training loop
        for n_iter in range(num_steps):
            print(f"Refinement epoch {n_iter}")
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
            distances = query[i : i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cuda())
        all_distances = torch.cat(all_distances)
        return all_distances.detach()

    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    if max_rank > 0:
        n_src = min(max_rank, n_src)

    # contextual dissimilarity measure
    # average distances to k nearest neighbors
    average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
    average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
    average_dist1 = average_dist1.type_as(emb1)
    average_dist2 = average_dist2.type_as(emb2)

    # for every source word
    for i in range(0, n_src, bs):

        # compute target words scores
        scores = emb2.mm(emb1[i : min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(
            average_dist1[i : min(n_src, i + bs)][:, None] + average_dist2[None, :]
        )
        best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

        # update scores / potential targets
        all_scores.append(best_scores.cuda())
        all_targets.append(best_targets.cuda())

    all_scores = torch.cat(all_scores, 0)
    all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat(
        [
            torch.arange(0, all_targets.size(0)).long().unsqueeze(1).cuda(),
            all_targets[:, 0].unsqueeze(1),
        ],
        1,
    )

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
    # Center and Normalizes
    src_mean = torch.mean(src_emb, axis=0)
    tgt_mean = torch.mean(tgt_emb, axis=0)
    src_emb = (src_emb - src_mean) + tgt_mean
    src_scale = torch.mean(torch.linalg.norm(src_emb, dim=1, ord=2))
    tgt_scale = torch.mean(torch.linalg.norm(tgt_emb, dim=1, ord=2))
    src_emb = (src_emb / src_scale) * tgt_scale
    trainer = Align(src_emb, tgt_emb, lr=1e-3)
    trainer.train()
    trainer.refine()
    aligned_src_emb = trainer.mapping(src_emb.float().cuda())
    return aligned_src_emb


en_embed = load_embeddings("embedding_files/en_embeddings.csv")
print(en_embed.shape)
es_embed = load_embeddings("embedding_files/es_embeddings.csv")
print(es_embed.shape)
print(abs(es_embed[1162] - en_embed[1109]).sum())
print(abs(es_embed[1653] - en_embed[1134]).sum())

aligned_es_embed = align_embeddings(es_embed, en_embed)
# Sanity Check on the Embedding Matrix
print(aligned_es_embed.shape)
out_np = aligned_es_embed.detach().cpu().numpy()
print(abs(out_np[1162] - en_embed[1109].numpy()).sum())
print(abs(out_np[1653] - en_embed[1134].numpy()).sum())

out_df = pd.DataFrame(out_np)
out_df.to_csv("embedding_files/aligned_es_embeddings.csv")
