import torch
from torch import nn
from sklearn.cluster import SpectralClustering


class SPDSpectralClustering:
    def __init__(self, n_clusters, bandwidth, n_jobs=-1):
        self.n_clusters = n_clusters
        self.bandwidth = bandwidth
        self.n_jobs = n_jobs

    def fit(self, X):
        S, U = torch.linalg.eigh(X.detach())
        S = S.log().diag_embed()
        log_X = U @ S @ U.transpose(-2, -1)
        pair_dis = torch.norm(
            log_X.unsqueeze(-4) - log_X.unsqueeze(-3), p='fro', dim=(-2, -1)
        )
        pair_dis_square = pair_dis**2
        affinity = torch.exp(-0.5 * pair_dis_square / (self.bandwidth * self.bandwidth))
        clustering = SpectralClustering(
            affinity='precomputed', n_clusters=self.n_clusters, n_jobs=self.n_jobs
        )
        self.labels_ = clustering.fit(affinity).labels_
        return self


class SPD_GBMS_RNN(nn.Module):
    def __init__(self, bandwidth=0.5):
        super(SPD_GBMS_RNN, self).__init__()
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth))

    def log(self, X):
        S, U = torch.linalg.eigh(X)
        S = S.log().diag_embed()
        return U @ S @ U.transpose(-2, -1)

    def exp(self, X):
        S, U = torch.linalg.eigh(X)
        S = S.exp().diag_embed()
        return U @ S @ U.transpose(-2, -1)

    def logm(self, X, Y):
        return self.log(Y) - self.log(X)

    def expm(self, X, Y):
        return self.exp(self.log(X) + Y)

    def forward(self, X):
        X = X.squeeze()
        bandwidth = self.bandwidth
        log_X = self.log(X)
        pair_dis = torch.norm(
            log_X.unsqueeze(-4) - log_X.unsqueeze(-3) + 1e-7, p='fro', dim=(-2, -1)
        )
        log_Y_X = log_X.unsqueeze(-4) - log_X.unsqueeze(-3)
        pair_dis_square = pair_dis**2
        W = torch.exp(-0.5 * pair_dis_square / (bandwidth * bandwidth))
        D = W.sum(dim=-1).diag_embed()
        D_inv = D.inverse()

        M = (
            (log_Y_X.permute(2, 3, 0, 1) @ W).diagonal(dim1=-2, dim2=-1) @ D_inv
        ).permute(2, 0, 1)
        output = self.expm(X, M)

        return output


def distance_loss(inputs, targets, alpha=0):
    identity_matrix = targets.unsqueeze(0) == targets.unsqueeze(0).T

    S, U = torch.linalg.eigh(inputs)
    S = S.log().diag_embed()
    log_X = U @ S @ U.transpose(-2, -1)
    pair_dis = torch.norm(
        log_X.unsqueeze(-4) - log_X.unsqueeze(-3) + 1e-7, p='fro', dim=(-2, -1)
    )

    pair_dis_sim = 1 / (1 + pair_dis)

    loss = (1 - pair_dis_sim.cpu()) * identity_matrix.cpu() + (
        pair_dis_sim.cpu() - alpha
    ) * (~identity_matrix.cpu())
    loss = loss.mean()
    return loss
