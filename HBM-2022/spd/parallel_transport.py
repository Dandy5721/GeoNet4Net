import torch
from tqdm import tqdm
import scipy.linalg as la


def log(X):
    S, U = torch.linalg.eigh(X)
    S = S.log().diag_embed()
    return U @ S @ U.transpose(-2, -1)


def exp(X):
    S, U = torch.linalg.eigh(X)
    S = S.exp().diag_embed()
    return U @ S @ U.transpose(-2, -1)


def sqrtm(X):
    S, U = torch.linalg.eigh(X)
    S = S.sqrt().diag_embed()
    return U @ S @ U.transpose(-2, -1)


def inv_sqrtm(X):
    S, U = torch.linalg.eigh(X)
    S = S.sqrt().reciprocal().diag_embed()
    return U @ S @ U.transpose(-2, -1)


def power(X, exponent):
    S, U = torch.linalg.eigh(X)
    S = S.pow(exponent).diag_embed()
    return U @ S @ U.transpose(-2, -1)


def geodesic(x, y, t):
    x_sqrt = torch.linalg.cholesky(x)
    x_sqrt_inv = x_sqrt.inverse()
    return (
        x_sqrt
        @ power(x_sqrt_inv @ y @ x_sqrt_inv.transpose(-2, -1), t)
        @ x_sqrt.transpose(-2, -1)
    )


def logm(x, y):
    c = torch.linalg.cholesky(x)
    c_inv = c.inverse()
    return c @ log(c_inv @ y @ c_inv.transpose(-2, -1)) @ c.transpose(-2, -1)


def expm(x, y):
    c = torch.linalg.cholesky(x)
    c_inv = c.inverse()
    return c @ exp(c_inv @ y @ c_inv.transpose(-2, -1)) @ c.transpose(-2, -1)


def riemannian_mean(spds, num_iter=20, eps_thresh=1e-4):
    mean = spds.mean(dim=-3).unsqueeze(-3)
    for iter in range(num_iter):
        tangent_mean = logm(mean, spds).mean(dim=-3).unsqueeze(-3)
        mean = expm(mean, tangent_mean)
        eps = tangent_mean.norm(p='fro', dim=(-2, -1)).mean()
        if eps < eps_thresh:
            break
    return mean.squeeze(-3)


def parallel_transport(all_spds1, all_spds2):
    mean_spds = []
    bar = tqdm(enumerate(all_spds1), desc='cal riemannian mean')
    for i, spds in bar:
        mean_spds.append(riemannian_mean(spds))
    mean_spds = torch.stack(mean_spds)

    final_mean = riemannian_mean(mean_spds)

    for i, spds in enumerate(all_spds2):
        E = torch.from_numpy(la.sqrtm(final_mean @ mean_spds[i].inverse()).real)
        yield E @ spds @ E.T
