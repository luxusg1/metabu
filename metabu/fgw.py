import copy
import logging as log

import numpy as np
import torch
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from metabu.prae import fgw, distance_matrix
from metabu.utils import get_ndcg_score


def train_fused_gromov_wasserstein(
    basic_representations,
    pairwise_dist_z,
    learning_rate,
    seed,
    intrinsic_dim,
    early_stopping,
    early_stopping_criterion_ndcg,
    alpha,
    lambda_reg,
    device,
    list_ids,
):
    id_reprs = {id: np.where(basic_representations.index == id)[0] for id in list_ids}

    torch.manual_seed(seed)
    m = len(list_ids)
    dim_in = basic_representations.shape[1]

    # Compute MDS
    mds = MDS(
        n_components=intrinsic_dim, random_state=seed, dissimilarity="precomputed"
    )
    U_ = StandardScaler().fit_transform(mds.fit_transform(pairwise_dist_z))

    # Learn metabu meta-features
    X = torch.from_numpy(basic_representations.values).float().to(device)
    U = torch.from_numpy(U_).float().to(device)
    M = np.zeros((m, m)) / m
    M[pairwise_dist_z.argsort().argsort() == 0] = 1
    M = torch.from_numpy(M).to(device).float()

    model = torch.nn.Linear(dim_in, intrinsic_dim, bias=False).to(device).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ids = range(len(list_ids))

    i = 0
    best_i = i
    best_ndcg = 0
    best_model = copy.deepcopy(model)
    no_improvement = 0

    while no_improvement <= early_stopping:

        optimizer.zero_grad()

        x_train = X[[np.random.choice(id_reprs[list_ids[_]]) for _ in ids]]
        U_train = U[ids]

        assert not torch.isnan(x_train).any()
        assert not torch.isnan(U_train).any()
        assert not torch.isnan(M).any()

        loss = fgw(
            source=model(x_train), target=U_train, device=device, alpha=alpha, M=M
        ) + lambda_reg * torch.norm(model.weight, 1)
        assert not torch.isnan(loss).any()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_train = X[[np.random.choice(id_reprs[list_ids[id]]) for id in ids]]
            U_pred_train = distance_matrix(
                pts_src=model(x_train), pts_dst=model(x_train)
            )

            dist_train = distance_matrix(pts_src=U, pts_dst=U)

            train_ndcg_score = get_ndcg_score(
                dist_pred=U_pred_train.detach().cpu().numpy(),
                dist_true=dist_train.detach().cpu().numpy(),
                k=early_stopping_criterion_ndcg,
            )

        if best_ndcg < train_ndcg_score:
            best_i = i
            best_ndcg = train_ndcg_score
            best_model = copy.deepcopy(model)
            no_improvement = 0
        else:
            no_improvement += 1

        loss = loss.item()
        log.info(
            "Epoch {}; train loss: {:.2f}; NDCG@{}: {:.2f}".format(
                i, loss, early_stopping_criterion_ndcg, train_ndcg_score
            )
        )
        i += 1

    log.info(
        "Total epoch: {} -- [BEST NDCG@{}: {:.2f} (Epoch: {})]".format(
            i, early_stopping_criterion_ndcg, best_ndcg, best_i
        )
    )

    return best_model, mds
