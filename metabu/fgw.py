import copy
import logging as log

import numpy as np
import torch
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from metabu.prae import fgw, distance_matrix
from metabu.utils import get_ndcg_score


def train_fused_gromov_wasserstein(basic_representations,
                                   pairwise_dist_z,
                                   learning_rate,
                                   seed,
                                   intrinsic_dim,
                                   early_stopping,
                                   early_stopping_criterion_ndcg,
                                   alpha,
                                   lambda_reg,
                                   device,
                                   list_ids):
    # list_ids = list(basic_representations.index.unique())
    id_reprs = {id: np.where(basic_representations.index == id)[0] for id in list_ids}

    torch.manual_seed(seed)
    m = len(list_ids)
    dim_in = basic_representations.shape[1]

    # Compute MDS
    intrinsic_dim = 2
    mds = MDS(n_components=intrinsic_dim, random_state=seed, dissimilarity="precomputed")
    U_ = StandardScaler().fit_transform(mds.fit_transform(pairwise_dist_z))

    # Learn metabu meta-features
    X = torch.from_numpy(basic_representations.values).float().to(device)
    U = torch.from_numpy(U_).float().to(device)
    M = np.zeros((m, m)) / m
    M[pairwise_dist_z.argsort().argsort() == 0] = 1
    M = torch.from_numpy(M).to(device).float()

    model = torch.nn.Linear(dim_in, intrinsic_dim, bias=False).to(device).float()
        # torch.nn.Sequential(
        #             torch.nn.Linear(dim_in, 512),
        #             torch.nn.ReLU(),
        #             torch.nn.Linear(512, 25),
        #             torch.nn.ReLU(),
        #             torch.nn.Linear(25, intrinsic_dim, bias=False)
        # ).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_ids, valid_ids = train_test_split(list(range(len(list_ids))), test_size=0.5, random_state=seed)

    i = 0
    best_i = i
    best_ndcg = 0
    best_model = copy.deepcopy(model)
    no_improvement = 0

    import neptune.new as neptune
    import matplotlib.pyplot as plt

    def plot_scatter(x, label):
        fig, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1])

        for i, txt in enumerate(label):
            ax.annotate(str(txt), (x[i, 0], x[i, 1]))
        return fig

    run = neptune.init(project='herilalaina/metabu', api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZjdjN2E0NS1lMzgwLTRhYWItODg5Mi1jZjg5N2E4NDhlMWUifQ==")

    p = plot_scatter(U_, list_ids)
    run["target_plot"].upload(neptune.types.File.as_image(p))

    while no_improvement <= early_stopping:
        optimizer.zero_grad()

        x_train = X[[np.random.choice(id_reprs[list_ids[_]]) for _ in train_ids+valid_ids]]
        U_train = U[train_ids+valid_ids]

        assert not torch.isnan(x_train).any()
        assert not torch.isnan(U_train).any()
        assert not torch.isnan(M).any()

        # loss = torch.nn.functional.mse_loss(model(x_train), U_train)
        loss = fgw(source=model(x_train),
                   target=U_train,
                   device=device,
                   alpha=alpha,
                   M=M) + lambda_reg * torch.norm(model.weight, 1)
        assert not torch.isnan(loss).any()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_train = X[[np.random.choice(id_reprs[list_ids[id]])for id in train_ids]]
            x_valid = X[[np.random.choice(id_reprs[list_ids[id]]) for id in valid_ids]]
            U_pred = distance_matrix(pts_src=model(x_valid), pts_dst=model(x_train))
            U_pred_train = distance_matrix(pts_src=model(x_train), pts_dst=model(x_train))

            oracle=distance_matrix(pts_src=U, pts_dst=U)
            dist_valid = oracle[valid_ids][:, train_ids]
            dist_train = oracle[train_ids][:, train_ids]
            valid_ndcg_score = get_ndcg_score(dist_pred=U_pred.detach().cpu().numpy(),
                                              dist_true=dist_valid.detach().cpu().numpy(),
                                              k=early_stopping_criterion_ndcg)
            train_ndcg_score = get_ndcg_score(dist_pred=U_pred_train.detach().cpu().numpy(),
                                              dist_true=dist_train.detach().cpu().numpy(),
                                              k=early_stopping_criterion_ndcg)

            pred_x_train = model(x_train).detach().cpu().numpy()
            pred_x_valid = model(x_valid).detach().cpu().numpy()

            run["pred_plot/image_train"].log(neptune.types.File.as_image(plot_scatter(np.concatenate([pred_x_train, pred_x_valid], axis=0), [list_ids[_] for _ in train_ids + valid_ids])))

        if best_ndcg < train_ndcg_score:
            best_i = i
            best_ndcg = train_ndcg_score
            best_model = copy.deepcopy(model)
            no_improvement = 0
        else:
            no_improvement += 1

        loss = loss.item()
        log.info("Epoch {}; train loss: {:.2f}; NDCG(valid/train): {:.2f}/{:.2f}".format(i, loss, valid_ndcg_score,
                                                                                         train_ndcg_score))
        i += 1

    log.info("Total epoch: {} -- [BEST NDCG valid: {:.2f} (Epoch: {})]".format(i, best_ndcg, best_i))

    return best_model, mds
