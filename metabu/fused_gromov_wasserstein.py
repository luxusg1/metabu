import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

from metabu.prae import fgw
from metabu.utils import intrinsic_estimator


def remove_outlier(labels, datasets_has_priors, cost_matrix):
    idx, counts = np.unique(labels, return_counts=True)
    outlier = [i for i, c in enumerate(labels) if counts[c] <= 1]
    return np.array([_ for i, _ in enumerate(datasets_has_priors) if i not in outlier]), \
           np.array([_ for i, _ in enumerate(labels) if i not in outlier]), \
           cost_matrix[~np.in1d(np.arange(len(cost_matrix)), outlier), :][:,
           ~np.in1d(np.arange(len(cost_matrix)), outlier)]


def train_fused_gromov_wasserstein(basic_representation_train,
                                   basic_representation_test,
                                   cost_matrix,
                                   lr=0.001, seed=42,
                                   early_stopping=20,
                                   **kwargs):
    print("Train embedding with Fused-Gromov-Wasserstein loss")

    x_train, idx_train = basic_representation_train

    l1 = kwargs.get("l1", 0.001)
    alpha = kwargs.get("alpha", 0.5)
    print(kwargs, l1, alpha)

    task_has_target_representation_used_for_train = kwargs["task_has_target_representation_used_for_train"]

    N_old = len(task_has_target_representation_used_for_train)
    cluster = AgglomerativeClustering(affinity="precomputed", linkage="complete", n_clusters=5)
    cluster.fit(cost_matrix)
    task_has_target_representation_used_for_train, labels_used, cost_matrix_used = \
        remove_outlier(cluster.labels_,
                       task_has_target_representation_used_for_train,
                       cost_matrix)

    N = len(task_has_target_representation_used_for_train)
    dim_in = x_train.shape[1]
    dim_out = intrinsic_estimator(cost_matrix_used)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embedding = MDS(n_components=dim_out, random_state=seed, dissimilarity="precomputed")
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(x_train.values)
    X_true = embedding.fit_transform(cost_matrix_used)

    xs = torch.from_numpy(scaler.transform(x_train.values)).float().to(device)
    xt = torch.from_numpy(X_true).float().to(device)

    x_test = torch.from_numpy(scaler.transform(basic_representation_test.values)).to(device).float()

    # M /= M.sum(0)

    M = np.zeros((N, N)) / N
    M[cost_matrix_used.argsort().argsort() == 0] = 1
    M = torch.from_numpy(M).to(device).float()

    model = torch.nn.Linear(dim_in, dim_out, bias=False).to(device).float()

    optimizer = torch.optim.Adam(model.parameters())
    best_loss = 9999
    no_improvement = 0

    for i in range(100000):
        optimizer.zero_grad()
        x = xs[[np.random.choice(idx_train[id]) for id in task_has_target_representation_used_for_train]]
        x_new = model(x)
        loss = fgw(source=x_new, target=xt, cost_source_target=None, device=device, beta=alpha,
                   M=M) + l1 * torch.norm(model.weight, 1)
        loss.backward()
        optimizer.step()

        if best_loss > loss.item():
            best_loss = loss.item()
            no_improvement = 0
        elif no_improvement >= early_stopping:
            break
        no_improvement += 1
        print(f"Epoch test {i}: ", loss.item())

    return model, model(xs[:N_old]).cpu().detach().numpy(), model(x_test).cpu().detach().numpy()
