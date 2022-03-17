import torch
from torch import nn
from torch import optim
#import Methods.evaluation as evaluation
#import Methods.models as method


class Prior(nn.Module):
    def __init__(self, data_size: list):
        super(Prior, self).__init__()
        self.data_size = data_size
        self.number_components = data_size[0]
        self.output_size = data_size[1]
        self.mu = nn.Parameter(torch.randn(data_size), requires_grad=True)
        self.logvar = nn.Parameter(torch.randn(data_size), requires_grad=True)

    def forward(self):
        return self.mu, self.logvar


def sum_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor):
    """
    Returns the matrix of "x_i + y_j".
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :return: [R, C, D] sum matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    return x_col + y_row


def prod_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor):
    """
    Returns the matrix of "x_i * y_j".
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :return: [R, C, D] sum matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    return x_col * y_row


def distance_tensor(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C, D] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.abs(x_col - y_row) ** p
    return distance


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def distance_gmm(mu_src: torch.Tensor, mu_dst: torch.Tensor, logvar_src: torch.Tensor, logvar_dst: torch.Tensor):
    """
    Calculate a Wasserstein distance matrix between the gmm distributions with diagonal variances
    :param mu_src: [R, D] matrix, the means of R Gaussian distributions
    :param mu_dst: [C, D] matrix, the means of C Gaussian distributions
    :param logvar_src: [R, D] matrix, the log(variance) of R Gaussian distributions
    :param logvar_dst: [C, D] matrix, the log(variance) of C Gaussian distributions
    :return: [R, C] distance matrix
    """
    std_src = torch.exp(0.5 * logvar_src)
    std_dst = torch.exp(0.5 * logvar_dst)
    distance_mean = distance_matrix(mu_src, mu_dst, p=2)
    distance_var = distance_matrix(std_src, std_dst, p=2)
    # distance_var = torch.sum(sum_matrix(std_src, std_dst) - 2 * (prod_matrix(std_src, std_dst) ** 0.5), 2)
    return distance_mean + distance_var + 1e-6


# def tensor_gmm(mu_src: torch.Tensor, mu_dst: torch.Tensor, logvar_src: torch.Tensor, logvar_dst: torch.Tensor):
#     """
#     Calculate a Wasserstein distance matrix between the gmm distributions with diagonal variances
#     :param mu_src: [R, D] matrix, the means of R Gaussian distributions
#     :param mu_dst: [C, D] matrix, the means of C Gaussian distributions
#     :param logvar_src: [R, D] matrix, the log(variance) of R Gaussian distributions
#     :param logvar_dst: [C, D] matrix, the log(variance) of C Gaussian distributions
#     :return: [R, C, D] distance tensor
#     """
#     std_src = torch.exp(0.5 * logvar_src)
#     std_dst = torch.exp(0.5 * logvar_dst)
#     distance_mean = distance_tensor(mu_src, mu_dst, p=2)
#     distance_var = sum_matrix(std_src, std_dst) - 2 * (prod_matrix(std_src, std_dst) ** 0.5)
#     return distance_mean + distance_var


def cost_mat(cost_s: torch.Tensor, cost_t: torch.Tensor, tran: torch.Tensor) -> torch.Tensor:
    """
    Implement cost_mat for Gromov-Wasserstein discrepancy (GWD)

    Suppose the loss function in GWD is |a-b|^2 = a^2 - 2ab + b^2. We have:

    f1(a) = a^2,
    f2(b) = b^2,
    h1(a) = a,
    h2(b) = 2b

    When the loss function can be represented in the following format: loss(a, b) = f1(a) + f2(b) - h1(a)h2(b), we have

    cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
    cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        tran: (ns, nt) matrix (torch tensor), representing the optimal transport from source to target domain.
    Returns:
        cost: (ns, nt) matrix (torch tensor), representing the cost matrix conditioned on current optimal transport
    """
    f1_st = torch.sum(cost_s ** 2, dim=1, keepdim=True) / cost_s.size(0)
    f2_st = torch.sum(cost_t ** 2, dim=1, keepdim=True) / cost_t.size(0)
    tmp = torch.sum(sum_matrix(f1_st, f2_st), dim=2)
    cost = tmp - 2 * cost_s @ tran @ torch.t(cost_t)
    return cost


def fgw_discrepancy(mu, z_mu, logvar, z_logvar, device, beta):
    cost_posterior = distance_gmm(mu, mu, logvar, logvar)
    cost_prior = distance_gmm(z_mu, z_mu, z_logvar, z_logvar)
    cost_pp = distance_gmm(mu, z_mu, logvar, z_logvar)

    ns = cost_posterior.size(0)
    nt = cost_prior.size(0)
    p_s = torch.ones(ns, 1) / ns
    p_t = torch.ones(nt, 1) / nt
    tran = torch.ones(ns, nt) / (ns * nt)
    p_s = p_s.to(device)
    p_t = p_t.to(device)
    tran = tran.to(device)
    dual = (torch.ones(ns, 1) / ns).to(device)
    for m in range(10):
        cost = beta * cost_mat(cost_posterior, cost_prior, tran) + (1 - beta) * cost_pp
        kernel = torch.exp(-cost / torch.max(torch.abs(cost))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        for i in range(5):
            dual = p_s / (kernel @ b)
            b = p_t / (torch.t(kernel) @ dual)
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    cost = beta * cost_mat(cost_posterior, cost_prior, tran.detach().data) + (1 - beta) * cost_pp
    d_fgw = (cost * tran.detach().data).sum()
    return d_fgw


def fgw(source, target, device, beta, cost_source_target, M=None):
    cost_source = distance_matrix(source, source)
    cost_target = distance_matrix(target, target)

    if cost_source_target is None:
        cost_source_target = distance_matrix(source, target)

    ns = cost_source.size(0)
    nt = cost_target.size(0)
    p_s = torch.ones(ns, 1) / ns
    p_t = torch.ones(nt, 1) / nt
    tran = torch.ones(ns, nt) / (ns * nt)
    p_s = p_s.to(device)
    p_t = p_t.to(device)

    if M is None:
        tran = tran.to(device)
    else:
        tran = M
    dual = (torch.ones(ns, 1) / ns).to(device)
    for m in range(10):
        cost = beta * cost_mat(cost_source, cost_target, tran) + (1 - beta) * cost_source_target
        kernel = torch.exp(-cost / torch.max(torch.abs(cost))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        for i in range(5):
            dual = p_s / (kernel @ b)
            b = p_t / (torch.t(kernel) @ dual)
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    cost = beta * cost_mat(cost_source, cost_target, tran.detach().data) + (1 - beta) * cost_source_target
    d_fgw = (cost * tran.detach().data).sum()
    return d_fgw



def train(model, prior, train_loader, optimizer, device, epoch, args):
    model.train()
    prior.train()
    train_rec_loss = 0
    train_reg_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, z, mu, logvar = model(data)
        z_mu, z_logvar = prior()
        rec_loss = method.loss_function(recon_batch, data, args.loss_type)
        reg_loss = args.gamma * fgw_discrepancy(mu, z_mu, logvar, z_logvar, device, args.beta)
        loss = rec_loss + reg_loss
        loss.backward()
        train_rec_loss += rec_loss.item()
        train_reg_loss += reg_loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Model Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))

    print('====> Epoch: {} Average RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        epoch, train_rec_loss / len(train_loader.dataset), train_reg_loss / len(train_loader.dataset),
        (train_rec_loss + train_reg_loss) / len(train_loader.dataset)))


def test(model, prior, test_loader, device, args):
    model.eval()
    prior.eval()
    test_rec_loss = 0
    test_reg_loss = 0
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, z, mu, logvar = model(data)
            z_mu, z_logvar = prior()
            rec_loss = method.loss_function(recon_batch, data, args.loss_type)
            reg_loss = args.gamma * fgw_discrepancy(mu, z_mu, logvar, z_logvar, device, args.beta)
            test_rec_loss += rec_loss.item()
            test_reg_loss += reg_loss.item()
            test_loss += (rec_loss.item() + reg_loss.item())

    test_rec_loss /= len(test_loader.dataset)
    test_reg_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('====> Test set RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        test_rec_loss, test_reg_loss, test_loss))
    return test_rec_loss, test_reg_loss, test_loss


def train_model(model, prior, train_loader, test_loader, device, args):
    model = model.to(device)
    prior = prior.to(device)
    loss_list = []
    optimizer = optim.Adam(list(model.parameters()) + list(prior.parameters()), lr=args.lr, betas=(0.9, 0.999))
    for epoch in range(1, args.epochs + 1):
        train(model, prior, train_loader, optimizer, device, epoch, args)
        test_rec_loss, test_reg_loss, test_loss = test(model, prior, test_loader, device, args)
        loss_list.append([test_rec_loss, test_reg_loss, test_loss])
        if epoch % args.landmark_interval == 0:
            evaluation.interpolation_2d(model, test_loader, device, epoch, args, prefix='rae')
            prior.eval()
            z_p_mean, z_p_logvar = prior()
            evaluation.sampling(model, device, epoch, args, prior=[z_p_mean, z_p_logvar], prefix='rae')
            evaluation.reconstruction(model, test_loader, device, epoch, args, prefix='rae')

    return loss_list
