import torch
from torch import nn
import numpy as np
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors


class WeightDecay(nn.Module):

    def __init__(self,
     network: nn.Module,
     loss_function: nn.Module,
      alpha: float=5e-4):
        super().__init__()
        self.net = network
        self.loss_function = loss_function
        self.weight_vector = self.vectorize_parameters()
        self.alpha = alpha

    def vectorize_parameters(self):
        parameters = list(self.net.parameters())
        parameters_vector = list()
        for p in parameters:
            if len(p.size()) > 1:
                parameters_vector.append(p.view(-1))
        v = torch.cat(parameters_vector, dim=-1)
        return v

    def forward(self, x_batch, y_batch):
        loss_value = self.loss_function(x_batch, y_batch)
        reg_value = torch.norm(self.weight_vector)
        return loss_value + self.alpha * reg_value


class Manifold(nn.Module):

    def __init__(self, 
    network: nn.Module, 
    loss_function: nn.Module, 
    alpha:float = 5e-4,
    k:int = 2,
    device = 'cuda'):
        super().__init__()
        self.net = network
        self.loss_function = loss_function
        self.alpha = alpha
        self.k = k
        self.device = device

    def distance(self, x, y):
        x_v = x.view(-1)
        y_v = y.view(-1)
        return torch.norm(x - y).cpu()

    def are_in_a_same_class(self, y1, y2):
        return y1 == y2

    def compute_weights_based_on_knn(self, x_batch, y_batch):
        # print('Computing K-NN ...')
        n = x_batch.size(0)
        w_int = torch.zeros(size=[n,n],requires_grad=False,dtype=torch.float32, device=self.device)
        w_pen = torch.zeros(size=[n,n],requires_grad=False,dtype=torch.float32, device=self.device)
        w_diff = torch.zeros(size=[n,n],requires_grad=False,dtype=torch.float32, device=self.device)
        n_x_batch = x_batch.view(x_batch.size(0), -1).cpu().numpy()
        nbrs = NearestNeighbors(x_batch.size(0), radius=80.).fit(n_x_batch)
        d, neighbors = nbrs.kneighbors(n_x_batch)
        neighbors = neighbors[:, :self.k + 1]
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        e_d = torch.exp(-d)

        for i in range(n):
            for j in range(n):
                if j in neighbors[i] and self.are_in_a_same_class(y_batch[i], y_batch[j]):
                    w_int[i,j] = e_d[i,j]
                elif j in neighbors[i] and ~self.are_in_a_same_class(y_batch[i], y_batch[j]):
                    w_pen[i,j] = e_d[i,j]

        w_diff = w_int - w_diff
        return w_diff

    def weighted_output(self, x_batch, y_batch, y_output):
        n = x_batch.size(0)
        w_diff = self.compute_weights_based_on_knn(x_batch, y_batch)
        reg_value = 0.
        for i in range(n):
            for j in range(n):
                reg_value += torch.norm(y_output[i] - y_output[j]) * w_diff[i,j]

        return reg_value

    def forward(self, x_batch, y_batch, y_output):
        loss_value = self.loss_function(self.net(x_batch), y_batch)
        reg_value = self.weighted_output(x_batch, y_batch, y_output)
        return loss_value + self.alpha * reg_value





class EnhanceDiversityFeatureExtracition(nn.Module):

    def __init__(self, 
    network: nn.Module, 
    loss_function: nn.Module, 
    alpha: float=5e-4,
    tau: float=0.2):
        super().__init__()
        assert 0 < tau < 1, 'The tau value should be between 0 and 1'
        self.net = network
        self.loss_function = loss_function
        self.alpha = alpha
        self.conv_param = self.conv_parameters()
        self.tau = tau

    def conv_parameters(self):
        parameters = list(self.net.parameters())
        conv_parameters = list()
        for p in parameters:
            if len(p.size()) == 4:
                conv_parameters.append(p)
        return conv_parameters

    def compute_similarity_between_filters_and_mask(self):
        print('Finding Similarity between Filters of the Convlution Layers ...')
        for l in range(len(self.conv_param)):
            conv_w = self.conv_param[l]
            filters = conv_w.size(2)
            sim = torch.zeros(size=[filters,filters], dtype=torch.float32, requires_grad=True)
            for i in range(filters):
                for j in range(filters):
                    sim[i,j] = torch.sum(conv_w[:,:,i,:] * conv_w[:,:,j,:])/(torch.norm(conv_w[:,:,i,:]) * torch.norm(conv_w[:,:,j,:]))
                    if not (self.tau < sim[i,j] <= 1) or i == j:
                        with torch.no_grad():
                            sim[i,j] = 0.
        return sim.cuda().sum()

    def forward(self, x_batch, y_batch):
        loss_value = self.loss_function(x_batch, y_batch)
        reg_value = self.compute_similarity_between_filters_and_mask()
        return loss_value + self.alpha * reg_value


class DifferentConvexFunction(nn.Module):

    def __init__(self,
     network: nn.Module,
     loss_function: nn.Module,
     alpha: float=5e-4):
        super().__init__(self)
        self.net = network
        self.loss_function = loss_function
        self.weight_vector = self.vectorize_parameters()
        self.alpha = alpha

    def vectorize_parameters(self):
        parameters = list(self.net.parameters())
        parameters_vector = list()
        for p in parameters:
            if len(p.size()) > 1:
                parameters_vector.append(p.view(-1))
        return parameters_vector

    def convex_reg(self):
        reg_value = 0.
        for w in self.weight_vector:
            norm_w = torch.norm(w)
            if norm_w * self.alpha < 1:
                reg_value += norm_w * self.alpha
        return reg_value

    def forward(self, x_batch, y_batch):
        loss_value = self.loss_function(x_batch, y_batch)
        reg_value = self.convex_reg()
        return loss_value + reg_value


class ElasticNet(nn.Module):

    def __init__(self,
     network: nn.Module,
     loss_function: nn.Module,
      alpha: float=5e-4,
      beta: float=5e-4):
        super().__init__()
        self.net = network
        self.loss_function = loss_function
        self.weight_vector = self.vectorize_parameters()
        self.alpha = alpha
        self.beta = beta

    def vectorize_parameters(self):
        parameters = list(self.net.parameters())
        parameters_vector = list()
        for p in parameters:
            if len(p.size()) > 1:
                parameters_vector.append(p.view(-1))
        v = torch.cat(parameters_vector, dim=-1)
        return v

    def forward(self, x_batch, y_batch):
        loss_value = self.loss_function(x_batch, y_batch)
        reg_value = self.alpha * torch.norm(self.weight_vector) + self.beta * torch.norm(self.weight_vector, 1)
        return loss_value + reg_value


class TransformedL1(nn.Module):

    def __init__(self,
     network: nn.Module,
     loss_function: nn.Module,
      alpha: float=5e-4,
      mu: float=0.5,
      a: float=0.2):
        super().__init__()
        self.net = network
        self.loss_function = loss_function
        self.weight_vector = self.vectorize_parameters()
        self.alpha = alpha
        self.a = a

    def vectorize_parameters(self):
        parameters = list(self.net.parameters())
        parameters_vector = list()
        for p in parameters:
            if len(p.size()) > 1:
                parameters_vector.append(p.view(-1))
        v = torch.cat(parameters_vector, dim=-1)
        return v

    def transformed_reg(self):
        w_norm = torch.norm(self.weight_vector)
        t_l = (((self.a + 1) * torch.abs(self.weight_vector))/ (self.a + self.weight_vector)).sum()
        return self.mu * t_l + (1 - self.mu) * w_norm

    def forward(self, x_batch, y_batch):
        loss_value = self.loss_function(x_batch, y_batch)
        reg_value = self.transformed_reg()
        return loss_value + self.alpha * reg_value

class SmoothL2(nn.Module):

    def __init__(self,
     network: nn.Module,
     loss_function: nn.Module,
      alpha: float=5e-4,
      a: float=1.0):
        super().__init__()
        self.net = network
        self.loss_function = loss_function
        self.weight_vector = self.vectorize_parameters()
        self.alpha = alpha
        self.a = a

    def vectorize_parameters(self):
        parameters = list(self.net.parameters())
        parameters_vector = list()
        for p in parameters:
            if len(p.size()) > 1:
                parameters_vector.append(p.view(-1))
        v = torch.cat(parameters_vector, dim=-1)
        return v

    def smooth_function(self):
        w_norm = torch.abs(self.weight_vector).sum()
        if w_norm > self.a:
            return w_norm
        else:
            return -1/(8 * self.a**3) * w_norm ** 4 + 3/(4*self.a) * w_norm ** 3 + 3/8 * self.a

    def forward(self, x_batch, y_batch):
        loss_value = self.loss_function(x_batch, y_batch)
        reg_value = self.smooth_function()
        return loss_value + self.alpha * reg_value

class LRFLoss(nn.Module):
    # approximation_function is a string which takes two values:
    # 'svd' for singular values decomposition
    # 'lrf' for low rank factorization 
    # default value is 'svd'
    def __init__(self, loss_function: nn.Module, net: nn.Module, approximation_function:str='svd', verbose=False):
        super().__init__()
        self.loss_function = loss_function
        self.net = net
        self.k = 1
        self.theta_star = list()
        self.verbose = verbose
        p_list = list(self.net.parameters())
        for p in p_list:
            t = p.detach().cpu().numpy()
            self.theta_star.append(torch.tensor(t, dtype=torch.float32))

    @staticmethod
    def vectorize_parameters(param_list):
        theta = list()
        for p in param_list:
            theta.append(p.view(-1))
        return theta

    @staticmethod
    def concat_vectors(vectors):
        return torch.cat(vectors, dim=0)

    def compute_condition_number(self, loss_value: torch.Tensor, verbose=False):
        params = list(self.net.parameters())
        condition_number_list = list()
        for p in params:
            if len(p.size()) > 1:
                j_theta_norm = torch.norm(p.grad)
                theta_norm = torch.norm(p)
                condition_number = j_theta_norm * theta_norm / loss_value
                condition_number_list.append(condition_number)
        if verbose:
            print('--The Condition Number of Layers is {0}'.format(str(condition_number_list)))

        return condition_number_list

    def approximate_lrf_tensor_kernel_filter_wise(self, w):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                m = np.min(w[i, j, :, :])
                w[i, j, :, :] -= m
                mdl = NMF(n_components=self.k, max_iter=10, tol=1.0)
                W = mdl.fit_transform(np.reshape(w[i, j, :, :], [w.shape[2], w.shape[3]]))
                H = mdl.components_
                w[i, j, :, :] = np.matmul(W, H) + m
        return w

    def approximation_nmf_matrix(self, w):
        m = np.min(w)
        w -= m
        mdl = NMF(n_components=self.k, max_iter=20, tol=1.0)
        W = mdl.fit_transform(w)
        H = mdl.components_
        return np.matmul(W, H) + m

    # loss_value: Last Loss Value on a Batch
    def update_theta_star(self, loss_value, verbose=False):
        condition_number_list = self.compute_condition_number(loss_value)
        max_condition_number = max(condition_number_list)
        counter = 0
        for i, p in enumerate(self.theta_star):
            if len(p.size()) > 1:
                c = condition_number_list[i] / max_condition_number
                r = np.random.rand()
                if r < c:
                    w = p.detach().cpu().numpy()
                    if len(w.shape) == 2:
                        w = self.approximation_nmf_matrix(w)
                    if len(w.shape) == 4:
                        w = self.approximate_lrf_tensor_kernel_filter_wise(w)
                    p.data = torch.tensor(w, dtype=torch.float32)
                    counter += 1
        if verbose:
            print('--Number of Factorizations are {0}'.format(counter))

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        loss = self.loss_function(output, target)
        theta = self.concat_vectors(self.vectorize_parameters(list(self.net.parameters())))
        reg = torch.norm(theta - self.theta_star)
        return loss + reg