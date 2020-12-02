import torch
from torch import nn
import numpy as np
import numpy.linalg as linalg
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
from typing import Callable



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

    def forward(self, output, target):
        loss_value = self.loss_function(output, target)
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
        for l in range(len(self.conv_param)):
            conv_w = self.conv_param[l]
            filters = conv_w.size(2)
            sim = torch.zeros(size=[filters,filters], dtype=torch.float32, requires_grad=True)
            for i in range(filters):
                for j in range(filters):
                    t_sim = torch.sum(conv_w[:,:,i,:] * conv_w[:,:,j,:])/(torch.norm(conv_w[:,:,i,:]) * torch.norm(conv_w[:,:,j,:]))
                    if self.tau < t_sim <= 1 and i != j:
                            sim[i,j] = t_sim
        return sim.cuda().sum()

    def forward(self, output, target):
        loss_value = self.loss_function(output, target)
        reg_value = self.compute_similarity_between_filters_and_mask()
        return loss_value + self.alpha * reg_value


class DifferentConvexFunction(nn.Module):

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
        return parameters_vector

    def convex_reg(self):
        reg_value = 0.
        for w in self.weight_vector:
            norm_w = torch.norm(w)
            if norm_w * self.alpha < 1:
                reg_value += norm_w * self.alpha
        return reg_value

    def forward(self, output, target):
        loss_value = self.loss_function(output, target)
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

    def forward(self, output, target):
        loss_value = self.loss_function(output, target)
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
        self.mu = mu

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

    def forward(self, output, target):
        loss_value = self.loss_function(output, target)
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

    def forward(self, output, target):
        loss_value = self.loss_function(output, target)
        reg_value = self.smooth_function()
        return loss_value + self.alpha * reg_value

class LRFLoss(nn.Module):
    # approximation_function is a string which takes two values:
    # 'svd' for singular values decomposition
    # 'lrf' for low rank factorization 
    # default value is 'svd'
    def __init__(self, 
        net: nn.Module, 
        loss_function: nn.Module, 
        approximation_function:str='svd', 
        shrink:Callable=None,
        verbose=False):
        super().__init__()
        self.loss_function = loss_function
        self.net = net
        self.k = 1
        self.af = approximation_function
        self.verbose = verbose
        self.shrink = shrink
        p_list = list(self.net.parameters())
        ts = list()
        self.theta = list()
        for p in p_list:
            if len(p.size()) > 1:
                t = p.detach().cpu().numpy()
                ts.append(torch.tensor(t, dtype=torch.float32).view(-1))
                self.theta.append(p)
        self.theta_star = self.concat_vectors(ts)

    def forward(self, output, target):
        loss = self.loss_function(output, target)
        if not self.theta:
            return loss
        theta = self.concat_vectors(self.vectorize_parameters(self.theta))
        reg = torch.norm(theta - self.theta_star.cuda())
        alpha = 1/len(self.theta)
        return loss + alpha * reg

    @staticmethod
    def vectorize_parameters(param_list):
        theta = list()
        for p in param_list:
            if len(p.size()) > 1:
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

    def approximate_svd_tensor(self, w: np.ndarray) -> np.ndarray:
        w_shape = w.shape
        n1 = w_shape[0]
        n2 = w_shape[1]
        ds = []
        if w_shape[2] == 1 or w_shape[3] == 1:
            return w
        u, s, v = linalg.svd(w)
        for i in range(n1):
            for j in range(n2):
                ds.append(self.optimal_d(s[i, j]))
        d = int(np.mean(ds))
        w = np.matmul(u[..., 0:d], s[..., 0:d, None] * v[..., 0:d, :])
        return w
		
    def approximate_svd_matrix(self, w):
        u, s, v = linalg.svd(w)
        d = self.optimal_d(s)
        w = np.matmul(u[:, 0:d], np.matmul(np.diag(s[0:d]), v[0:d, :]))
        return w

    @staticmethod
    def optimal_d(s):
        variance = np.std(s)
        mean = np.average(s)
        for i in range(s.shape[0] - 1):
            if s[i] < mean + variance:
                return i
        return s.shape[0] - 1

    # loss_value: Last Loss Value on a Batch
    def update_w_star(self, loss_value, epoch=1, verbose=False):
        condition_number_list = self.compute_condition_number(loss_value)
        max_condition_number = max(condition_number_list)
        ts = list()
        self.theta = list()
        counter = 0
        for p in list(self.net.parameters()):
            if len(p.size()) > 1:
                c = condition_number_list[counter] / max_condition_number
                r = np.random.rand()
                if self.shrink is not None:
                    r *= 1/(self.shrink(epoch) + 1)
                w = p.detach().cpu().numpy()
                if r < c:
                    if len(w.shape) == 2:
                        if self.af == 'svd':
                            w = self.approximate_svd_matrix(w)
                        elif self.af == 'nmf':
                            w = self.approximation_nmf_matrix(w)
                        else:
                            raise Exception('The approximation function is not currect (nmf or svd)')
                            exit(-1)
                    if len(w.shape) == 4:
                        if self.af == 'svd':
                            w = self.approximate_svd_tensor(w)
                        elif self.af == 'nmf':
                            w = self.approximate_lrf_tensor_kernel_filter_wise(w)
                        else:
                            raise Exception('The approximation function is not currect (nmf or svd)')
                            exit(-1)
                    if len(w.shape) == 3:
                        raise Exception('One (or more than one) of layers has weights with lenght 3.')
                    ts.append(torch.tensor(w, dtype=torch.float32).view(-1))
                    self.theta.append(p)
                    counter += 1
        if ts:
            self.theta_star = self.concat_vectors(ts)
        if verbose:
            print('--Number of Factorizations are {0}'.format(counter))
		
class ASRLoss(nn.Module):

    def __init__(self, net: nn.Module, loss_function, alpha:float = 0.05):
        super().__init__()
        self.net = net
        self.loss_function = loss_function
        self.last_loss = 100000.
        self.last_acc = 0.
        self.alpha = alpha
        self.w_star = self.get_w_star(list(self.net.parameters()))

    def forward(self, output, target):
        alpha = torch.tensor(self.alpha, requires_grad=False).cuda()
        loss_val = self.loss_function(output, target)  + alpha * torch.norm(
        self.vectorize(list(self.net.parameters())) - self.w_star)
        return loss_val

    def get_w_star(self, t_param):
        vec_list = list()
        for p in t_param:
            w = p.detach().cpu().numpy()
            if len(w.shape) == 4:
                w = self.approximate_svd_tensor(w)
            vec_list.append(np.reshape(w, -1))
        vec = np.concatenate(vec_list, axis=0)
        vec = torch.tensor(vec, requires_grad=False).cuda()
        return vec

    def update_w_star_by_loss(self, current_val_loss):
        if current_val_loss < self.last_loss:
            self.last_loss = current_val_loss.detach().cpu().numpy()
            self.w_star = self.get_w_star(list(self.net.parameters()))

    def update_w_star_by_acc(self, current_val_acc):
        if current_val_acc > self.last_acc:
            self.last_acc = current_val_acc
            self.w_star = self.get_w_star(list(self.net.parameters()))

    def vectorize(self, t_param):
        vec_list = list()
        for p in t_param:
            vec_list.append(p.view(-1))
        vec = torch.cat(vec_list, dim=0)
        return vec

    def approximate_svd_tensor(self, w: np.ndarray) -> np.ndarray:
        w_shape = w.shape
        n1 = w_shape[0]
        n2 = w_shape[1]
        ds = []
        if w_shape[2] == 1 or w_shape[3] == 1:
            return w
        u, s, v = linalg.svd(w)
        for i in range(n1):
            for j in range(n2):
                ds.append(self.optimal_d(s[i, j]))
        d = int(np.mean(ds))
        w = np.matmul(u[..., 0:d], s[..., 0:d, None] * v[..., 0:d, :])
        return w
		
    def approximate_svd_matrix(self, w):
        u, s, v = linalg.svd(w)
        d = self.optimal_d(s)
        w = np.matmul(u[:, 0:d], np.matmul(np.diag(s[0:d]), v[:,0:d]))
        return w

    @staticmethod
    def optimal_d(s):
        variance = np.std(s)
        mean = np.average(s)
        for i in range(s.shape[0] - 1):
            if s[i] < mean + variance:
                return i
        return s.shape[0] - 1
