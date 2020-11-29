import torch
from torch import nn
import numpy as np


class WeightDecay(nn.Module):

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
        super().__init__(self)
        self.net = network
        self.loss_function = loss_function
        self.weight_vector = self.vectorize_parameters()
        self.alpha = alpha
        self.k = k
        self.device = device

    def distance(self, x, y):
        x_v = x.view(-1)
        y_v = y.view(-1)
        return torch.norm(x - y).cpu()

    def are_in_a_same_class(self, y1, y2):
        bs = y1 == y2
        b = True
        for i in range(y1.size(0)):
            b = bs[i] and b
        return b

    def compute_weights_based_on_knn(self, x_batch, y_batch):
        print('Computing K-NN ...')
        n = x_batch.size(0)
        d = torch.zeros(size=[n,n],requires_grad=False,dtype=torch.float32)
        w_int = torch.zeros(size=[n,n],requires_grad=False,dtype=torch.float32, device=self.device)
        w_pen = torch.zeros(size=[n,n],requires_grad=False,dtype=torch.float32, device=self.device)
        w_diff = torch.zeros(size=[n,n],requires_grad=False,dtype=torch.float32, device=self.device)
        neighbors = list()
        for i in range(n):
            for j in range(n):
                if i != j:
                    d[i,j] = self.distance(x_batch[i], x_batch[j])

        for i in range(n):
            neighbor = list()
            max_d = 0.
            max_idx = 0
            for w in range(self.k):
                for j in range(n):
                    if len(neighbor) < self.k:
                        neighbor.append(j)
                        if max_d < d[i,j]:
                            max_d = d[i,j]
                            max_idx = j
                    if max_d > d[i,j]:
                        idx = np.argmax(neighbor)
                        neighbor[idx] = j
                        idx = np.argmax(neighbor)
                        max_d = np.max(neighbor)
            neighbors.append(neighbor)


        for i in range(n):
            for j in range(n):
                if j in neighbors[i] and self.are_in_a_same_class(y_batch[i], y_batch[j]):
                    w_int[i,j] = torch.exp(-d[i,j])
                elif j in neighbors[i] and ~self.are_in_a_same_class(y_batch[i], y_batch[j]):
                    w_pen[i,j] = torch.exp(-d[i,j])

        w_diff = w_int - w_diff
        return w_diff

    def weighted_output(self, x_batch, y_batch, y_output):
        n = x_batch.size(0)
        w_diff = self.compute_weights_based_on_knn(x_batch, y_batch)
        reg_value = 0.
        for i in range(n):
            for j in range(n):
                reg_value += torch.norm(y_output[i] - y_output[j]) * w_diff

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
        super().__init__(self)
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
            if len(p.size) == 4:
                conv_parameters.append(p)
        return conv_parameters

    def compute_similarity_between_filters_and_mask(self):
        print('Finding Similarity between Filters of the Convlution Layers ...')
        for l in len(self.conv_param):
            conv_w = self.conv_param[l]
            f1 = conv_w.size(2)
            f2 = conv_w.size(3)
            sim = torch.zeros(size=[f1,f2], dtype=torch.float32, requires_grad=True)
            for i in range(f1):
                for j in range(f2):
                    sim[i,j] = conv_w[:,:,i,:].view(-1).dot(conv_w[:,:,:,j].view(-1))/(torch.norm(conv_w[:,:,i,:]) * torch.norm(conv_w[:,:,:,j]))
                    if not (self.tau < sim[i,j] <= 1) or i == j:
                        sim[i,j] = 0.
        return sim.sum()

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
        super().__init__(self)
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
        super().__init__(self)
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

class WeightDecay(nn.Module):

    def __init__(self,
     network: nn.Module,
     loss_function: nn.Module,
      alpha: float=5e-4,
      a: float=1.0):
        super().__init__(self)
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


class WeightDecay(nn.Module):

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
        v = torch.cat(parameters_vector, dim=-1)
        return v

    def forward(self, x_batch, y_batch):
        loss_value = self.loss_function(x_batch, y_batch)
        reg_value = torch.norm(self.weight_vector)
        return loss_value + self.alpha * reg_value