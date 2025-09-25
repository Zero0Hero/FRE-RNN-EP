import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

import copy


def dataset(task, device, bs, validbstimes=1, Norm=False):
    ## Dataset
    

    if task == "FMNIST" :        
        transform = transforms.Compose([
            transforms.ToTensor(),  # 
        ])
        input_nodes = 28*28
        out_nodes = 10
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform)
    elif task == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),  # 
        ])
        input_nodes = 28*28
        out_nodes = 10
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
    elif task == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),  # 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if Norm else transforms.Lambda(lambda x: x)
        ])
        
        input_nodes = 32*32*3
        out_nodes = 10
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    elif task == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),  # 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if Norm else transforms.Lambda(lambda x: x)
        ])
        input_nodes = 32*32*3
        out_nodes = 100
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform)

    testbs =  bs*validbstimes
    train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(dataset=train_dataset, batch_size=testbs, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=testbs, shuffle=False)

    train_loader_dev = [(data.to(device), target.to(device)) for data, target in train_loader]
    valid_loader_dev = [(data.to(device), target.to(device)) for data, target in valid_loader]
    test_loader_dev = [(data.to(device), target.to(device)) for data, target in test_loader]

    return [train_loader,valid_loader,test_loader],[train_loader_dev,valid_loader_dev,test_loader_dev],[input_nodes,out_nodes]

## 
class EPNN_LN(torch.nn.Module):
    def __init__(self, W_conn, nodes, Para_t=None, Para_EP=None, device='cpu'):
        super(EPNN_LN, self).__init__()
        self.device = device

        self.W_conn = W_conn
        self.nodes = nodes

        # ----Para training        
        self.t_method = Para_t['method'] 
        self.t_task = Para_t['task']

        self.f = Para_t['f'] 
        self.fd = Para_t['fd'] 

        self.ff = Para_t['ff'] 
        self.ffd = Para_t['ffd'] 
        
        self.sc_bias = Para_t['bias_init_sc'] # 
        self.biasLearning = Para_t['bias_learn']
        # -----        
        self.nL_hidd = len(nodes)-2  

        # ----Para EP
        self.EP_It2sta = Para_EP['It2sta']
        self.EP_It2sta2 = Para_EP['It2sta2']

        # 
        self.EP_symm_W = Para_EP['EP_symm_W']
        self.EP_f_sc = Para_EP['fsc']
        self.EP_b_sc = Para_EP['bsc']
        self.EP_nudge = Para_EP['nudge']
        

        self.layers = [[None for _ in range(self.nL_hidd+2)] for _ in range(self.nL_hidd+2)]
        for iL in range(self.nL_hidd+2):
            for iiL in range(self.nL_hidd+2):
                if self.W_conn[iL,iiL]>0:
                    n_in, n_out = self.nodes[iL], self.nodes[iiL]
                    
                    W = (torch.rand(n_in, n_out,device=self.device)*2-1)* np.sqrt(6.0/(n_in+n_out) )
                    # E = (torch.rand(n_out, n_in,device=self.device)*2-1)* np.sqrt(6.0/(n_in+n_out) )
                    self.layers[iL][iiL] = W

        # self.layersb = []
        # for iL in range(1,self.nL_hidd+2):
        #     n_in = self.nodes[iL]
        #     b = (torch.rand(n_out, device=self.device)*2-1) * self.sc_bias * 1.0/np.sqrt(n_in)
        #     self.layersb.append(b)
        dim = sum(nodes[1:-1])
        self.Wa = torch.zeros([dim, dim], device=self.device)
        self.update_Wa()
        self.Wsc = (self.EP_f_sc*torch.triu(torch.ones_like(self.Wa,device=self.device), diagonal=1) + self.EP_b_sc*torch.tril(torch.ones_like(self.Wa,device=self.device), diagonal=-1))*(self.Wa!=0.0)
        # self.ba = torch.zeros([dim], device=self.device)
        

        # 
        self.grads = [[None for _ in range(self.nL_hidd+2)] for _ in range(self.nL_hidd+2)]
        self.opt_m = [[None for _ in range(self.nL_hidd+2)] for _ in range(self.nL_hidd+2)]
        self.opt_v = [[None for _ in range(self.nL_hidd+2)] for _ in range(self.nL_hidd+2)]
        if Para_t['adam']:
            for iL in range(self.nL_hidd+2):
                for iiL in range(self.nL_hidd+2):
                    if self.W_conn[iL,iiL]>0:
                        self.opt_m[iL][iiL] = torch.zeros_like(self.layers[iL][iiL], device=self.device)
                        self.opt_v[iL][iiL] = torch.zeros_like(self.layers[iL][iiL], device=self.device)
                        # self.opt_m_b = [torch.zeros_like(layer[1],device=self.device) for layer in self.layers]
                        # self.opt_v_b = [torch.zeros_like(layer[1],device=self.device) for layer in self.layers]

            self.opt_beta1, self.opt_beta2 = 0.9, 0.999
            self.opt_epsilon = 1e-8
            self.opt_t = 0  # 

        self.opt_eta = Para_t['eta']

        ##
        if self.t_method == 'EP':
            self.forward = self.forward_EP 
            self.backward = self.backward_EP
        
    #
    def forward_EP(self, x):
        # x = x.to(self.device)

        self.hb = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)
        self.z = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)
        self.h = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)

        for iL in range(1,self.nL_hidd+1):
            if self.W_conn[0,iL]>0:
                self.hb[:,sum(self.nodes[1:iL]):sum(self.nodes[1:iL+1])] = x.mm(self.layers[0][iL])

        for it in range(self.EP_It2sta):
            self.h = self.z.mm(self.Wa*self.Wsc) + self.hb #+ self.ba
            self.z = self.f(self.h)

        out = torch.zeros([x.size(0), self.nodes[-1]], device=self.device)
        for iL in range(1,self.nL_hidd+1):
            if self.W_conn[iL,-1]>0:
                out += self.z[:, sum(self.nodes[1:iL]):sum(self.nodes[1:iL+1])].mm(self.layers[iL][-1])
        out = self.ff(out)
        return out
    
    def backward_EP(self, x, y, output):
        m = y.size(0)
        self.ef = output - y  # dzf
        
        self.y = self.z.clone().detach()
        self.hb = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)

        for iL in range(1,self.nL_hidd+1):
            if self.W_conn[0,iL]>0:
                self.hb[:,sum(self.nodes[1:iL]):sum(self.nodes[1:iL+1])] = x.mm(self.layers[0][iL])
        
        tmp_ef=self.ef
        for it in range(self.EP_It2sta2):
            for iL in range(1,self.nL_hidd+1):
                if self.W_conn[-1,iL]>0:
                    E = self.layers[iL][-1].t() if self.EP_symm_W else self.layers[-1][iL]
                    self.hb[:,sum(self.nodes[1:iL]):sum(self.nodes[1:iL+1])] += - tmp_ef.mm(E)*self.EP_nudge

            self.h = self.y.mm(self.Wa*self.Wsc) + self.hb # + self.ba
            self.y = self.f(self.h)

            out = torch.zeros([x.size(0), self.nodes[-1]], device=self.device)
            for iL in range(1,self.nL_hidd+1):
                if self.W_conn[iL,-1]>0:
                    out += self.y[:, sum(self.nodes[1:iL]):sum(self.nodes[1:iL+1])].mm(self.layers[iL][-1])
            out = self.ff(out)
            self.ef = out - y

        # self.ef = output - out
        self.e = self.z - self.y
        for iL in range(self.nL_hidd+2):
            for iiL in range(self.nL_hidd+2):
                if self.W_conn[iL,iiL]>0:
                    if iL<iiL:
                        z = self.z[:, sum(self.nodes[1:iL]):sum(self.nodes[1:iL+1])] if iL>0 else x
                        e = self.e[:, sum(self.nodes[1:iiL]):sum(self.nodes[1:iiL+1])] if iiL<(self.nL_hidd+1) else self.ef
                        self.grads[iL][iiL] = z.t().mm(e)/m

        return 

    
    def Lyapunov_EP(self, x, y=None, t_e=200, ret_zall=False):
        dim = sum(self.nodes[1:-1]) 
        delta_x = torch.rand(dim, device=self.device)
        delta_x /= torch.norm(delta_x)
        e_sum = torch.zeros([t_e])
        if ret_zall: zall = torch.zeros([dim, t_e])


        self.hb = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)
        self.z = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)
        self.h = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)

        for iL in range(1,self.nL_hidd+1):
            if self.W_conn[0,iL]>0:
                self.hb[:,sum(self.nodes[1:iL]):sum(self.nodes[1:iL+1])] = x.mm(self.layers[0][iL])

        for it in range(t_e):
            self.h = self.z.mm(self.Wa*self.Wsc) + self.hb #+ self.ba
            self.z = self.f(self.h)
            if ret_zall: zall[:, it] = self.z[0,:]


            hfd = torch.mean(self.fd(self.h[:,:]),dim=0)

            delta_x = torch.einsum('i,ij->j', delta_x, self.Wa*self.Wsc) 
            delta_x =  torch.einsum('j,j->j', hfd, delta_x) 

            e_sum[it] = torch.log(torch.norm(delta_x))
            delta_x /= torch.norm(delta_x)
    
        SR = torch.max(torch.abs(torch.linalg.eigvals(self.Wa*self.Wsc))).item()
        if ret_zall: return SR, torch.mean(e_sum[:self.EP_It2sta]), torch.mean(e_sum[t_e//2:]), zall
        else: return SR, torch.mean(e_sum[:self.EP_It2sta]), torch.mean(e_sum[t_e//2:])

        
    def ret_error(self):
        dim = sum(self.nodes[1:])        
        error = np.zeros(dim)
        error[:-self.nodes[-1]] = torch.mean(self.e,dim=0).cpu()
        error[-self.nodes[-1]:] = torch.mean(self.ef,dim=0).cpu()
        return error
    ##      
    def update_Wa(self):
        for iL in range(1,self.nL_hidd+1):
            for iiL in range(1,self.nL_hidd+1):
                if self.W_conn[iL,iiL]>0:
                    if iL<iiL:
                        W = self.layers[iL][iiL]
                    else:
                        W = self.layers[iiL][iL].t() if self.EP_symm_W else self.layers[iL][iiL]
                    self.Wa[sum(self.nodes[1:iL]):sum(self.nodes[1:iL+1]), sum(self.nodes[1:iiL]):sum(self.nodes[1:iiL+1])] = W #* self.EP_f_sc
                    # self.Wa[sum(self.nodes[1:iiL]):sum(self.nodes[1:iiL+1]), sum(self.nodes[1:iL]):sum(self.nodes[1:iL+1])] = E #* self.EP_b_sc
                    # self.ba[sum(self.nodes[1:iL]):sum(self.nodes[1:iL+1])] = b

        return
    
    def update_weights(self):
        for iL in range(self.nL_hidd+2):
            for iiL in range(self.nL_hidd+2):
                if self.W_conn[iL,iiL]>0:
                    if iL<iiL:
                        self.layers[iL][iiL] -= self.opt_eta * self.grads[iL][iiL]

        self.update_Wa()
        return

    def update_weights_adam(self,coelrnA=torch.tensor([1,1])):
        # 
        self.opt_t += 1
        
        for iL in range(self.nL_hidd+2):
            for iiL in range(self.nL_hidd+2):
                if self.W_conn[iL,iiL]>0:
                    if iL<iiL:
                        dw_update, self.opt_m[iL][iiL], self.opt_v[iL][iiL] = adam_update(self.opt_m[iL][iiL], self.opt_v[iL][iiL], self.grads[iL][iiL], self.opt_beta1, self.opt_beta2, self.opt_t)
                        if (abs(iiL-iL)-0.5)>1:
                            self.layers[iL][iiL] -= self.opt_eta * dw_update * coelrnA[1]
                        else:
                            self.layers[iL][iiL] -= self.opt_eta * dw_update * coelrnA[0]

        self.update_Wa()
        return


def rand_sparse_matrix(rows, cols, connection_rate):

    assert 0 < connection_rate <= 1, "The connection rate must be between (0,1]"

    # Calculate the number of non-zero elements
    num_elements = rows * cols
    num_nonzero = int(num_elements * connection_rate)

    # Randomly generate the position of non-zero elements
    row_indices = torch.randint(0, rows, (num_nonzero,))
    col_indices = torch.randint(0, cols, (num_nonzero,))

    # Stack row and column indices into a two-dimensional tensor
    indices = torch.stack((row_indices, col_indices))

    # Randomly generate values for non-zero elements
    values = torch.rand(num_nonzero)*2-1

    # Create Sparse Matrix
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (rows, cols))

    return sparse_matrix


def adam_update(m, v, dw, beta1, beta2, t, epsilon=1e-8):
    # Update first-order moment estimation
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    
    # First and Second Order Moment Estimation for Deviation Correction Calculation
    m_corr = m / (1 - beta1 ** t)
    v_corr = v / (1 - beta2 ** t)
    
    # Calculate update value
    update = m_corr / (torch.sqrt(v_corr) + epsilon)
    
    return update, m, v


def linear(x):
    return x

def linear_d(x):
    return torch.ones_like(x)

def tanh(x):
    return torch.tanh(x)

def tanh_d(x):
    return 1 - torch.tanh(x) ** 2  

def sign(x):
    return torch.sign(x)

def sign_d(x):
    return torch.zeros_like(x)

def sigmoid(x):
    return torch.sigmoid(x)

def sigmoid_d(x):
    return (1-torch.sigmoid(x)) * torch.sigmoid(x)

def relu(x):
    return torch.max(x, torch.zeros_like(x))  

def relu_d(x):
    return torch.where(x >= 0, torch.ones_like(x), torch.zeros_like(x))

def relu6(x):
    return torch.max(torch.min(x, torch.ones_like(x) * 6), torch.zeros_like(x))  

def relu6_d(x):
    return torch.where((x > 0) & (x <= 6), torch.ones_like(x), torch.zeros_like(x))

def lrelu(x):
    return F.leaky_relu(x, negative_slope=0.1)

def lrelu_d(x):
    return torch.where(x >= 0, torch.ones_like(x), torch.ones_like(x)*0.1)

def softmax(x):
    x = x - torch.max(x, dim=1, keepdim=True)[0]  # Subtract the maximum value to prevent overflow
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)

def softmax_d(x):
    s = softmax(x)
    return s * (1 - s)  # Only computes the i=j case

# One-hot 
def one_hot(labels, n_out, device='cpu'):
    one_hot_labels = torch.zeros(labels.size(0), n_out, device=device)
    one_hot_labels[torch.arange(labels.size(0)), labels] = 1
    return one_hot_labels

def cross_entropy_loss(output, target):
    epsilon = 1e-8  # avoid log(0)
    output = torch.clamp(output, epsilon, 1. - epsilon)  # Limit the output to [epsilon, 1-epsilon] 
    return -torch.mean(torch.sum(target * torch.log(output), dim=1))
