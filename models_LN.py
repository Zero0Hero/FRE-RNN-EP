import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

import copy


def dataset(config):
    ## Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # 
    ])

    if config.train_task == "FMNIST" :
        config.train_eta_global = 10e-4
        config.n_input = 28*28*1
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform)
    elif config.train_task == "MNIST":
        config.train_eta_global = 10e-4
        config.n_input = 28*28*1
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
    elif config.train_task == "CIFAR10":
        config.train_eta_global = 2e-4
        config.n_input = 32*32*3
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    elif config.train_task == "CIFAR100":
        config.train_eta_global = 2e-4
        config.n_input = 32*32*3
        config.n_out = 100
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform)

    testbs = config.test_batch_size if hasattr(config,'test_batch_size') else config.train_batch_size*4
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=train_dataset, batch_size=testbs, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=testbs, shuffle=False)
    return train_loader,valid_loader,test_loader,config

## 
class FNN_LN(torch.nn.Module):
    def __init__(self, config):
        super(FNN_LN, self).__init__()
        self.device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.train_tmethod = config.train_tmethod 
        self.f = config.f
        self.fd = config.fd
        # self.fl = config.fl if hasattr(config, 'fl') else config.f
        # self.fld = config.fld if hasattr(config, 'fld') else config.fd
        self.fl = config.fl if hasattr(config, 'fl') else linear
        self.fld = config.fld if hasattr(config, 'fld') else linear_d
        self.ff = config.ff if hasattr(config, 'ff') else softmax
        self.ffd = config.ffd if hasattr(config, 'ffd') else softmax_d
        self.nL_hidd = config.nL_hidd  # 

        self.sc_forward = config.sc_forward
        self.sc_bias = config.sc_bias
        self.sc_back = config.sc_back

        self.n_input = config.n_input
        self.n_hidd = config.n_hidd
        self.n_out = config.n_out

        self.LRA_beta = config.LRA_beta if hasattr(config, 'LRA_beta') else 0.1
        self.LRA_gamma = config.LRA_gamma if hasattr(config, 'LRA_gamma') else 0.1
        self.EP_It2sta = config.EP_It2sta
        self.EP_It2sta2 = config.EP_It2sta2 if hasattr(config, 'EP_It2sta2') else config.EP_It2sta
        # 
        self.layers = []
        n_input = config.n_input
        self.train_task = config.train_task

        self.EP_symm_W = config.EP_symm_W if hasattr(config, 'EP_symm_W') else True
        self.EP_f_sc = config.EP_f_sc if hasattr(config, 'EP_f_sc') else 1
        self.EP_b_sc = config.EP_b_sc if hasattr(config, 'EP_b_sc') else 1
        self.EP_nudge = config.EP_nudge if hasattr(config, 'EP_nudge') else 1
        ## !!
        self.biasLearning = config.biasLearning if hasattr(config, 'biasLearning') else False
        self.EP_b_learn = config.EP_b_learn if hasattr(config, 'EP_b_learn') else False
        self.EP_b_learn_replay = config.EP_b_learn_replay if hasattr(config, 'EP_b_learn_replay') else False
        # self.Wr=[]
        for _ in range(self.nL_hidd+1):
            n_hidd = self.n_hidd if _ < self.nL_hidd else self.n_out  # 最后一层为输出层
            W = (torch.rand(n_input, n_hidd,device=self.device)*2-1)   * self.sc_forward* np.sqrt(6.0/(n_input+n_hidd) )
            b = (torch.rand(n_hidd, device=self.device)*2-1)               * self.sc_bias       * 1.0/np.sqrt(n_input )
            E = (torch.rand(n_hidd, n_input,device=self.device)*2-1)   * self.sc_back   * np.sqrt(6.0/(n_input+n_hidd) )

            self.layers.append([W, b, E])
            n_input = n_hidd

        dim = self.nL_hidd*self.n_hidd # +self.n_out
        self.Wa = torch.zeros([dim, dim], device=self.device)
        self.Wsc = self.EP_f_sc*torch.triu(torch.ones_like(self.Wa,device=self.device), diagonal=1) + self.EP_b_sc*torch.tril(torch.ones_like(self.Wa,device=self.device), diagonal=-1)
        self.ba = torch.zeros([dim], device=self.device)
        self.update_Wa()
        # Adam优化器参数
        self.opt_m_w = [torch.zeros_like(layer[0],device=self.device) for layer in self.layers]
        self.opt_v_w = [torch.zeros_like(layer[0],device=self.device) for layer in self.layers]
        self.opt_m_b = [torch.zeros_like(layer[1],device=self.device) for layer in self.layers]
        self.opt_v_b = [torch.zeros_like(layer[1],device=self.device) for layer in self.layers]
        self.opt_m_E = [torch.zeros_like(layer[2],device=self.device) for layer in self.layers]
        self.opt_v_E = [torch.zeros_like(layer[2],device=self.device) for layer in self.layers]

        self.opt_beta1, self.opt_beta2 = 0.9, 0.999
        self.opt_epsilon = 1e-8
        self.opt_eta = config.train_eta_global
        self.opt_t = 0  # 时间步

        # 存储梯度
        self.grads = [None] * len(self.layers)

        ##
        if self.train_tmethod == 'FA':
            self.forward = self.forward_FA
            self.backward = self.backward_FA
        elif self.train_tmethod == 'LRA':
            self.forward = self.forward_LRA 
            self.backward = self.backward_LRA
        elif self.train_tmethod == 'BP':
            self.forward = self.forward_BP 
            self.backward = self.backward_BP
        elif self.train_tmethod == 'EP':
            self.forward = self.forward_EP 
            self.backward = self.backward_EP

    ##
    #
    def forward_BP(self, x):
        x = x.to(self.device)
        self.z = [x]
        self.h = [x]
        for iL, (W, b, E) in enumerate(self.layers):
            self.h.append( self.z[iL].mm(W) + b)
            self.z.append(self.f(self.h[iL+1]))

        self.z[-1] = self.ff(self.h[-1])
        return self.z[-1]

    def backward_BP(self, x, y, output):
        x = x.to(self.device)
        y = y.to(self.device)
        output = output.to(self.device)
        m = y.size(0)
        ef = output - y  # dzf
        self.grads[-1] = [self.z[-2].t().mm(ef) / m, ef.sum(0) / m]  # 

        for iL in range(len(self.layers) - 2, -1, -1):  
            e = ef.mm(self.layers[iL+1][0].t()) * self.fd(self.h[iL+1])  
                
            self.grads[iL] = [self.z[iL].t().mm(e) / m, e.sum(0) / m]  # 
            ef = e  # 
    #
    def forward_FA(self, x):
        x = x.to(self.device)
        self.z = [x]
        self.h = [x]
        for iL, (W, b, E) in enumerate(self.layers):
            self.h.append( self.z[iL].mm(W) + b)
            self.z.append(self.f(self.h[iL+1]))

        self.z[-1] = self.ff(self.h[-1])
        return self.z[-1]

    def backward_FA(self, x, y, output):
        x = x.to(self.device)
        y = y.to(self.device)
        output = output.to(self.device)
        m = y.size(0)
        ef = output - y  # dzf
        self.grads[-1] = [self.z[-2].t().mm(ef) / m, ef.sum(0) / m]  # 

        for iL in range(len(self.layers) - 2, -1, -1):   
            e = ef.mm(self.layers[iL+1][2]) * self.fd(self.h[iL+1])  
                
            self.grads[iL] = [self.z[iL].t().mm(e) / m, e.sum(0) / m]  # 
            ef = e  # 

    #
    def forward_LRA(self, x):
        x = x.to(self.device)
        self.z = [x]
        self.h = [x]
        for iL, (W, b, E) in enumerate(self.layers):
            self.h.append( self.z[iL].mm(W) + b)
            self.z.append(self.f(self.h[iL+1]))

        self.z[-1] = self.ff(self.h[-1])
        return self.z[-1]

    def backward_LRA(self, x, y, output):
        x = x.to(self.device)
        y = y.to(self.device)
        output = output.to(self.device)
        m = y.size(0)
        self.y = copy.deepcopy(self.h)

        ef = output - y  # dzf
        self.grads[-1] = [self.z[-2].t().mm(ef) / m, ef.sum(0) / m, (+self.LRA_beta * ef.mm(self.layers[-1][2])).t().mm(ef) /m]  # 

        for iL in range(len(self.layers) - 2, -1, -1):   
            self.y[iL+1] = self.f( self.y[iL+1] - self.LRA_beta * ef.mm(self.layers[iL+1][2])  )
            e =  self.z[iL+1] - self.y[iL+1]
            self.grads[iL] = [self.z[iL].t().mm(e) / m, e.sum(0) / m, (+self.LRA_beta * ef.mm(self.layers[iL+1][2]) ).t().mm(ef)/m]  # 
            ef = e  #

    #
    def forward_EP(self, x):
        # x = x.to(self.device)

        self.hb = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)
        self.z = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)
        self.h = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)

        self.hb[:,:self.n_hidd] = x.mm(self.layers[0][0])

        for it in range(self.EP_It2sta):
            self.h = self.z.mm(self.Wa*self.Wsc) + self.hb + self.ba
            self.z = self.f(self.h)

        out = self.ff(self.z[:, -self.n_hidd:].mm(self.layers[-1][0]))

        return out
    
    def backward_EP(self, x, y, output):
        # x = x.to(self.device)
        # y = y.to(self.device)
        # output = output.to(self.device)

        m = y.size(0)
        self.ef = output - y  # dzf
        
        self.y = torch.tensor(self.z, device=self.device)
        self.hb = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)

        self.hb[:,:self.n_hidd] = x.mm(self.layers[0][0])
        E = self.layers[-1][0].t() if self.EP_symm_W else self.layers[-1][2]
        
        tef = self.ef
        for it in range(self.EP_It2sta2):
            self.hb[:,-self.n_hidd:] = - tef.mm(E)*self.EP_nudge

            self.h = self.y.mm(self.Wa*self.Wsc) + self.hb + self.ba
            self.y = self.f(self.h)
            out = self.ff(self.z[:, -self.n_hidd:].mm(self.layers[-1][0]))
            tef = out - y

        self.e = self.z - self.y
        self.grads[-1] = [( self.z[:,-self.n_hidd:] ).t().mm(self.ef) / m, self.ef.sum(0) / m, [None]]  # 
        for iL in range(1,len(self.layers)-1):  
            z = self.z[:, (iL-1)*self.n_hidd:(iL)*self.n_hidd]
            e = self.e[:, (iL)*self.n_hidd:(iL+1)*self.n_hidd]
            self.grads[iL] = [z.t().mm(e) / m, e.sum(0) / m, [None]]  #
        
        e = self.e[:, (0)*self.n_hidd:(1)*self.n_hidd]  
        self.grads[0] = [x.t().mm(e) / m, e.sum(0) / m, [None]] 

        return 

    
    def Lyapunov_EP(self, x, y=None, t_e=200, ret_zall=False):
        dim = self.nL_hidd*self.n_hidd # +self.n_out
        delta_x = torch.rand(dim, device=self.device)
        delta_x /= torch.norm(delta_x)
        e_sum = torch.zeros([t_e])
        if ret_zall: zall = torch.zeros([dim, t_e])


        self.hb = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)
        self.z = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)
        self.h = torch.zeros([x.size(0), self.Wa.size(0)], device=self.device)

        self.hb[:,:self.n_hidd] = x.mm(self.layers[0][0])
        # hb[-self.n_hidd-1:-1] = 0
        for it in range(t_e):
            self.h = self.z.mm(self.Wa*self.Wsc) + self.hb + self.ba
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
        dim = self.nL_hidd*self.n_hidd +self.n_out        
        error = np.zeros(dim)
        error[:-self.n_out] = torch.mean(self.e,dim=0).cpu()
        error[-self.n_out:] = torch.mean(self.ef,dim=0).cpu()
        return error
    ##      
    def update_Wa(self):
        self.ba[(0)*self.n_hidd:(1)*self.n_hidd] = self.layers[0][1]
        for iL in range(self.nL_hidd-1):
            (W, b, E) = self.layers[iL+1]
            E = W.t() if self.EP_symm_W else E
            self.Wa[(iL)*self.n_hidd:(iL+1)*self.n_hidd, (iL+1)*self.n_hidd:(iL+2)*self.n_hidd] = W #* self.EP_f_sc
            self.Wa[(iL+1)*self.n_hidd:(iL+2)*self.n_hidd, (iL)*self.n_hidd:(iL+1)*self.n_hidd] = E #* self.EP_b_sc
            self.ba[(iL+1)*self.n_hidd:(iL+2)*self.n_hidd] = b

        return
    
    def update_weights(self, lr=None):
        if lr == None:
            for iL in range(len(self.layers)):
                self.layers[iL][0] -= self.opt_eta * self.grads[iL][0]  # 

                if self.biasLearning: 
                    self.layers[iL][1] -= self.opt_eta * self.grads[iL][1]

                if (not self.EP_symm_W) and self.EP_b_learn and iL>0: 
                    self.layers[iL][2] -= self.opt_eta * self.grads[iL][2]
        else:
            for iL in range(len(self.layers)):
                self.layers[iL][0] -= lr[-iL] * self.grads[iL][0]  # 

                if self.biasLearning: 
                    self.layers[iL][1] -= lr[-iL] * self.grads[iL][1]

                if (not self.EP_symm_W) and self.EP_b_learn and iL>0: 
                    self.layers[iL][2] -= lr[-iL] * self.grads[iL][2]

        self.update_Wa()

    def update_weights_adam(self):
        # 
        self.opt_t += 1
        
        for iL in range(len(self.layers)):
            # 
            dw_update, self.opt_m_w[iL], self.opt_v_w[iL] = adam_update(self.opt_m_w[iL], self.opt_v_w[iL], self.grads[iL][0], self.opt_beta1, self.opt_beta2, self.opt_t)
            self.layers[iL][0] -= self.opt_eta * dw_update

            if self.biasLearning: 
                db_update, self.opt_m_b[iL], self.opt_v_b[iL] = adam_update(self.opt_m_b[iL], self.opt_v_b[iL], self.grads[iL][1], self.opt_beta1, self.opt_beta2, self.opt_t)
                self.layers[iL][1] -= self.opt_eta * db_update
            if (not self.EP_symm_W) and self.EP_b_learn and iL>0: 
                dE_update, self.opt_m_E[iL], self.opt_v_E[iL] = adam_update(self.opt_m_E[iL], self.opt_v_E[iL], self.grads[iL][2], self.opt_beta1, self.opt_beta2, self.opt_t)
                self.layers[iL][2] -= self.opt_eta * dE_update


            if (self.train_tmethod == 'LRA') & iL>0 :
                if self.train_task == "CIFAR10":
                    dE_update, self.opt_m_E[iL], self.opt_v_E[iL] = adam_update(self.opt_m_E[iL], self.opt_v_E[iL], self.grads[iL][2].t(), self.opt_beta1, self.opt_beta2, self.opt_t)
                    self.layers[iL][2] -= self.LRA_gamma * self.opt_eta * dE_update
                else:
                    dE_update, self.opt_m_E[iL], self.opt_v_E[iL] = adam_update(self.opt_m_E[iL], self.opt_v_E[iL], self.grads[iL][0].t(), self.opt_beta1, self.opt_beta2, self.opt_t)
                    self.layers[iL][2] -= self.LRA_gamma * self.opt_eta * dE_update

        self.update_Wa()





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
