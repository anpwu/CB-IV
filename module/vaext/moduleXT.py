import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

class FullyConnected(nn.Sequential):
    def __init__(self, sizes, final_activation=None):
        layers = []
        layers.append(nn.Linear(sizes[0],sizes[1]))
        for in_size, out_size in zip(sizes[1:], sizes[2:]):
            layers.append(nn.ELU())
            layers.append(nn.Linear(in_size, out_size))
        if final_activation is not None:
            layers.append(final_activation)
        self.length = len(layers)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)
        
    def __len__(self):
        return self.length

class Encoder(nn.Module):
    def __init__(self, x_dim,z_dim,q_z_nn_layers,q_z_nn_width,device,common_stds):
        super().__init__()
        self.z_dim = z_dim
        self.common_stds = common_stds
        
        # q(z|x,t)
        self.q_z_nn = FullyConnected([x_dim+1] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        
        self.z_log_std = nn.Parameter(torch.ones(z_dim, device=device))
        self.to(device)
        
    def forward(self, x, t):
        z_res = self.q_z_nn(torch.cat([x, t], axis=1))
        z_pred = z_res[:,:self.z_dim]
        z_std = torch.exp(z_res[:,self.z_dim:])
        if self.common_stds:
            z_std = torch.exp(self.z_log_std).repeat(x.shape[0],1)
        return z_pred, z_std
        
class Decoder(nn.Module):
    def __init__(self,x_dim,z_dim,x_mode,t_mode,y_mode,p_y_zt_nn_layers,p_y_zt_nn_width,p_t_z_nn_layers,p_t_z_nn_width,p_x_z_nn_layers,p_x_z_nn_width,device,common_stds):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.x_mode = x_mode
        self.t_mode = t_mode
        self.y_mode = y_mode
        self.common_stds = common_stds
        self.x_n_continuous = sum(np.array(x_mode)==2)
        
        self.x_nn = FullyConnected([z_dim] + p_x_z_nn_layers*[p_x_z_nn_width] + [sum(x_mode)])
        self.t_nn = FullyConnected([z_dim] + p_t_z_nn_layers*[p_t_z_nn_width] + [sum(t_mode)])
        self.y_nn = FullyConnected([z_dim+1] + p_y_zt_nn_layers*[p_y_zt_nn_width] + [sum(y_mode)])

        self.x_log_std = nn.Parameter(torch.FloatTensor(self.x_n_continuous*[1.], device=device))
        self.t_log_std = nn.Parameter(torch.FloatTensor([1.], device=device))
        self.y_log_std = nn.Parameter(torch.FloatTensor([1.], device=device))
        
        self.to(device)
        
    def forward(self, z, t):
        
        x_res = self.x_nn(z)
        x_pred = x_res[:,self.x_n_continuous:]
        x_std = torch.exp(x_res[:,:self.x_n_continuous])
        if self.common_stds:
            x_std = torch.exp(self.x_log_std).repeat(t.shape[0],1)
        
        t_res = self.t_nn(z)
        if self.t_mode == [2]:
            t_pred = t_res[:,:1]
            t_std = torch.exp(t_res[:,1:])
            if self.common_stds:
                t_std = torch.exp(self.t_log_std).repeat(t.shape[0],1)
        else:
            t_pred = t_res
            t_std = t_res-t_res+1e-4
        
        y_res = self.y_nn(torch.cat([z,t],1))
        if self.y_mode == [2]:
            y_pred = y_res[:,:1]
            y_std = torch.exp(y_res[:,1:])
            if self.common_stds:
                y_std = torch.exp(self.y_log_std).repeat(t.shape[0],1)
        else:
            y_pred = y_res
            y_std = y_res-y_res+1e-4
        
        return x_pred,x_std,t_pred,t_std,y_pred,y_std

class CEVAE(nn.Module):

    def __init__(self, x_dim,z_dim,x_mode,t_mode,y_mode,device,encoder,decoder):
        super().__init__()
        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.y_mode = y_mode
        self.t_mode = t_mode
        self.x_mode = x_mode
        
        self.encoder = encoder
        self.decoder = decoder
        self.to(device)
        self.float()

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x, t, y):
        z_pred, z_std = self.encoder(x, t)
        z = self.reparameterize(z_pred, z_std)
        x_pred, x_std, t_pred, t_std, y_pred, y_std = self.decoder(z,t)
        
        return z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std
    
    def sample(self, n):
        z_sample = torch.randn(n, self.z_dim).to(self.device)
        t_sample = dist.Bernoulli(logits=self.decoder.t_nn(z_sample)[:,[0]]).sample()#binary t
        x_pred,x_std,t_pred,t_std,y_pred,y_std = self.decoder(z_sample, t_sample)
        y_sample = dist.Normal(loc=y_pred, scale=y_std).sample()#Continuous y
        x_sample = np.zeros((n, self.x_dim))

        pred_i = 0        #x_pred is much longer than x if x has categorical variables with more categories than 2
        for i,mode in enumerate(self.x_mode):
            if mode==2:
                x_sample[:,i] = dist.Normal(loc=x_pred[:,pred_i], scale=x_std[:,pred_i]).sample().detach().numpy()
                pred_i += 1
            elif mode==1:
                x_sample[:,i] = dist.Bernoulli(logits=x_pred[:,pred_i]).sample().detach().numpy()
                pred_i += 1
            else:
                x_sample[:,i] = dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).sample().detach().numpy()
                pred_i += mode
        
        return z_sample, x_sample, t_sample, y_sample