import numpy as np
from .moduleXT import Encoder, Decoder, CEVAE
import torch
import torch.distributions as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from module.utils import set_seed, cat

def kld_loss(mu, std):
    #Note that the sum is over the dimensions of z as well as over the units in the batch here
    var = std.pow(2)
    kld = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
    return kld

def kld_loss_binary(z_pred, pz_logit):
    probs = torch.sigmoid(z_pred)
    prior = torch.sigmoid(pz_logit)
    kld = (probs*torch.log(probs/prior) + (1-probs)*torch.log((1-probs)/(1-prior))).sum()
    return kld

def get_losses(z_mean, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std,
                  x, t, y, x_mode, t_mode, y_mode, kl_scaling=1):

    kld = kld_loss(z_mean,z_std)*kl_scaling
    x_loss = 0
    t_loss = 0
    y_loss = 0
    pred_i = 0 # x_pred is much longer than x if x has categorical variables with more categories than 2
    for i,mode in enumerate(x_mode):
        if mode==2:
            x_loss += -dist.Normal(loc=x_pred[:,pred_i],scale=x_std[:,pred_i]).log_prob(x[:,i]).sum()
            pred_i += 1
        elif mode==1:
            x_loss += -dist.Bernoulli(logits=x_pred[:,pred_i]).log_prob(x[:,i]).sum()
            pred_i += 1
        else:
            x_loss += -dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).log_prob(x[:,i]).sum()
            pred_i += mode
    if t_mode==[1]:
        t_loss = -dist.Bernoulli(logits=t_pred).log_prob(t).sum()
    elif t_mode==[2]:
        t_loss = -dist.Normal(loc=t_pred,scale=t_std).log_prob(t).sum()
    else:
        t_loss = -dist.Categorical(logits=t_pred).log_prob(t[:,0]).sum()
    if y_mode ==[1]:
        y_loss = -dist.Bernoulli(logits=y_pred).log_prob(y).sum()
    elif y_mode == [2]:
        y_loss = -dist.Normal(loc=y_pred,scale=y_std).log_prob(y).sum()
    else:
        y_loss = -dist.Categorical(logits=y_pred).log_prob(y[:,0]).sum()
    return kld, x_loss, t_loss, y_loss

def get_losses_binary(z_pred, x_pred, x_std, t_pred, t_std, y_pred, y_std, x, t, y, pz_logit, x_mode, t_mode, y_mode):
    kld = kld_loss_binary(z_pred, pz_logit)
    qz_probs = torch.cat([1-torch.sigmoid(z_pred),torch.sigmoid(z_pred)],0).squeeze()
    x = torch.cat([x,x],0)#Purpose is to evaluate for both parts of the expected value, t_pred etc. should be prepared for this
    y = torch.cat([y,y],0)
    t = torch.cat([t,t],0)
    x_loss = 0
    t_loss = 0
    y_loss = 0
    pred_i = 0
    for i,mode in enumerate(x_mode):
        if mode==2:
            x_loss += -(dist.Normal(loc=x_pred[:,pred_i],scale=x_std[:,pred_i]).log_prob(x[:,i])*qz_probs).sum()
            pred_i += 1
        elif mode==1:
            x_loss += -(dist.Bernoulli(logits=x_pred[:,pred_i]).log_prob(x[:,i])*qz_probs).sum()
            pred_i += 1
        else:
            x_loss += -(dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).log_prob(x[:,i])*qz_probs).sum()
            pred_i += mode
    if t_mode==[1]:
        t_loss = -(dist.Bernoulli(logits=t_pred).log_prob(t).squeeze()*qz_probs).sum()
    elif t_mode==[2]:
        t_loss = -(dist.Normal(loc=t_pred,scale=t_std).log_prob(t).squeeze()*qz_probs).sum()
    else:
        t_loss = -(dist.Categorical(logits=t_pred).log_prob(t[:,0]).squeeze()*qz_probs).sum()
    if y_mode ==[1]:
        y_loss = -(dist.Bernoulli(logits=y_pred).log_prob(y).squeeze()*qz_probs).sum()
    elif y_mode == [2]:
        y_loss = -(dist.Normal(loc=y_pred,scale=y_std).log_prob(y).squeeze()*qz_probs).sum()
    else:
        y_loss = -(dist.Categorical(logits=y_pred).log_prob(y[:,0]).squeeze()*qz_probs).sum()
    return kld, x_loss, t_loss, y_loss

class VAEXT(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'VAEXT',
                    'device': 'cpu',
                    'q_z_nn_layers': 3,
                    'q_z_nn_width': 64,
                    'latent_dim': 5,
                    'latent_mode': 2,
                    'x_mode': [2,7],
                    't_mode': [2],
                    'y_mode': [2],
                    'lr_start': 0.0001,
                    'lr_end': 0.00001,
                    'epochs': 300,
                    'print_logs': True,
                    'x_loss_scaling': 1,
                    'show_per_epoch': 10,
                    'save_per_epoch': 100,
                    'batch_size':200,
                    'seed': 2022,   
                    'save_path': './results/'
                    }

    def set_Configuration(self, config):
        self.config = config

    def fit(self, data, exp=-1, config=None):
        if config is None:
            config = self.config
        
        
        device = torch.device(config['device'])
        batch_size = config['batch_size']
        latent_dim = config['latent_dim']
        q_z_nn_layers = config['q_z_nn_layers']
        q_z_nn_width = config['q_z_nn_width']
        # 2 for continuous (Gaussian), 1 for binary, 3 or more for categorical distributions
        x_mode = config['x_mode']
        t_mode = config['t_mode']
        y_mode = config['y_mode']
        latent_mode = config['latent_mode']
        lr_start=config['lr_start']
        lr_end=config['lr_end']
        epochs=config['epochs']
        print_logs=config['print_logs']
        x_loss_scaling=config['x_loss_scaling']
        seed = config['seed']
        show_per_epoch = config['show_per_epoch']
        save_path = config['save_path']
        save_per_epoch = config['save_per_epoch']

        self.data = data
        self.device = device
        self.x_mode = x_mode
        self.t_mode = t_mode
        self.y_mode = y_mode
        self.x_dim = data.train.x.shape[1]
        self.save_path = save_path

        set_seed(seed)
        data.tensor()
        data.to(device)
        
        train_loader = DataLoader(data.train, shuffle=True, batch_size=batch_size)

        encoder = Encoder(self.x_dim,latent_dim,q_z_nn_layers,q_z_nn_width,device,False)
        decoder = Decoder(self.x_dim,latent_dim,x_mode,t_mode,y_mode,3,64,3,64,3,64,device,False)
        model = CEVAE(self.x_dim,latent_dim,x_mode,t_mode,y_mode,device,encoder,decoder)
        self.model = model

        optimizer = Adam(model.parameters(), lr=lr_start)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/epochs))
        
        losses = {"total": [], "kld": [], "x": [], "t": [], "y": []}
        self.losses = losses
        
        kl_scalings = []
        kl_scaling_schedule=None
        if kl_scaling_schedule:
            for i in range(epochs):
                index = 0
                for j in range(len(kl_scaling_schedule[0])):
                    if i/epochs >= kl_scaling_schedule[0][j]:
                        index = j
                kl_scaling = ((kl_scaling_schedule[0][index+1]-i/epochs)*kl_scaling_schedule[1][index] + (i/epochs-kl_scaling_schedule[0][index])*kl_scaling_schedule[1][index+1])/(kl_scaling_schedule[0][index+1]-kl_scaling_schedule[0][index])
                kl_scalings.append(kl_scaling)
        
        for epoch in range(epochs):
            #i = 0
            epoch_loss = 0
            epoch_kld_loss = 0
            epoch_x_loss = 0
            epoch_t_loss = 0
            epoch_y_loss = 0
            if print_logs:
                if kl_scaling_schedule:
                    print("KL scaling: {}".format(kl_scalings[epoch]))
            for dataitem in train_loader:
                x = dataitem['x'].to(device)
                t = dataitem['t'].to(device)
                y = dataitem['y'].to(device)
                z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std = model(x,t,y)
                if latent_mode == 2:
                    if kl_scaling_schedule is not None:
                        kld, x_loss, t_loss, y_loss = get_losses(z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std, x, t, y,
                                                            x_mode, t_mode, y_mode, kl_scalings[epoch])
                    else:
                        # print(x.shape)
                        kld, x_loss, t_loss, y_loss = get_losses(z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std, x, t, y,
                                                            x_mode, t_mode, y_mode)
                elif latent_mode == 1:
                    kld, x_loss, t_loss, y_loss = get_losses_binary(z_pred, x_pred, x_std, t_pred, t_std, y_pred, y_std, x, t, y,
                                                            model.pz_logit, x_mode, t_mode, y_mode)
                x_loss *= x_loss_scaling
                loss = kld + x_loss + t_loss + y_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #i += 1
                #if i%100 == 0 and print_logs:
                #    print("Sample batch loss: {}".format(loss))
                epoch_loss += loss.item()
                epoch_kld_loss += kld.item()
                epoch_x_loss += x_loss.item()
                epoch_t_loss += t_loss.item()
                epoch_y_loss += y_loss.item()
                # print(kld, x_loss, t_loss, y_loss)
            if epoch % show_per_epoch == 0:
                print("Epoch - {},  loss: {}".format(epoch, loss))
                t1,t2,t3,t4 = self.evaluation(data.train)
                print("train {:.4f} {:.4f} {:.4f} {:.4f}".format(t1,t2,t3,t4))
                l1,l2,l3,l4 = self.evaluation(data.valid)
                print("valid {:.4f} {:.4f} {:.4f} {:.4f}".format(l1,l2,l3,l4))
                with open("data.txt","a+") as f:
                    f.write("Epoch - {},  loss: {:.4f}, \n".format(epoch, loss))
                    f.write("train {:.4f} {:.4f} {:.4f} {:.4f} \n".format(t1,t2,t3,t4))
                    f.write("valid {:.4f} {:.4f} {:.4f} {:.4f} \n".format(l1,l2,l3,l4))

            if epoch % save_per_epoch == 0:
                self.save_latent(data,epoch)
            
            losses['total'].append(epoch_loss)
            losses['kld'].append(epoch_kld_loss)
            losses['x'].append(epoch_x_loss)
            losses['t'].append(epoch_t_loss)
            losses['y'].append(epoch_y_loss)

            scheduler.step()

        return 

    def evaluation(self, data):
        x_mode = self.x_mode
        t_mode = self.t_mode
        y_mode = self.y_mode
        z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std = self.model(data.x,data.t,data.y)
        kld, x_loss, t_loss, y_loss = get_losses(z_pred, z_std, x_pred, x_std, t_pred, t_std, 
                                 y_pred, y_std, data.x,data.t,data.y, x_mode, t_mode, y_mode)
        return kld.item(), x_loss.item(), t_loss.item(), y_loss.item()

    def predict(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std = self.model(x,t,None)

        return y_pred.detach().cpu().numpy()

    def detail_rlt(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t
        
        z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std = self.model(x,t,None)
        result_list = [z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std]
        for i in range(len(result_list)):
            result_list[i] = result_list[i].detach().cpu().numpy()
        return result_list

    def save_latent(self, data, epoch):
        data_path = f'{self.save_path}{epoch}_result.npz'

        trt_rlt = self.detail_rlt(data.train)
        val_rlt = self.detail_rlt(data.valid)
        tst_rlt = self.detail_rlt(data.test)

        np.savez(data_path, trt_rlt=trt_rlt[0], val_rlt=val_rlt[0], tst_rlt=tst_rlt[0],
                trt_rlt_std=trt_rlt[1], val_rlt_std=val_rlt[1], tst_rlt_std=tst_rlt[1])

    def ITE(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        _, _, _, _, _, _, y_pred_0, _ = self.model(x,t-t,None)
        _, _, _, _, _, _, y_pred_1, _ = self.model(x,t-t+1,None)
        _, _, _, _, _, _, y_pred_t, _ = self.model(x,t,None)

        ITE_0 = y_pred_0.detach().cpu().numpy()
        ITE_1 = y_pred_1.detach().cpu().numpy()
        ITE_t = y_pred_t.detach().cpu().numpy()

        return ITE_0,ITE_1,ITE_t

    def ATE(self, data=None, t=None, x=None):
        ITE_0,ITE_1,ITE_t = self.ITE(data,t,x)

        return np.mean(ITE_1-ITE_0), np.mean(ITE_t-ITE_0)