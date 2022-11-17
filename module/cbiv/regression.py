import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from module.utils import set_seed

def log(logfile,str,out=True):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str+'\n')
    if out:
        print(str)

def get_gain(activation):
    if activation.__class__.__name__ == "LeakyReLU":
        gain = nn.init.calculate_gain("leaky_relu",
                                            activation.negative_slope)
    else:
        activation_name = activation.__class__.__name__.lower()
        try:
            gain = nn.init.calculate_gain(activation_name)
        except ValueError:
            gain = 1.0
    return gain

class MLPModel(nn.Module):
    def __init__(self, input_dim, layer_widths, activation=None,last_layer=None, num_out=1):
        nn.Module.__init__(self)
        self.gain=get_gain(activation)

        if len(layer_widths) == 0:
            layers = [nn.Linear(input_dim, num_out)]
        else:
            num_layers = len(layer_widths)
            if activation is None:
                layers = [nn.Linear(input_dim, layer_widths[0])]
            else:
                layers = [nn.Linear(input_dim, layer_widths[0]), activation]
            for i in range(1, num_layers):
                w_in = layer_widths[i-1]
                w_out = layer_widths[i]
                if activation is None:
                    layers.extend([nn.Linear(w_in, w_out)])
                else:
                    layers.extend([nn.Linear(w_in, w_out), activation])
            layers.append(nn.Linear(layer_widths[-1], num_out))
        if last_layer:
            layers.append(last_layer)
        self.model = nn.Sequential(*layers)

    def initialize(self, gain=1.0):
        for layer in self.model[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=self.gain)
                nn.init.zeros_(layer.bias.data)
        final_layer = self.model[-1]
        nn.init.xavier_normal_(final_layer.weight.data, gain=gain)
        nn.init.zeros_(final_layer.bias.data)

    def forward(self, data):
        # print(data.shape)
        num_data = data.shape[0]
        data = data.view(num_data, -1)
        return self.model(data)

class MultipleMLPModel(nn.Module):
    def __init__(self, input_dim, layer_widths, num_models=1, activation=None,last_layer=None, num_out=1):
        nn.Module.__init__(self)
        self.models = nn.ModuleList([MLPModel(
            input_dim, layer_widths, activation=activation,
            last_layer=last_layer, num_out=num_out) for _ in range(num_models)])
        self.num_models = num_models

    def forward(self, data):
        num_data = data.shape[0]
        data = data.view(num_data, -1)
        outputs = [self.models[i](data) for i in range(self.num_models)]
        return torch.cat(outputs, dim=1)


class CBIV4T(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'CBIV4T',
                    'epochs': 20,
                    'batch_size':500,
                    'device': 'cpu',
                    'learning_rate': 5e-3,
                    'seed': 2022,   
                    'save_path': './results/',
                    'output_delay': 20,
                    }

    def set_Configuration(self, config):
        self.config = config

    def fit(self, data, exp=-1, config=None):
        if config is None:
            config = self.config

        batch_size = config['batch_size']
        lr = config['learning_rate']
        num_epoch = config['epochs']
        seed = config['seed']
        resultDir = config['save_path']
        device = torch.device(config['device'])
        
        set_seed(seed)

        logfile = f'{resultDir}/log.txt'
        _logfile = f'{resultDir}/Regression.txt'


        data.tensor()
        data.to(device)
        train_loader = DataLoader(data.train, shuffle=True, batch_size=batch_size)

        input_dim = data.train.x.shape[1]
        mlp = MLPModel(input_dim, layer_widths=[128, 64], activation=nn.ReLU(),last_layer=None, num_out=1)
        net = nn.Sequential(mlp)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        loss_func = torch.nn.MSELoss()

        for epoch in range(num_epoch):
            log(logfile, f"Exp {exp} :this is the {epoch}/{num_epoch} epochs.")
            log(_logfile, f"Exp {exp} :this is the {epoch}/{num_epoch} epochs.", False)
            for idx, inputs in enumerate(train_loader):
                x = inputs['x']
                t = inputs['t']

                prediction = net(x) 
                loss = loss_func(prediction, t)

                optimizer.zero_grad()   # 清空上一步的残余更新参数值
                loss.backward()         # 误差反向传播, 计算参数更新值
                optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

            log(logfile, 'The train mse: {:.2f}'.format(loss_func(net(data.train.x),data.train.t).item()))
            log(_logfile, 'The valid mse: {:.2f}'.format(loss_func(net(data.valid.x),data.valid.t).item()))

        train_that = net(data.train.x).cpu().detach().numpy()
        valid_that = net(data.valid.x).cpu().detach().numpy()
        test_that  = net(data.test.x ).cpu().detach().numpy()

        self.train_that = train_that
        self.valid_that = valid_that
        self.test_that = test_that
        np.savez(f'{resultDir}/Regression.npz', train_that=train_that, 
                valid_that=valid_that, test_that=test_that)

        return train_that, valid_that, test_that

    def predict(self, data=None, t=None, x=None):
        return self.train_that, self.valid_that, self.test_that