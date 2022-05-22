import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import set_seed, log

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

def run(exp, args, dataDir, resultDir, train, val, test, device):
    batch_size = args.regt_batch_size
    lr = args.regt_lr
    num_epoch = args.regt_num_epoch
    logfile = f'{resultDir}/log.txt'
    _logfile = f'{resultDir}/Regression.txt'
    set_seed(args.seed)

    try:
        train.to_tensor()
        val.to_tensor()
        test.to_tensor()
    except:
        pass

    train_loader = DataLoader(train, batch_size=batch_size)

    if args.mode == 'v':
        input_dim = args.mV
        train_input = train.v
        val_input = val.v
        test_input = test.v
    elif args.mode == 'x':
        input_dim = args.mX
        train_input = train.x
        val_input = val.x
        test_input = test.x
    else:
        input_dim = args.mV + args.mX
        train_input = torch.cat((train.v, train.x),1)
        val_input = torch.cat((val.v, val.x),1)
        test_input = torch.cat((test.v, test.x),1)

    mlp = MLPModel(input_dim, layer_widths=[128, 64], activation=nn.ReLU(),last_layer=nn.BatchNorm1d(2), num_out=2)
    net = nn.Sequential(mlp)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        log(logfile, f"Exp {exp} :this is the {epoch}/{num_epoch} epochs.")
        log(_logfile, f"Exp {exp} :this is the {epoch}/{num_epoch} epochs.", False)
        for idx, inputs in enumerate(train_loader):
            u = inputs['u']
            v = inputs['v']
            x = inputs['x']
            t = inputs['t'].reshape(-1).type(torch.LongTensor)

            if args.mode == 'v':
                input_batch = v
            elif args.mode == 'x':
                input_batch = x
            else:
                input_batch = torch.cat((v, x),1)

            prediction = net(input_batch) 
            loss = loss_func(prediction, t)

            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()    

        log(logfile, 'The train accuracy: {:.2f} %'.format((sum(train.t.reshape(-1) == torch.max(F.softmax(net(train_input) , dim=1), 1)[1])/len(train.t)).item() * 100))
        log(_logfile, 'The test  accuracy: {:.2f} %'.format((sum(test.t.reshape(-1) == torch.max(F.softmax(net(test_input) , dim=1), 1)[1])/len(test.t)).item() * 100))

    train.s = F.softmax(net(train_input) , dim=1)[:,1:2]
    val.s = F.softmax(net(val_input) , dim=1)[:,1:2]
    test.s = F.softmax(net(test_input) , dim=1)[:,1:2]

    os.makedirs(os.path.dirname(dataDir + f'{exp}/{args.mode}/'), exist_ok=True)

    train.to_cpu()
    train.detach()
    tmp_df = train.to_pandas()
    tmp_df.to_csv(dataDir + f'{exp}/{args.mode}/train.csv', index=False)

    val.to_cpu()
    val.detach()
    tmp_df = val.to_pandas()
    tmp_df.to_csv(dataDir + f'{exp}/{args.mode}/val.csv', index=False)

    test.to_cpu()
    test.detach()
    tmp_df = test.to_pandas()
    tmp_df.to_csv(dataDir + f'{exp}/{args.mode}/test.csv', index=False)

    return train,val,test