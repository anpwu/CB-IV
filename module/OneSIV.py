import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from utils import set_seed, log

class OneSIV(nn.Module):
    def __init__(self, v_dim, x_dim, dropout, mode, loss):
        super(OneSIV, self).__init__()
        self.mode = mode
        self.loss = loss

        if self.mode == 'v':
            t_input_dim, y_input_dim = v_dim, x_dim+1
        elif self.mode == 'x':
            t_input_dim, y_input_dim = x_dim, x_dim+1
        elif self.mode == 'vx':
            t_input_dim, y_input_dim = v_dim+x_dim, x_dim+1
        elif self.mode == 'xx':
            t_input_dim, y_input_dim = v_dim+x_dim, v_dim+x_dim+1

        self.t_net = nn.Sequential(nn.Linear(t_input_dim, 128),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(64, 32),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(32, 2))

        self.y_net = nn.Sequential(nn.Linear(y_input_dim, 128),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(32, 1))

    def get_input(self, v, x, t, mode):
        if self.mode == 'v':
            return v, torch.cat((t-t,x), 1), torch.cat((t-t+1,x), 1)
        elif self.mode == 'x':
            return x, torch.cat((t-t,x), 1), torch.cat((t-t+1,x), 1)
        elif self.mode == 'vx':
            return torch.cat((v,x), 1), torch.cat((t-t,x), 1), torch.cat((t-t+1,x), 1)
        elif self.mode == 'xx':
            return torch.cat((v,x), 1), torch.cat((t-t,v,x), 1), torch.cat((t-t+1,v,x), 1)
        
    def forward(self, v, x, t):
        t_input, y0_input, y1_input = self.get_input(v,x,t,self.mode)
        
        pi_t = self.t_net(t_input)
        p_t = F.softmax(pi_t,dim=1)
        y_0 = self.y_net(y0_input)
        y_1 = self.y_net(y1_input)

        if self.loss == 'log':
            y_0 = F.sigmoid(y_0) + 0.0001
            y_1 = F.sigmoid(y_1) + 0.0001

        y = y_0 * p_t[:,0:1] + y_1 * p_t[:,1:2]

        return pi_t, y, y_0, y_1

def loss_fun(pred_y, y):
    return -(torch.log(pred_y) * y + torch.log(1-pred_y) * (1-y) ).mean()

def run(exp, args, dataDir, resultDir, train, val, test, device):
    set_seed(args.seed)
    logfile = f'{resultDir}/log.txt'
    _logfile = f'{resultDir}/OneSIV.txt'
    
    try:
        train.to_tensor()
        val.to_tensor()
        test.to_tensor()
    except:
        pass
    
    if torch.cuda.is_available() and args.use_gpu:
        train.to_cuda()
        val.to_cuda()
        test.to_cuda()

    train_loader = DataLoader(train, batch_size=args.onesiv_batch_size)

    OneSIV_dict = {
        'v_dim':args.mV, 
        'x_dim':args.mX, 
        'dropout':args.onesiv_dropout,
        'mode':args.mode,
        'loss':args.onesiv_loss,
    }

    log(logfile, f'Train OneSIV: mode {args.mode}, dropout {args.onesiv_dropout}. ')
    log(_logfile, f'Train OneSIV: mode {args.mode}, dropout {args.onesiv_dropout}. ', False)

    net = OneSIV(**OneSIV_dict)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.onesiv_lr, betas=(args.onesiv_beta1, args.onesiv_beta2),eps=args.onesiv_eps)
    t_loss = torch.nn.CrossEntropyLoss()

    if args.onesiv_loss == 'log':
        y_loss = loss_fun
    else:
        y_loss = torch.nn.MSELoss()

    obj_save = 9999
    obj_ate = None
    mse_save = 9999
    mse_ate = None
    final_ate = None
    res20_ate = None
    for epoch in range(args.onesiv_epochs):
        net.train()

        for idx, inputs in enumerate(train_loader):
            v = inputs['v'].to(device)
            x = inputs['x'].to(device)
            t = inputs['t'].to(device)
            y = inputs['y'].to(device)

            pred_t, pred_y, _, _ = net(v,x,t) 
            loss = args.onesiv_w1 * y_loss(pred_y, y) + args.onesiv_w2 * t_loss(pred_t, t.reshape(-1).type(torch.LongTensor).to(device))

            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step()   


        net.eval()

        pi_t,_,mu0_train, mu1_train = net(train.v,train.x,train.t) 
        _,_,mu0_test, mu1_test = net(test.v,test.x,test.t) 
        pred_ate_train = (mu1_train.mean() - mu0_train.mean()).item()
        pred_ate_test = (mu1_test.mean() - mu0_test.mean()).item()

        final_ate = [pred_ate_train, pred_ate_test]

        pred_t_val, pred_y_val, _, _ = net(val.v,val.x,val.t) 
        y_loss_val = y_loss(pred_y_val, val.y)
        loss_val = args.onesiv_w1 * y_loss(pred_y_val, val.y) + args.onesiv_w2 * t_loss(pred_t_val, val.t.reshape(-1).type(torch.LongTensor).to(device))

        if y_loss_val < mse_save:
            mse_save = y_loss_val
            mse_ate = [pred_ate_train, pred_ate_test]

        if loss_val < obj_save:
            obj_save = loss_val
            obj_ate = [pred_ate_train, pred_ate_test]

        if epoch == 19:
            res20_ate = [pred_ate_train, pred_ate_test]

        if args.verbose:
            if epoch % args.epoch_show == 0 or epoch == args.onesiv_epochs-1:
                log(logfile, 'epoch-{}: val_y_mse {:.4f}, val_loss {:.4f}, train_ate {:.4f}, test_ate {:.4f}'.format(epoch, y_loss_val, loss_val, pred_ate_train, pred_ate_test))
                log(_logfile, 'epoch-{}: val_y_mse {:.4f}, val_loss {:.4f}, train_ate {:.4f}, test_ate {:.4f}'.format(epoch, y_loss_val, loss_val, pred_ate_train, pred_ate_test), False)
    return mse_ate, obj_ate, res20_ate, final_ate