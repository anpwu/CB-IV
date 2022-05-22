import os
import csv
import pandas as pd
import numpy as np
from utils import set_seed
import scipy

def ihdp2csv(dataDir, exps=3, val_fraction=0.3, targetDir='./Data/data/IHDP/', seed=2021):

    ihdp_train_path = dataDir + 'ihdp_npci_1-100.train.npz'
    ihdp_test_path  = dataDir + 'ihdp_npci_1-100.test.npz'

    set_seed(seed)
    ihdp_train = np.load(ihdp_train_path)
    ihdp_test  = np.load(ihdp_test_path)
    num   = ihdp_train['x'].shape[0]
    x_dim = ihdp_train['x'].shape[1]

    for exp in range(exps):
        print(f'run {exp}/{exps}.')

        train_data = []
        valid_data = []
        test_data = []
        n_valid = int(val_fraction*num)
        n_train = num-n_valid
        I = np.random.permutation(range(0,num))
        I_train = I[:n_train]
        I_valid = I[n_train:]
        train_data.append(ihdp_train['x'][I_train,:,exp])
        valid_data.append(ihdp_train['x'][I_valid,:,exp])
        test_data.append(ihdp_test['x'][:,:,exp])

        for item in ['t','yf','ycf','mu0','mu1']:
            train_data.append(ihdp_train[item][I_train,exp:exp+1])
            valid_data.append(ihdp_train[item][I_valid,exp:exp+1])
            test_data.append(ihdp_test[item][:,exp:exp+1])

        train_data.append(ihdp_train[item][I_train,exp:exp+1]-ihdp_train[item][I_train,exp:exp+1]+ihdp_train['ate'])
        valid_data.append(ihdp_train[item][I_valid,exp:exp+1]-ihdp_train[item][I_valid,exp:exp+1]+ihdp_train['ate'])
        test_data.append(ihdp_test[item][:,exp:exp+1]-ihdp_test[item][:,exp:exp+1]+ihdp_train['ate'])

        train_df = pd.DataFrame(np.concatenate(train_data,1),
                    columns=['x{}'.format(i+1) for i in range(x_dim)] + 
                    ['t','y','f','mu0','mu1','ate'])

        valid_df = pd.DataFrame(np.concatenate(valid_data,1),
                    columns=['x{}'.format(i+1) for i in range(x_dim)] + 
                    ['t','y','f','mu0','mu1','ate'])

        test_df = pd.DataFrame(np.concatenate(test_data,1),
                    columns=['x{}'.format(i+1) for i in range(x_dim)] + 
                    ['t','y','f','mu0','mu1','ate'])

        csvDir = targetDir+f'{exp}/'
        os.makedirs(os.path.dirname(csvDir), exist_ok=True)

        train_df.to_csv(csvDir + 'train.csv', index=False)
        valid_df.to_csv(csvDir + 'val.csv', index=False)
        test_df.to_csv(csvDir + 'test.csv', index=False)

class IHDP_Generator(object):
    def __init__(self, sc=1, sh=0, one=1, VX=1, mV=2, mX=3, mU=3, seed=2021, dataDir='./Data/Causal/', storage_path='./Data/data/',details=False):     
        ihdp_train_path = dataDir + 'ihdp_npci_1-100.train.npz'
        ihdp_test_path  = dataDir + 'ihdp_npci_1-100.test.npz'
        ihdp_train = np.load(ihdp_train_path)
        ihdp_test  = np.load(ihdp_test_path)
        
        self.sc = sc
        self.sh = sh
        self.one = one
        self.VX = VX
        self.m = mV+mX+mU
        self.mV = mV
        self.mX = mX
        self.mU = mU
        self.seed = seed
        self.dataDir = dataDir
        self.storage_path = storage_path
        self.ihdp_train = ihdp_train
        self.ihdp_test = ihdp_test
        
        self.nTrain = round(ihdp_train['x'].shape[0] * 0.7)
        self.nValid = round(ihdp_train['x'].shape[0] * 0.3)
        self.nTest  = ihdp_test['x'].shape[0]
        self.n = self.nTrain + self.nValid + self.nTest
        
        self.set_path()
        self.set_coefs()
        self.get_data(details)
            
        
        
    def set_path(self):
        which_benchmark = 'IHDP_'+'_'.join(str(item) for item in [self.sc, self.sh, self.one, self.VX])
        data_path = self.storage_path+'/data/'+which_benchmark
        which_dataset = '_'.join(str(item) for item in [self.mV, self.mX, self.mU])
        data_path += '/'+which_dataset+'/'
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        self.data_path = data_path
        self.which_benchmark = which_benchmark
        self.which_dataset = which_dataset
        
    def set_coefs(self):
        if self.one:
            self.coefs_VXU = np.ones(shape=self.m)
        else:
            np.random.seed(self.seed * 1)	          # <--
            self.coefs_VXU = np.random.normal(size=self.m)
            
        with open(self.data_path+'coefs.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(self.coefs_VXU)
            
    def get_data(self, details):
        ihdp_train = self.ihdp_train
        ihdp_test  = self.ihdp_test
        
        np.random.seed(self.seed * 2)	          # <--
        V = np.random.multivariate_normal(mean=np.zeros(self.mV), cov=np.eye(self.mV), size=747)
        XU = np.concatenate((ihdp_train['x'][:,:self.mX+self.mU,0], ihdp_test['x'][:,:self.mX+self.mU,0]), 0)
        mu0 = np.concatenate((ihdp_train['mu0'][:,0:1], ihdp_test['mu0'][:,0:1]), 0)
        mu1 = np.concatenate((ihdp_train['mu1'][:,0:1], ihdp_test['mu1'][:,0:1]), 0)
        
        if self.VX:
            T_vars = np.concatenate([V * XU[:, 0:self.mV],XU], axis=1)
        else:
            T_vars = np.concatenate([V,XU], axis=1)
            
        np.random.seed(self.seed * 3)	          # <--	
        z = np.dot(T_vars, self.coefs_VXU)
        
        pi0_t1 = scipy.special.expit( self.sc*(z+self.sh) )
        t = np.array([])
        for p in pi0_t1:
            t = np.append(t, np.random.binomial(1, p, 1))
            
        np.random.seed(self.seed * 4)	          # <--	
        y = np.zeros((self.n, 2))
        y[:,0:1] = mu0 + np.random.normal(loc=0., scale=.01, size=(self.n,1))
        y[:,1:2] = mu1 + np.random.normal(loc=0., scale=.01, size=(self.n,1))
        
        yf = np.array([])
        ycf = np.array([])
        for i, t_i in enumerate(t):
            yf = np.append(yf, y[i, int(t_i)])
            ycf = np.append(ycf, y[i, int(1-t_i)])
            
        if details:
            print('#'*30)
            print('The data path is: {}'.format(self.data_path))
            print('The mean of z/p/t and ATE:')
            print('-'*30)
            print('z: {:.4}'.format(z.mean()))
            print('p: {:.4}'.format(pi0_t1.mean()))
            print('t: {:.4}'.format(t.mean()))  
            print('ate: {:.4}'.format(mu1.mean()-mu0.mean()))  
            print('-'*30)
        
        data = [V,XU,z.reshape(-1,1),pi0_t1.reshape(-1,1),t.reshape(-1,1),mu0,mu1,yf.reshape(-1,1), ycf.reshape(-1,1)]
        data_df = pd.DataFrame(np.concatenate(data, 1),columns=['v{}'.format(i+1) for i in range(self.mV)] + ['x{}'.format(i+1) for i in range(self.mX)] + ['u{}'.format(i+1) for i in range(self.mU)] + ['z','p','t','mu0','mu1','y','f'])
        self.data_df = data_df
    
    def run(self, num_reps=10):
        self.num_reps = num_reps
        np.random.seed(self.seed * 5)	          # <--	

        print('Next, run dataGenerator: ')
        
        for exp in range(num_reps):
            print(f'Run {exp}/{num_reps}. ')
            os.makedirs(os.path.dirname(self.data_path + f'{exp}/'), exist_ok=True)
            df = self.data_df.sample(frac=1).reset_index(drop=True)
            train_df = df[:self.nTrain]
            valid_df = df[self.nTrain:-self.nTest]
            test_df  = df[-self.nTest:]
            train_df.to_csv(self.data_path + f'{exp}/train.csv', index=False)
            valid_df.to_csv(self.data_path + f'{exp}/val.csv', index=False)
            test_df.to_csv(self.data_path + f'{exp}/test.csv', index=False)
        
        print('-'*30)