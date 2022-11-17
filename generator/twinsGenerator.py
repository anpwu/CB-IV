import os
import csv
import pandas as pd
import numpy as np
import scipy

class Twins_Generator(object):
    def __init__(self, sc=1, sh=-2, one=1, VX=1, mV=5, mX=5, mU=3, seed=2021, dataDir='./Data/Causal/', storage_path='./Data/data/',details=False):    
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
        
        self.get_df(dataDir + 'twins.csv')
        self.nTrain = round(self.n * 0.63)
        self.nValid = round(self.n * 0.27)
        self.nTest  = self.n - self.nTrain - self.nValid

        self.set_path()
        self.set_coefs()
        self.get_data(details)
        
    def get_df(self, dataPath):
        df = pd.read_csv(dataPath)
        
        # step.1 Return the same sex twins weighing less than 2000g
        df = df[df['dbirwt_1'] < 2000]
        ate = df['mort_1'].mean() - df['mort_0'].mean()

        # step.2 Return object with labels on given axis omitted where alternately any or all of the data are missing
        df = df.dropna()
        
        self.ate = ate
        self.df = df
        self.n = len(df)
        
    def set_path(self):
        which_benchmark = 'Twins_'+'_'.join(str(item) for item in [self.sc, self.sh, self.one, self.VX])
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
        df = self.df
        mV = self.mV
        mX = self.mX
        mU = self.mU
        
        np.random.seed(self.seed * 2)	          # <--
        w0 = df.values[:,0:1]
        w1 = df.values[:,1:2]
        mu0 =  df.values[:,2:3]
        mu1 = df.values[:,3:4]
        V = np.random.multivariate_normal(mean=np.zeros(mV), cov=np.eye(mV), size=len(df))
        XU = df.values[:,4:4+mX+mU]
        
        XU = XU / XU.max(0)

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
        y[:,0:1] = mu0
        y[:,1:2] = mu1
        
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
            print('ate: {:.4}'.format(self.ate))  
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
            test_df.to_csv(self.data_path  + f'{exp}/test.csv', index=False)

        print('-'*30)