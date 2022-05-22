import numpy as np
import pandas as pd
import scipy.special
import csv
import sys
import os
from scipy.stats import norm

from utils import * 

class Syn_Generator(object):
    def __init__(self, n,ate,sc,sh,one,depX,depU,VX,mV,mX,mU,init_seed=7,seed_coef=10,details=0,storage_path='./Data/'):
        self.n = n
        self.ate = ate
        self.sc = sc
        self.sh = sh
        self.depX = depX
        self.depU = depU
        self.one = one
        self.VX = VX
        self.mV = mV
        self.mX = mX
        self.mU = mU
        self.seed = init_seed
        self.seed_coef = seed_coef
        self.storage_path = storage_path

        assert mV<=mX, 'Assume: the dimension of the IVs is less than Confounders'
        
        if one:
            self.coefs_VXU = np.ones(shape=mV+mX+mU)
            self.coefs_XU0 = np.ones(shape=mX+mU)
            self.coefs_XU1 = np.ones(shape=mX+mU)
        else:
            np.random.seed(1*seed_coef*init_seed+3)	          # <--
            self.coefs_VXU = np.random.normal(size=mV+mX+mU)
            
            np.random.seed(2*seed_coef*init_seed+5)	# <--
            self.coefs_XU0 = np.random.normal(size=mX+mU)
            self.coefs_XU1 = np.random.normal(size=mX+mU)
            

        self.set_path(details)
        
        with open(self.data_path+'coefs.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(self.coefs_VXU)
            csv_writer.writerow(self.coefs_XU0)
            csv_writer.writerow(self.coefs_XU1)
        
        mu, sig = self.get_normal_params(mV, mX, mU, depX, depU)
        self.set_normal_params(mu, sig)

        with open(self.data_path+'norm.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(mu)
            for row in range(len(sig)):
                csv_writer.writerow(sig[row])

    def get_normal_params(self, mV, mX, mU, depX, depU):
        m = mV + mX + mU
        mu = np.zeros(m)
        
        sig = np.eye(m)
        temp_sig = np.ones(shape=(m-mV,m-mV))
        temp_sig = temp_sig * depU
        sig[mV:,mV:] = temp_sig

        sig_temp = np.ones(shape=(mX,mX)) * depX
        sig[mV:-mU,mV:-mU] = sig_temp

        sig[np.diag_indices_from(sig)] = 1

        return mu, sig

    def set_normal_params(self, mu, sig):
        self.mu = mu
        self.sig = sig
            
    def set_path(self,details):
        which_benchmark = 'Syn_'+'_'.join(str(item) for item in [self.sc, self.sh, self.one, self.depX, self.depU,self.VX])
        data_path = self.storage_path+'/data/'+which_benchmark
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        which_dataset = '_'.join(str(item) for item in [self.mV, self.mX, self.mU])
        data_path += '/'+which_dataset+'/'
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        self.data_path = data_path
        self.which_benchmark = which_benchmark
        self.which_dataset = which_dataset

        if details:
            print('#'*30)
            print('The data path is: {}'.format(self.data_path))
            print('The ATE:')
            print('-'*30)
            print(f'ate: {1+self.ate}')  
            print('-'*30)
        
    def run(self, n=None, num_reps=10):
        self.num_reps = num_reps
        
        mu = self.mu
        sig = self.sig
        seed_coef = self.seed_coef
        init_seed = self.seed

        if n is None:
            n = self.n

        print('Next, run dataGenerator: ')

        for perm in range(num_reps):
            print(f'Run {perm}/{num_reps}. ')
            train_dict, train_df = self.get_data(n, mu, sig, 3*seed_coef*init_seed+perm+777)
            val_dict, val_df = self.get_data(n, mu, sig, 4*seed_coef*init_seed+perm+777)
            test_dict, test_df = self.get_data(n, mu, sig, 5*seed_coef*init_seed+perm+777)
            all_df = train_df.append([val_df, test_df])

            data_path = self.data_path + '/{}/'.format(perm)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            os.makedirs(os.path.dirname(data_path+'info/'), exist_ok=True)
        
            train_df.to_csv(data_path + '/train.csv', index=False)
            val_df.to_csv(data_path + '/val.csv', index=False)
            test_df.to_csv(data_path + '/test.csv', index=False)

            num_pts = 250
            plot(train_dict['z'][:num_pts], train_dict['pi'][:num_pts], train_dict['t'][:num_pts], train_dict['y'][:num_pts],data_path)

            with open(data_path+'info/specs.csv'.format(perm), 'a') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                temp = [np.mean(all_df['t'].values), np.min(all_df['pi'].values), np.max(all_df['pi'].values), np.mean(all_df['pi'].values), np.std(all_df['pi'].values)]
                temp.append(lindisc_np(get_var_df(all_df,'x'), all_df['t'].values, np.mean(all_df['t'].values)))
                csv_writer.writerow(temp)
                
            with open(data_path+'info/mu.csv'.format(perm), 'w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                E_T_C = ACE(get_var_df(train_df,'m'),train_df['t'].values)
                csv_writer.writerow(E_T_C)
                E_T_C = ACE(get_var_df(val_df,'m'),val_df['t'].values)
                csv_writer.writerow(E_T_C)
                E_T_C = ACE(get_var_df(val_df,'m'),val_df['t'].values)
                csv_writer.writerow(E_T_C)

        print('-'*30)
            
    def get_data(self, n, mu, sig, seed):
        np.random.seed(seed)

        mV = self.mV
        mX = self.mX
        mU = self.mU
        
        temp = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
        V = temp[:, 0:mV]
        X = temp[:, mV:mV+mX]
        U = temp[:, mV+mX:mV+mX+mU]

        if self.VX:
            T_vars = np.concatenate([V * X[:, 0:mV],X,U], axis=1)
        else:
            T_vars = np.concatenate([V,X,U], axis=1)
        Y_vars = np.concatenate([X,U], axis=1)
        
        np.random.seed(2*seed)	                # <--------------
        z = np.dot(T_vars, self.coefs_VXU)
        pi0_t1 = scipy.special.expit( self.sc*(z+self.sh) )
        t = np.array([])
        for p in pi0_t1:
            t = np.append(t, np.random.binomial(1, p, 1))
            
        mu_0 = np.dot(Y_vars**1, self.coefs_XU0) / (mX+mU)
        mu_1 = np.dot(Y_vars**2, self.coefs_XU1) / (mX+mU) + self.ate
        
        np.random.seed(3*seed)	                # <--------------
        y = np.zeros((n, 2))
        y[:,0] = mu_0 + np.random.normal(loc=0., scale=.01, size=n)
        y[:,1] = mu_1 + np.random.normal(loc=0., scale=.01, size=n)

        yf = np.array([])
        ycf = np.array([])
        for i, t_i in enumerate(t):
            yf = np.append(yf, y[i, int(t_i)])
            ycf = np.append(ycf, y[i, int(1-t_i)])
            
        data_dict = {'V':V, 'X':X, 'U':U, 'z':z, 'pi':pi0_t1, 't':t, 'mu0':mu_0, 'mu1':mu_1, 'yf':yf, 'y':y, 'ycf':ycf}
        data_all = np.concatenate([V, X, U, z.reshape(-1,1), pi0_t1.reshape(-1,1), t.reshape(-1,1), mu_0.reshape(-1,1), mu_1.reshape(-1,1), yf.reshape(-1,1), ycf.reshape(-1,1)], axis=1)
        data_df = pd.DataFrame(data_all,
                               columns=['v{}'.format(i+1) for i in range(V.shape[1])] + 
                               ['x{}'.format(i+1) for i in range(X.shape[1])] + 
                               ['u{}'.format(i+1) for i in range(U.shape[1])] + 
                               ['z','pi','t','mu0','mu1','y','f'])
        
        return data_dict, data_df
