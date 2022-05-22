import pandas as pd
import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import Dataset, DataLoader

def get_var_df(df,var):
    var_cols = [c for c in df.columns if c.startswith(var)]
    return df[var_cols].to_numpy()
        
class CausalDataset(Dataset):
    def __init__(self, df, variables = ['u','x','v','z','p','m','t','y','f','c'], observe_vars=['v', 'x']):
        self.length = len(df)
        self.variables = variables
        
        for var in variables:
            exec(f'self.{var}=get_var_df(df, \'{var}\')')
        
        observe_list = []
        for item in observe_vars:
            exec(f'observe_list.append(self.{item})')
        self.c = np.concatenate(observe_list, axis=1)
            
    def to_cpu(self):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.cpu()')
            
    def to_cuda(self,n=0):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.cuda({n})')
    
    def to_tensor(self):
        for var in self.variables:
            exec(f'self.{var} = torch.Tensor(self.{var})')
            
    def to_double(self):
        for var in self.variables:
            exec(f'self.{var} = torch.Tensor(self.{var}).double()')
            
    def to_numpy(self):
        try:
            self.detach()
            self.to_cpu()
        except:
            self.to_cpu()
        for var in self.variables:
            exec(f'self.{var} = self.{var}.numpy()')
            
    def to_pandas(self):
        var_list = []
        var_dims = []
        var_name = []
        for var in self.variables:
            exec(f'var_list.append(self.{var})')
            exec(f'var_dims.append(self.{var}.shape[1])')
        for i in range(len(self.variables)):
            for d in range(var_dims[i]):
                var_name.append(self.variables[i]+str(d))
        df = pd.DataFrame(np.concatenate(var_list, axis=1),columns=var_name)
        return df
    
    def detach(self):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.detach()')
        
    def __getitem__(self, idx):
        var_dict = {}
        for var in self.variables:
            exec(f'var_dict[\'{var}\']=self.{var}[idx]')
        
        return var_dict

    def __len__(self):
        return self.length