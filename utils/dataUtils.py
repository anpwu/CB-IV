import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def get_multivariate_normal_params(dep, m, seed=0):
    np.random.seed(seed)

    if dep:
        mu = np.zeros(shape=m)
        ''' sample random positive semi-definite matrix for cov '''
        temp = np.random.uniform(size=(m,m))
        temp = .5*(np.transpose(temp)+temp)
        sig = (temp + m*np.eye(m))/10.
    else:
        mu = np.zeros(m)
        sig = np.eye(m)

    return mu, sig

def get_latent(n, m, dep, seed):
    L = np.array((n*[[]]))
    if m != 0:
        mu, sig = get_multivariate_normal_params(dep, m, seed)

        L = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
    return L

def ACE(mu,t):
    mu0 = mu[:,0:1]
    mu1 = mu[:,1:2]
        
    it = np.where(t>0.5)
    ic = np.where(t<0.5)

    mu0_t = mu0[it]
    mu1_t = mu1[it]
    
    mu0_c = mu0[ic]
    mu1_c = mu1[ic]

    return [np.mean(mu0),np.mean(mu1),np.mean(mu1)-np.mean(mu0),np.mean(mu0_t),np.mean(mu1_t),np.mean(mu1_t)-np.mean(mu0_t),np.mean(mu0_c),np.mean(mu1_c),np.mean(mu1_c)-np.mean(mu0_c)]
    
def lindisc_np(X,t,p):
    ''' Linear MMD '''

    it = np.where(t>0)
    ic = np.where(t<1)

    Xc = X[ic]
    Xt = X[it]

    mean_control = np.mean(Xc,axis=0)
    mean_treated = np.mean(Xt,axis=0)

    c = np.square(2*p-1)*0.25
    f = np.sign(p-0.5)

    mmd = np.sum(np.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + np.sqrt(c + mmd)

    return mmd

def plot(z, pi0_t1, t, y, data_path):
    gridspec.GridSpec(3,1)

    z_min = np.min(z) #- np.std(z)
    z_max = np.max(z) #+ np.std(z)
    z_grid = np.linspace(z_min, z_max, 100)

    ax = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ind = np.where(t==0)
    plt.plot(z[ind], np.squeeze(y[ind,0]), '+', color='r')
    ind = np.where(t==1)
    plt.plot(z[ind], np.squeeze(y[ind,1]), '.', color='b')
    plt.legend(['t=0', 't=1'])

    ax = plt.subplot2grid((3,1), (2,0), rowspan=1)
    ind = np.where(t==0)
    mu, std = norm.fit(z[ind])
    p = norm.pdf(z_grid, mu, std)
    plt.plot(z_grid, p, color='r', linewidth=2)
    ind = np.where(t==1)
    mu, std = norm.fit(z[ind])
    p = norm.pdf(z_grid, mu, std)
    plt.plot(z_grid, p, color='b', linewidth=2)

    plt.savefig(data_path+'info/distribution.png')
    plt.close()
    
def get_var_df(df,var):
    var_cols = [c for c in df.columns if c.startswith(var)]
    return df[var_cols].to_numpy()