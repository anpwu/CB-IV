# CB-IV
## Introduction
This repository contains the implementation code for paper:

**Instrumental Variable Regression with Confounder Balancing** 

Anpeng Wu, Kun Kuang, Bo Li, and Fei Wu

## Env:

```shell
conda create -n tf-torch python=3.6
source activate tf-torch
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
conda install tensorflow-estimator==1.15.1 tensorflow-gpu==1.15.0
```

## The code for CB-IV:

1. dataProcess.ipynb
2. synCBIV.ipynb
3. run.ipynb
4. twinsCBIV.ipynb
5. ihdpCFR.ipynb

## The code for CB-IV-L (to appear):

1. runGenerator.py
2. runVAE_Demand.py
3. run_VAE_CBIV_Demand.py
