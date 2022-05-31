#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 23:57:36 2022

"""


import numpy as np
import model, data, kernel, utils, estimator
from objective import *
import tests, tst

import mmdagg as agg

import torch 
import time 

import matplotlib.pyplot as plt
import matplotlib
# font options
font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 16
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--d", type=int, default=3)
# parser.add_argument("--test", type=str, default="Approx")
parser.add_argument("--n_gen", type=int, default=200)
args = parser.parse_args()


## test power for KSD and MMD
d =args.d
print(d)
n_gen = args.n_gen #sample size for testing


# Test level
alpha = 0.05


## Isotropic Gaussian
# mean = np.zeros(d)
# variance = 1.0 #/float(d)

## true model
# p = model.IsotropicNormal(mean, variance)
# ds = p.get_datasource()


## Mixture of Gaussian Model
sc = 1. #mean separation scale
# means = torch.stack([torch.ones(d)*sc, -sc*torch.ones(d)], dim=0)
means = torch.stack([torch.zeros(d), torch.zeros(d)], dim=0)
means[0][0] = sc
means[1][0] = -sc
# means
covs = torch.stack([torch.eye(d),torch.eye(d)], dim=0)
prop=0.5
pmix0 = torch.Tensor([prop, 1-prop])

#true model
p = model.MixtureGaussian(means, covs, pmix0)
ds = p.get_datasource()

split = 50 ##show test power after the number of tests


m_val = [100, 200, 500, 800] #sample size from generated model



## MMDAgg 
n = n_gen
n_perm = 200
l_minus=-2; l_plus=2


# number of tests to run
l = 100


# per = np.arange(-0.9,1.8,.3) #perturb GVD
# per = np.arange(-0.8,1.,.4) #perturb normal covariance term
per = np.arange(0,1.,.2) #perturb MoG

asst_power_collect = np.zeros([len(m_val), len(per)])
asst2_power_collect = np.zeros([len(m_val), len(per)])
asst_g_power_collect = np.zeros([len(m_val), len(per)])
asst_c_power_collect = np.zeros([len(m_val), len(per)])
# asst_wb_power_collect = np.zeros([len(m_val), len(per)])
mmdagg_power_collect = np.zeros([len(m_val), len(per)])
ksd_power_collect = np.zeros([len(m_val), len(per)])


for mi, m in enumerate(m_val):   
    start = time.time()     
    SM_est = estimator.ESScoreSM(lr=0.0005, n_epoch=300)
    p_est = model.ApproxLogDen(ds, SM_est,n=m)
    end = time.time()
    
    # SM_est400 = estimator.ESScoreSM(lr=0.001, n_epoch=400)
    # p_est2 = model.ApproxLogDen(ds, SM_est400,n=m)
    
    G_est = estimator.ESGaussSM(lr=0.01, n_epoch=200)
    p_est_g = model.ApproxLogDen(ds, G_est,n=m)
    
    M_est = estimator.ESMeanSM(lr=0.0005, n_epoch=300)
    p_est_m = model.ApproxLogDen(ds, M_est,n=m)
    for j in range(len(per)):
    
    	## for Isotropic Gaussian
        # draw_means = means
        # draw_variances = variances + per[j]
        # # draw_variances = variances
        # draw_variances[0] += per[j]
        # cov = np.eye(d) + 2. * np.diag(np.ones(d-1),1) * (0.0+per[j]) 
        # draw_cov = (cov + cov.T)/2.
        # p1 = model.Normal(draw_mean, draw_cov)
        # p1 = model.IsoGaussianMixture(draw_means, draw_variances)

        draw_means = means
        draw_covs = covs / (1. + per[j])
        p1 = model.MixtureGaussian(draw_means, draw_covs, pmix0)
        ds1 = p1.get_datasource()
        
        mmdagg_power = 0
        # asst_wb_power = 0 
        asst_power = 0 
        asst2_power = 0
        asst_g_power = 0
        asst_c_power = 0
        ksd_power = 0
        
        for i in range(l):
            dat = ds1.sample(n_gen, seed=11311*j+m+i)
            X = dat.X
            Y = ds.sample(m, seed=11121+m+i).X
            start =time.time()
            mmdagg_power += agg.mmdagg(X, Y, l_minus=l_minus, l_plus=l_plus, B1=200, B2=200, B3=100)
            end=time.time()
            
            sig2 = utils.meddistance(X, subsample=1000)**2
            k = kernel.KGauss(sig2)
            
            
            kstein = tests.KernelSteinTest(p, k, alpha=alpha, seed=1231+i)
            kstein_res = kstein.perform_test(dat)
            ksd_power += kstein_res['h0_rejected']
            
                    
            # kstein_est = tests.KernelSteinTest(p_est, k, alpha=alpha, seed=1+i)
            # ksteine_res = kstein_est.perform_test(dat)
            # asst_wb_power += ksteine_res['h0_rejected']
            
            kstein_est = tests.SteinMCTest(p_est, k, alpha=alpha, seed=1+i)
            ksteine_res = kstein_est.perform_test(dat)
            asst_power += ksteine_res['h0_rejected']

            # if i % 20==0:
            #     # print(X.shape, Y.shape)
            #     print(asst_power/float(i+1), mmdagg_power/float(i+1), ksd_power/float(i+1))
            
            
            # kstein_est = tests.SteinMCTest(p_est2, k, alpha=alpha, seed=1+i)
            # ksteine_res = kstein_est.perform_test(dat)
            # asst2_power += ksteine_res['h0_rejected']

            kstein_est = tests.SteinMCTest(p_est_g, k, alpha=alpha, seed=1+i)
            ksteine_res = kstein_est.perform_test(dat)
            asst_g_power += ksteine_res['h0_rejected']
            
            kstein_est = tests.SteinMCTest(p_est_m, k, alpha=alpha, seed=1+i)
            kstein_e_res = kstein_est.perform_test(dat)
            asst_c_power += kstein_e_res['h0_rejected']
            
            if i % split==0:
                # print(X.shape, Y.shape)
                print(m, i, asst_power, asst_g_power, asst_c_power, mmdagg_power, ksd_power)
                print("time:", (ksteine_res['time_secs']),int(kstein_e_res['time_secs']), end-start, kstein_res['time_secs'])
        mmdagg_power_collect[mi,j] = mmdagg_power/float(l)
        ksd_power_collect[mi,j] = ksd_power/float(l)
        # asst_wb_power_collect[mi,j] = asst_wb_power/float(l)
        asst_power_collect[mi,j] = asst_power/float(l)
        # asst2_power_collect[mi,j] = asst2_power/float(l)
        asst_g_power_collect[mi,j] = asst_g_power/float(l)
        asst_c_power_collect[mi,j] = asst_c_power/float(l)

        # np.savez("res/isogaussian_compare_conditional_n"+str(n_gen)+"_d"+str(d)+".npz", asst = asst_power_collect, asst_c = asst_c_power_collect,
        #          asst_g = asst_g_power_collect, mmdagg = mmdagg_power_collect, ksd=ksd_power_collect)
        np.savez("res/MoG_small_lr_n"+str(n_gen)+"_m"+str(m)+"_d"+str(d)+".npz", asst = asst_power_collect[mi,:], asst_c = asst_c_power_collect[mi,:],
                 asst_g = asst_g_power_collect[mi,:], mmdagg = mmdagg_power_collect[mi,:], ksd=ksd_power_collect[mi,:], per=per)
        print("Test: m="+str(m)+" power (mmd, ksd, asst, asst_c)=", mmdagg_power_collect[mi,j], ksd_power_collect[mi,j], asst_power_collect[mi,j],asst_c_power_collect[mi,j], "per"+str(per[j]))         

np.savez("res/MoG_small_lr_n"+str(n_gen)+"_d"+str(d)+".npz", asst = asst_power_collect, asst_c = asst_c_power_collect,asst_g = asst_g_power_collect, mmdagg = mmdagg_power_collect, ksd=ksd_power_collect, per=per)

# np.savez("res/gaussian_compare_n"+str(n_gen)+"_d"+str(d)+".npz", asst = asst_power_collect, asst2 = asst2_power_collect, 
#          asst_wb = asst_wb_power_collect,asst_g = asst_g_power_collect, mmdagg = mmdagg_power_collect, ksd=ksd_power_collect, per = per, m_val=m_val)



