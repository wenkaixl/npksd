#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 23:53:58 2022

"""

# from abc import ABCMeta, abstractmethod
from abc import abstractmethod
import torch
import torch.optim as optim
from objective import *
# import numpy as np


## estimator class

class Estimator(object):
    """
    base class for constructing an estimator
    """
    @abstractmethod
    def fit(self, X):
        """
        Fit the estimator with samples
        """
        raise NotImplementedError()

class ESScoreSM(Estimator):
    """
    Learning model density via Energy-function with Score Matching
    """
    def __init__(self, H=128, seed=1112, lr=0.01, n_epoch = 100):
        self.H = H
        self.seed = seed
        self.lr = lr
        self.n_epoch = n_epoch
        
    def fit(self, X):
        d = X.shape[1]         
        H = self.H

        torch.random.manual_seed(self.seed)
        logp_model = torch.nn.Sequential(
                    torch.nn.Linear(d, H, bias=True),
                    torch.nn.Softplus(),
                    torch.nn.Linear(H, H, bias=True),
                    torch.nn.Softplus(),
                    torch.nn.Linear(H, H, bias=True),
                    torch.nn.Softplus(),
                    torch.nn.Linear(H, 1, bias=True))

        X = torch.Tensor(X)
        optimizer = optim.Adam(logp_model.parameters(), lr=self.lr)
        n_epoch = self.n_epoch
        for i in range(n_epoch):
            sm_obj = exact_score_matching(logp_model, X, train=True).mean()
            optimizer.zero_grad()
            sm_obj.backward(retain_graph=True)
            optimizer.step()
            if i % 50 == 0:
                print(sm_obj.detach())
        return logp_model 


class NN_log_den_gauss(torch.nn.Module):
    """
    function to evaluate the (unnormalised) log denstiy of Gaussian
    """
    def __init__(self, d):
        super(NN_log_den_gauss, self).__init__()
        self.d = d
        self.mean = torch.nn.Parameter(torch.ones(1,d, requires_grad=True))
        self.prec = torch.nn.Parameter(torch.ones(d,d, requires_grad=True))
    
    def forward(self, X):
        d = self.d
        unden = - torch.diag((X - self.mean)@self.prec@(X - self.mean).T)
        return unden 

class NN_log_den_isotropic_gauss(torch.nn.Module):
    """
    function to evaluate the (unnormalised) log denstiy of Gaussian
    """
    def __init__(self, d):
        super(NN_log_den_isotropic_gauss, self).__init__()
        self.d = d
        self.mean = torch.nn.Parameter(torch.ones(1,d, requires_grad=True))
        self.scale = torch.nn.Parameter(torch.ones(1, requires_grad=True))
    
    def forward(self, X):
        unden = - torch.diag((X - self.mean)@(X - self.mean).T)/self.scale
        return unden 


class ESGaussSM(Estimator):
    """
    Learning model density via Gaussian form with Score Matching
    """
    def __init__(self, seed=1112, log_den = NN_log_den_isotropic_gauss, lr=0.01, n_epoch = 100):
        self.seed = seed
        self.log_den = log_den
        self.lr = lr
        self.n_epoch = n_epoch
        
    def fit(self, X):
        d = X.shape[1]         

        torch.random.manual_seed(self.seed)
        logp_model = self.log_den(d)
        
        X = torch.Tensor(X)
        optimizer = optim.Adam(logp_model.parameters(), lr=self.lr)
        n_epoch = self.n_epoch
        for i in range(n_epoch):
            sm_obj = exact_score_matching(logp_model, X, train=True).mean()
            optimizer.zero_grad()
            sm_obj.backward(retain_graph=True)
            optimizer.step()
            if i % 50 == 0:
                print(sm_obj.detach())
        return logp_model 


class NN_log_den_t(torch.nn.Module):
    """
    function to evaluate the (unnormalised) log denstiy of student-t of degree of freedom(df)
    """
    def __init__(self, d, uncentered=False):
        super(NN_log_den_t, self).__init__()
        self.d = d
        
        self.mean = torch.nn.Parameter(torch.zeros(d, requires_grad=uncentered))
        # self.prec = torch.nn.Parameter(torch.ones([d,d], requires_grad=uncentered))
        self.df = torch.nn.Parameter(torch.ones(1, requires_grad=True))
        
    def forward(self, X):
        logden = torch.log(1. + ((X - self.mean)@(X - self.mean).T)/self.df)
        unden = 0.5*(self.df + self.d) * torch.diag(logden)
        return -unden 


class ESTDistSM(Estimator):
    """
    Learning model density via T-distribution form with Score Matching
    """
    def __init__(self, seed=1112, log_den = NN_log_den_t, uncentered=False, lr=0.01, n_epoch = 100):
        self.seed = seed
        self.log_den = log_den
        self.lr = lr
        self.n_epoch = n_epoch
        self.uncentered = uncentered
        
    def fit(self, X):
        d = X.shape[1]         

        torch.random.manual_seed(self.seed)
        logp_model = self.log_den(d, uncentered=self.uncentered)

        X = torch.Tensor(X)
        optimizer = optim.Adam(logp_model.parameters(), lr=self.lr)
        n_epoch = self.n_epoch
        for i in range(n_epoch):
            sm_obj = exact_score_matching(logp_model, X, train=True).mean()
            optimizer.zero_grad()
            sm_obj.backward(retain_graph=True)
            optimizer.step()
            if i % 50 == 0:
                print(sm_obj.detach())
        return logp_model 



class NN_cond_log_den_mean(torch.nn.Module):
    """
    function to evaluate the (unnormalised) log denstiy condition on summary statistics
    """
    def __init__(self, d, H):
        super(NN_cond_log_den_mean, self).__init__()
        self.d = d
        self.H = H
        self.logp_model = torch.nn.Sequential(
                torch.nn.Linear(1 + (d>1), H, bias=True),
                torch.nn.Softplus(),
                torch.nn.Linear(H, H, bias=True),
                torch.nn.Softplus(),
                # torch.nn.Linear(H, H, bias=True),
                # torch.nn.Softplus(),
                torch.nn.Linear(H, 1, bias=True))

        
    def forward(self, X):
        d = X.shape[1]   
        
        if not torch.is_tensor(X):
            X = torch.Tensor(X)  

        if d>1:
            Xd = X.T.reshape([-1,1])
            Xs = X.sum(axis=1)
            Xsd = Xs.repeat(d)
            Xsd = Xsd.unsqueeze(1)
            Xt = (Xsd - Xd)/(d-1.)
            assert Xt.shape[0]==Xd.shape[0]
            Xnew = torch.cat([Xd, Xt], dim=1)
            logden = self.logp_model(Xnew)
        else:
            logden = self.logp_model(X) 
        return logden

        


class ESMeanSM(Estimator):
    """
    Learning model density via mean summary statistics with Score Matching objective
    """
    def __init__(self, H=128, seed=1112, log_den = NN_cond_log_den_mean,
                 lr=0.01, n_epoch = 100):
        self.H = H
        self.seed = seed
        self.log_den = log_den
        self.lr = lr
        self.n_epoch = n_epoch
        
    def fit(self, X):
        d = X.shape[1]         
        H = self.H
        
        
        torch.random.manual_seed(self.seed)
        logp_model = self.log_den(d, H)
        
        if not torch.is_tensor(X):
            X = torch.Tensor(X)  

        optimizer = optim.Adam(logp_model.parameters(), lr=self.lr)
        n_epoch = self.n_epoch
        
        for i in range(n_epoch):
            sm_obj = exact_score_matching(logp_model, X, train=True).mean()
            optimizer.zero_grad()
            sm_obj.backward(retain_graph=True)
            optimizer.step()
            if i % 50 == 0:
                print(sm_obj.detach())
        return logp_model 
