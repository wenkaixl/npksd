#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 23:28:33 2022

"""

import numpy as np
import torch
import torch.autograd as autograd


def exact_score_matching(energy_net, samples, train=False):
    """
    score matching objective based on learning energy function
    adapted and modified from implementation on https://github.com/ermongroup/ncsnv2.git
    """
    
    samples.requires_grad_(True)
    logp = -energy_net(samples).sum()
    grad1 = autograd.grad(logp, samples, create_graph=True)[0]
    loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.
    loss2 = torch.zeros(samples.shape[0], device=samples.device)

    # if samples.shape[1] > 100:
    #     iterator = tqdm(range(samples.shape[1]))
    # else:
    iterator = range(samples.shape[1])

    for i in iterator:
        if train:
            grad = autograd.grad(grad1[:, i].sum(), samples, create_graph=True, retain_graph=True)[0][:, i]
        if not train:
            grad = autograd.grad(grad1[:, i].sum(), samples, create_graph=False, retain_graph=True)[0][:, i]
            grad = grad.detach()
        loss2 += grad

    loss = loss1 + loss2

    if not train:
        loss = loss.detach()

    return loss


# def conditional_score_matching(cond_energy_net, samples, cond, train=False):
#     """
#     score matching objective based on learning energy function
#     """
#     cond.requires_grad_(False)
#     samples.requires_grad_(True)
#     logp = -cond_energy_net(samples, cond).sum()
#     grad1 = autograd.grad(logp, samples, create_graph=True)[0]
#     loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.
#     loss2 = torch.zeros(samples.shape[0], device=samples.device)

#     # if samples.shape[1] > 100:
#     #     iterator = tqdm(range(samples.shape[1]))
#     # else:
#     iterator = range(samples.shape[1])

#     for i in iterator:
#         if train:
#             grad = autograd.grad(grad1[:, i].sum(), samples, create_graph=True, retain_graph=True)[0][:, i]
#         if not train:
#             grad = autograd.grad(grad1[:, i].sum(), samples, create_graph=False, retain_graph=True)[0][:, i]
#             grad = grad.detach()
#         loss2 += grad

#     loss = loss1 + loss2

#     if not train:
#         loss = loss.detach()

#     return loss


def elbo_obj(energy_net, samples, train=False):
    
    samples.requires_grad_(True)
    logp = -energy_net(samples).sum()
    grad1 = autograd.grad(logp, samples, create_graph=True)[0]
    loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.
    loss2 = torch.zeros(samples.shape[0], device=samples.device)

    # if samples.shape[1] > 100:
    #     iterator = tqdm(range(samples.shape[1]))
    # else:
    iterator = range(samples.shape[1])

    for i in iterator:
        if train:
            grad = autograd.grad(grad1[:, i].sum(), samples, create_graph=True, retain_graph=True)[0][:, i]
        if not train:
            grad = autograd.grad(grad1[:, i].sum(), samples, create_graph=False, retain_graph=True)[0][:, i]
            grad = grad.detach()
        loss2 += grad

    loss = loss1 + loss2

    if not train:
        loss = loss.detach()

    return loss


