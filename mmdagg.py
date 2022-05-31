#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:49:38 2022

Implemtation for Aggregated Maximum-mean-discrepancy (MMD) test
Schrab et al. MMDAgg: an MMD aggregated two-sample test 2021
code adapted and modified from 
https://github.com/antoninschrab/mmdagg-paper.git
"""

import numpy as np

##weights for each bandwidth
def create_weights(N, weights_type="uniform"):
    """
    Create weights as defined in Section 5.1 of our paper.
    inputs: N: number of bandwidths to test
            weights_type: "uniform" or "decreasing" or "increasing" or "centred"
    output: (N,) array of weights
    """
    if weights_type == "uniform":
        weights = np.array([1 / N,] * N)
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centred":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = np.array(
                [1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)]
            )
        else:
            normaliser = sum(
                [1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)]
            )
            weights = np.array(
                [
                    1 / ((abs((N + 1) / 2 - i) + 0.5) * normaliser)
                    for i in range(1, N + 1)
                ]
            )
    else:
        raise ValueError(
            'The value of weights_type should be "uniform" or'
            '"decreasing" or "increasing" or "centred".'
        )
    return weights


def compute_median_bandwidth_subset(seed, X, Y, max_samples=2000, min_value = 0.0001):
    """
    Compute the median distance in each dimension between all the points in X and Y
    using at most max_samples samples and using a threshold value min_value.
    inputs: seed: random seed
            X: (m,d) array of samples
            Y: (n,d) array of samples
            max_samples: number of samples used to compute the median (int or None)
    output: (d,) array: median of absolute difference in each component
    """
    if max_samples != None:
        rs = np.random.RandomState(seed)
        pX = rs.choice(X.shape[0], min(max_samples // 2, X.shape[0]), replace=False)
        pY = rs.choice(Y.shape[0], min(max_samples // 2, Y.shape[0]), replace=False)
        Z = np.concatenate((X[pX], Y[pY]))
    else:
        Z = np.concatenate((X, Y))
    median_bandwidth = compute_median_bandwidth(Z)
    return np.maximum(median_bandwidth, min_value)


def compute_median_bandwidth(Z):
    """
    Compute the median distance in each dimension between all the points in Z.
    input: Z: (m+n,d) array of pooled samples  
    output: (d,) array: median of absolute different in each component
    """
    mn, d = Z.shape
    diff = np.zeros((d, int((mn ** 2 - mn) / 2)))
    output = np.zeros(d)
    for u in range(d):
        k = 0
        for i in range(mn - 1):
            for j in range(i + 1, mn):
                diff[u, k] = np.abs(Z[i, u] - Z[j, u])
                k += 1
        output[u] = np.median(diff[u])
    return output

###kernel related functions

def mutate_K(K, approx_type="permutation"):
    """
    Mutate the kernel matrix K depending on the type of approximation.
    inputs: K: kernel matrix of size (m+n,m+n) consisting of 
               four matrices of sizes (m,m), (m,n), (n,m) and (n,n)
               m and n are the numbers of samples from p and q respectively
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            
    output: if approx_type is "permutation" then the estimate is MMD_a (Eq. (3)) and 
               the matrix K is mutated to have zero diagonal entries
            if approx_type is "wild bootstrap" then the estimate is MMD_b (Eq. (6)),
               we have m = n and the matrix K is mutated so that the four matrices 
               have zero diagonal entries
    """
    if approx_type == "permutation":
        for i in range(K.shape[0]):
            K[i, i] = 0      
    if approx_type == "wild bootstrap":
        m = int(K.shape[0] / 2)  # m = n
        for i in range(m):
            K[i, i] = 0
            K[m + i, m + i] = 0
            K[i, m + i] = 0 
            K[m + i, i] = 0
            
def pairwise_square_l2_distance(Z):
    """
    Compute the pairwise L^2-distance matrix between all points in Z.
    inputs: Z is (mn,d) array
    output: (mn,mn) array of pairwise squared distances (L^2)
    https://stackoverflow.com/questions/53376686/what-is-the-most-efficient-way-to-compute-the-square-euclidean-distance-between/53380192#53380192
    faster than scipy.spatial.distance.cdist(Z,Z,'sqeuclidean')
    """
    mn, d = Z.shape
    dist = np.dot(Z, Z.T)  
    TMP = np.empty(mn, dtype=Z.dtype)
    for i in range(mn):
        sum_Zi = 0.0
        for j in range(d):
            sum_Zi += Z[i, j] ** 2
        TMP[i] = sum_Zi
    for i in range(mn):
        for j in range(mn):
            dist[i, j] = -2.0 * dist[i, j] + TMP[i] + TMP[j]
    return dist


def pairwise_l1_distance(Z):
    """
    Compute the pairwise L^1-distance matrix between all points in Z.
    inputs: Z is (mn,d) array
    output: (mn,mn) array of pairwise squared distances (L^1)
    """
    mn, d = Z.shape
    output = np.zeros((mn, mn))
    for i in range(mn):
        for j in range(mn):
            temp = 0.0
            for u in range(d):
                temp += np.abs(Z[i, u] - Z[j, u])
            output[i, j] = temp
    return output
           

def kernel_matrices(X, Y, bandwidth, bandwidth_multipliers, kernel_type="gaussian"):
    """
    Compute kernel matrices for several bandwidths.
    inputs: kernel_type: "gaussian" or "laplace"
            X is (m,d) array (m d-dimensional points)
            Y is (n,d) array (n d-dimensional points)
            bandwidth is (d,) array
            bandwidth_multipliers is (N,) array such that: 
                collection_bandwidths = [c*bandwidth for c in bandwidth_multipliers]
            kernel_type: "gaussian" or "laplace" (as defined in Section 5.3 of our paper)
    outputs: list of N kernel matrices for the pooled sample with the N bandwidths
    """
    m, d = X.shape
    # Z = np.concatenate((X / bandwidth, Y / bandwidth))
    Z = np.concatenate((X, Y))
    Z /= bandwidth
    if kernel_type == "gaussian":
        pairwise_sq_l2_dists = pairwise_square_l2_distance(Z) 
        prod = np.prod(bandwidth)
        output_list = []
        for c in bandwidth_multipliers:
            output_list.append(np.exp(-pairwise_sq_l2_dists / (c ** 2))) 
        return output_list
    elif kernel_type == "laplace":
        pairwise_l1_dists = pairwise_l1_distance(Z) 
        prod = np.prod(bandwidth)
        output_list = []
        for c in bandwidth_multipliers:
            output_list.append(np.exp(-pairwise_l1_dists / c)) 
        return output_list
    else:
        raise ValueError(
            'The value of kernel_type should be either "gaussian" or "laplace"'
        )
        
        
## MMDAgg implementation

def mmdagg(
    X, Y, alpha=0.05, seed=112,  kernel_type="gaussian", approx_type="permutation", weights_type="uniform", 
    l_minus=-2, l_plus=2, B1=500, B2=500, B3=100):
    """
    Compute MMDAgg as defined in Algorithm 1 in our paper using the collection of
    bandwidths defined in Eq. (16) and the weighting strategies proposed in Section 5.1.
    inputs: seed: integer random seed
            X: (m,d) array (m d-dimensional points)
            Y: (n,d) array (n d-dimensional points)
            alpha: real number in (0,1) (level of the test)
            kernel_type: "gaussian" or "laplace"
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            weights_type: "uniform", "decreasing", "increasing" or "centred" (Section 5.1 of our paper)
            l_minus: integer (for collection of bandwidths Eq. (16) in our paper)
            l_plus: integer (for collection of bandwidths Eq. (16) in our paper)
            B1: number of simulated test statistics to estimate the quantiles
            B2: number of simulated test statistics to estimate the probability in Eq. (13) in our paper
            B3: number of iterations for the bisection method
    output: result of MMDAgg (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    m = X.shape[0]
    n = Y.shape[0]
    mn = m + n
    assert n >= 2 and m >= 2
    assert X.shape[1] == Y.shape[1]
    assert 0 < alpha  and alpha < 1
    assert kernel_type in ["gaussian", "laplace"]
    assert approx_type in ["permutation", "wild bootstrap"]
    assert weights_type in ["uniform", "decreasing", "increasing", "centred"]
    assert l_plus >= l_minus

    # compute median bandwidth
    median_bandwidth = compute_median_bandwidth_subset(seed, X, Y)
    
    # define bandwidth_multipliers and weights
    bandwidth_multipliers = np.array([2 ** i for i in range(l_minus, l_plus + 1)])
    N = bandwidth_multipliers.shape[0]  # N = 1 + l_plus - l_minus
    weights = create_weights(N, weights_type)
    
    # compute the kernel matrices
    kernel_matrices_list = kernel_matrices(
        X, Y, median_bandwidth, bandwidth_multipliers, kernel_type) 

    return mmdagg_custom(
        seed, 
        kernel_matrices_list, 
        weights, 
        m, 
        alpha, 
        approx_type, 
        B1, 
        B2, 
        B3,
    )


def mmdagg_custom(
    seed, kernel_matrices_list, weights, m, alpha, approx_type, B1, B2, B3
):
    """
    Compute MMDAgg as defined in Algorithm 1 in our paper with custom kernel matrices
    and weights.
    inputs: seed: integer random seed
            kernel_matrices_list: list of N kernel matrices
                these can correspond to kernel matrices obtained by considering
                different bandwidths of a fixed kernel as we consider in our paper
                but one can also use N fundamentally different kernels.
                It is assumed that the kernel matrices are of shape (m+n,m+n) with
                the top left (m,m) submatrix corresponding to samples from X and 
                the bottom right (n,n) submatrix corresponding to samples from Y
            weights: array of shape (N,) consisting of positive entries summing to 1
            m: the number of samples from X used to create the kernel matrices
            alpha: real number in (0,1) (level of the test)
            kernel_type: "gaussian" or "laplace"
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            B1: number of simulated test statistics to estimate the quantiles
            B2: number of simulated test statistics to estimate the probability in Eq. (13) in our paper
            B3: number of iterations for the bisection method
    output: result of MMDAgg (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    n = kernel_matrices_list[0].shape[0] - m
    mn = m + n
    N = len(kernel_matrices_list)
    assert len(kernel_matrices_list) == weights.shape[0]
    assert n >= 2 and m >= 2
    assert 0 < alpha  and alpha < 1
    assert approx_type in ["permutation", "wild bootstrap"]
    
    # Step 1: compute all simulated MMD estimates (efficient as in Appendix C in our paper)
    M  = np.zeros((N, B1 + B2 + 1))  
    rs = np.random.RandomState(seed)
    if approx_type == "permutation":
        idx = rs.rand(B1 + B2 + 1, m + n).argsort(axis=1)  # (B1+B2+1, m+n): rows of permuted indices
        #11
        v11 = np.concatenate((np.ones(m), -np.ones(n)))  # (m+n, )
        V11i = np.tile(v11, (B1 + B2 + 1, 1))  # (B1+B2+1, m+n)
        V11 = np.take_along_axis(V11i, idx, axis=1)  # (B1+B2+1, m+n): permute the entries of the rows
        V11[B1] = v11  # (B1+1)th entry is the original MMD (no permutation)
        V11 = V11.transpose()  # (m+n, B1+B2+1)
        #10
        v10 = np.concatenate((np.ones(m), np.zeros(n)))
        V10i = np.tile(v10, (B1 + B2 + 1, 1))
        V10 = np.take_along_axis(V10i, idx, axis=1)
        V10[B1] = v10
        V10 = V10.transpose() 
        #01
        v01 = np.concatenate((np.zeros(m), -np.ones(n)))
        V01i = np.tile(v01, (B1 + B2 + 1, 1))
        V01 = np.take_along_axis(V01i, idx, axis=1)
        V01[B1] = v01
        V01 = V01.transpose() 
        for i in range(N):
            K = kernel_matrices_list[i]
            mutate_K(K, approx_type)
            M[i] = (
                np.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
                + np.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
                + np.sum(V11 * (K @ V11), 0) / (m * n)
            )  # (B1+B2+1, ) permuted MMD estimates
    elif approx_type == "wild bootstrap":
        R = rs.choice([-1.0, 1.0], size=(B1 + B2 + 1, n))
        R[B1] = np.ones(n)
        R = R.transpose()
        R = np.concatenate((R, -R))  # (2n, B1+B2+1) 
        for i in range(N):
            K = kernel_matrices_list[i]
            mutate_K(K, approx_type)
            M[i] = np.sum(R * (K @ R) , 0) /(n * (n - 1))
    else:
        raise ValueError(
            'The value of approx_type should be either "permutation" or "wild bootstrap".'
        )
    MMD_original = M[:, B1]
    M1_sorted = np.sort(M[:, :B1 + 1])  # (N, B1+1)
    M2 = M[:, B1 + 1:]  # (N, B2)
    
    # Step 2: compute u_alpha_hat using the bisection method
    quantiles = np.zeros((N, 1))  # (1-u*w_lambda)-quantiles for the N bandwidths
    u_min = 0
    u_max = np.min(1 / weights)
    for _ in range(B3): 
        u = (u_max + u_min) / 2
        for i in range(N):
            quantiles[i] = M1_sorted[
                i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
            ]
        P_u = np.sum(np.max(M2 - quantiles, 0) > 0) / B2
        if P_u <= alpha:
            u_min = u
        else:
            u_max = u
    u = u_min
        
    # Step 3: output test result
    for i in range(N):
        if ( MMD_original[i] 
            > M1_sorted[i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1]
        ):
            return 1
    return 0 
