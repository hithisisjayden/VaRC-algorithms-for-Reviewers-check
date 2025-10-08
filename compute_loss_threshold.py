import pandas as pd
import numpy as np
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import time
import csv
import tqdm

np.random.seed(42)

# Portfolio size
d = 10

# Repetition for scenarios
n_repetitions = 10

# Simulation paths
simulation_runs = 10_000_000 # 10000000 一千万是极限了 再算restarting kernal了
# simulation_runs = 1000 # demo

# Bandwidth
bandwidth = 0.010 # 0.005

# Confidence level
alpha_values = [0.95, 0.96, 0.97, 0.98, 0.99]

# LGD Shape parameters
LGD_a, LGD_b = 2, 5

# rho
rho = np.sqrt(0.5)

# rho_S, correlation of PD and LGD
rho_S = 0.5
covariance_matrix = np.array([[1, rho_S],
                              [rho_S, 1]])

# Z = np.random.multivariate_normal([0,0], covariance_matrix, simulation_runs)
# Z_L = Z[:, 0]
# Z_D = Z[:, 1]

# Default probability function
def default_probability(d):
    return np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    
def loss_driver(common_factor, idiosyncratic_factor):
    coefficient = np.sqrt(0.5)
    return coefficient * common_factor[:, np.newaxis] + np.sqrt(1 - coefficient ** 2) * idiosyncratic_factor

def default_driver(common_factor, idiosyncratic_factor):
    coefficient = np.sqrt(0.5)
    return coefficient * common_factor[:, np.newaxis] + np.sqrt(1 - coefficient ** 2) * idiosyncratic_factor

def generate_samples_pmc(d, LGD_a, LGD_b, simulation_runs):
    
    Z = np.random.multivariate_normal([0,0], covariance_matrix, simulation_runs)
    Z_L = Z[:, 0]
    Z_D = Z[:, 1]
    
    eta_L = np.random.normal(size=(simulation_runs, d))
    eta_D = np.random.normal(size=(simulation_runs, d))
    Y = loss_driver(Z_L, eta_L)
    X = default_driver(Z_D, eta_D)
    epsilon = beta.ppf(norm.cdf(Y), LGD_a, LGD_b)
    p = default_probability(d)
    x_threshold = norm.ppf(1-p)
    D = (X > x_threshold).astype(int)
    L = np.sum(epsilon * D, axis = 1)
    return L

EL = np.mean([generate_samples_pmc(d, LGD_a, LGD_b, simulation_runs) for _ in range(n_repetitions)])
loss_threshold = 3 * EL
















