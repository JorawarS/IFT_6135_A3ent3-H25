"""
Solutions for Question 1 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""
import math
import numpy as np
import torch

torch.manual_seed(42)

def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    #TODO: compute log_likelihood_bernoulli
    ll_bernoulli = target * torch.log(mu) + (1 - target) * torch.log(1 - mu) # shape: (batch_size, input_size)

    return ll_bernoulli.sum(dim=1)


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    d = mu.size(1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)
    

    # Compute squared Euclidean distance between z and mu
    squared_dist_normal = ((z - mu) ** 2)/torch.exp(logvar) # shape: (batch_size, input_size)
    
    #TODO: compute log normal
    ll_normal = -0.5 * (d * math.log(2 * math.pi) + logvar.sum(dim=1) + squared_dist_normal.sum(dim=1)) # shape: (batch_size,)
    
    return ll_normal


def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    #TODO: compute log_mean_exp
    a =   y.max(dim=1)[0].unsqueeze(1) # shape: (batch_size, 1)
    exp_diff_sum = torch.exp(y - a).sum(dim=1)  # shape: (batch_size,)
    lme = a + torch.log(exp_diff_sum/sample_size) # shape: (batch_size,)

    return lme


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    #TODO: compute kld
    var_q = torch.exp(logvar_q) # shape: (batch_size, input_size)
    var_p = torch.exp(logvar_p) # shape: (batch_size, input_size)
    kl_gg = 0.5 * ( (var_q/var_p) +((mu_p - mu_q) ** 2)/var_p + logvar_p - logvar_q - 1 ) # shape: (batch_size, input_size)
    

    return kl_gg.sum(dim=1) # shape: (batch_size,)


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    #TODO: compute kld
    std_q = torch.exp(logvar_q / 2) # shape: (batch_size, num_samples, input_size)
    noise = torch.randn_like(std_q) # shape: (batch_size, num_samples, input_size)
    z = mu_q + std_q * noise # shape: (batch_size, num_samples, input_size)
    log_q = log_likelihood_normal(mu_q, logvar_q, z)/num_samples # shape: (batch_size,)
    log_p = log_likelihood_normal(mu_p, logvar_p, z)/num_samples # shape: (batch_size,)
    kl_mc = log_q - log_p # shape: (batch_size,)
    
    


    return kl_mc
