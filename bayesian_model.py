import numpy as np

def model_gauss_exp(x, a1, mu1, sigma1, a2, mu2, sigma2, b, k):
    gaussian1 = a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
    gaussian2 = a2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
    background = b * np.exp(-k * x)
    return gaussian1 + gaussian2 + background 

def log_likelihood_probability(x, y, sigma_y, model, params): 
    prediction = model(x, *params)
    n = len(y)
    # log of the PDF of a multivariate Gaussian
    return (
        -0.5 * np.sum((y - prediction)**2/sigma_y**2)  # Exponent
        - n/2*np.log(2*np.pi*sigma_y**2)               # Normalisation
    )
    
def log_prior_probability(params, priors):
    prob = 0
    if len(params) != len(priors):
        raise ValueError("The number of parameters and priors must be equal.")
    else:
        for i in range(len(params)):
            prob += priors[i].logpdf(params[i])
        return prob
    
def sample_prior(n_sample, priors):
    """Sample n_sample times from the prior distribution."""
    
    return np.array([prior.rvs(n_sample) for prior in priors]).T

def log_posterior_probability(params, x, y, sigma_y, model, priors):
    return (
        log_likelihood_probability(x, y, sigma_y, model, params)
        + log_prior_probability(params, priors)
    )
    
def negative_log_posterior(params, x, y, sigma_y, model, priors):
    return -log_posterior_probability(params, x, y, sigma_y, model, priors)
    
                     
