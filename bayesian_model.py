import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

#------------------------------------- MODEL FUNCTIONS ----------------------------------

def model_gauss_exp(x, a1, mu1, sigma1, a2, mu2, sigma2, b, k):
    gaussian1 = a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
    gaussian2 = a2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
    background = b * np.exp(-k * x)
    return gaussian1 + gaussian2 + background 

def model_single_gauss_exp(x, a1, mu1, sigma1, b, k):
    gaussian = a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
    background = b * np.exp(-k * x)
    return gaussian + background

def background_poly3(x, c0, c1, c2, c3):
    return c0 + c1 * x + c2 * x**2 + c3 * x**3

def gauss_peak(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def model_poly3(x, a, mu, sigma, c0, c1, c2, c3):
    return gauss_peak(x, a, mu, sigma) + background_poly3(x, c0, c1, c2, c3)

def background_exp(x, b, k):
    return b * np.exp(-k * x)

#------------------------------------- LIKELIHOOD FUNCTIONS ----------------------------------

def log_likelihood_probability(x, y, sigma_y, model, params): 
    prediction = model(x, *params)
    n = len(y)
    # log of the PDF of a multivariate Gaussian
    return (
        -0.5 * np.sum((y - prediction)**2/sigma_y**2)  # Exponent
        - n/2*np.log(2*np.pi*sigma_y**2)               # Normalisation
    )
    
def log_likelihood_heteroscedastic(x, y,sigma_y, model, params):
    prediction = model(x, *params)

    return (
        -0.5 * np.sum((y - prediction)**2/sigma_y**2)    # Exponent
        -0.5 * np.sum(np.log(2*np.pi*sigma_y**2))        # Normalisation
    )
    
def log_likelihood_poisson(x, y, model, params):
    # Model predictions (λ values)
    lambda_model = model(x, *params)
    
    # Log-likelihood computation
    # Safeguard against taking log of zero or negative λ
    lambda_model = np.maximum(lambda_model, 1e-10)
    log_likelihood_value = np.sum(y * np.log(lambda_model) - lambda_model - gammaln(y + 1))
    
    return log_likelihood_value

#------------------------------------- PRIOR FUNCTIONS ----------------------------------
    
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

#------------------------------------- POSTERIOR FUNCTIONS ----------------------------------

def log_posterior_probability(params, x, y, sigma_y, model, priors, mode='gaussian'):
    
    if mode == 'gaussian':
        return (
            log_likelihood_probability(x, y, sigma_y, model, params)
            + log_prior_probability(params, priors)
        )
        
    elif mode == 'poisson':
        return (
            log_likelihood_poisson(x, y, model, params)
            + log_prior_probability(params, priors)
        )
        
    elif mode == 'gaussian heteroscedastic':
        return (
            log_likelihood_heteroscedastic(x, y, sigma_y, model, params)
            + log_prior_probability(params, priors)
        )
    
def negative_log_posterior(params, x, y, sigma_y, model, priors, mode='gaussian'):
    return -log_posterior_probability(params, x, y, sigma_y, model, priors, mode)

#############################################################################################

def log_posterior_heteroscedastic(params, x, y, sigma_y, model, priors):
    return log_likelihood_heteroscedastic(x, y,sigma_y, model, params) + log_prior_probability(params, priors)

def negative_log_posterior_heteroscedastic(params, x, y, sigma_y, model, priors):
    return -log_posterior_heteroscedastic(params, x, y, sigma_y, model, priors)

#############################################################################################

def sample_posterior(x, y, y_errors, chain, model, seed=15, n_samples=200, plot=True, mode='gaussian'):
    
    np.random.seed(seed)
    flat_chain = chain.reshape(-1, chain.shape[-1])
    chain_samples = flat_chain[np.random.choice(chain.shape[0], size=n_samples)]

    # Evaluate the model at the sample parameters
    model_predictive = np.array(
        [model(x, *sample) for sample in chain_samples]
    )
    model_quantiles = np.quantile(
        model_predictive, q=[0.025, 0.16, 0.84, 0.975], axis=0
    )
        
    def predict_poisson(params, x):
        lambda_model = model(x, *params)
        lambda_model = np.maximum(lambda_model, 0)

        # Draw from the Poisson distribution using the computed lambda
        return np.random.poisson(lam=lambda_model)
    
    def predict_gaussian(params, x, y_errors):
        prediction = model(x, *params)
        return np.random.normal(prediction, y_errors)

    if mode == 'poisson':
        posterior_predictive = np.array(
            [predict_poisson(sample, x) for sample in chain_samples])
        
    elif mode == 'gaussian':
        posterior_predictive = np.array(
            [predict_gaussian(sample, x, y_errors) for sample in chain_samples])
        
    quantiles = np.percentile(posterior_predictive, [2.5, 16, 84, 97.5], axis=0)
    
    if plot == True:
        
        plt.grid()
        plt.errorbar(x, y, yerr=y_errors, fmt=".", color="b", label="Data", alpha=0.7)
        plt.fill_between(x, model_quantiles[0], model_quantiles[-1], alpha=0.5, facecolor="C1",
                    label="Model predictive distribution")
        plt.fill_between(x, model_quantiles[1], model_quantiles[-2], alpha=0.5, facecolor="C1")
        #plt.plot(x, y, ".", color = 'black')

        plt.fill_between(x, quantiles[0], quantiles[-1], alpha=0.5, facecolor="C17",
                            label="Posterior predictive distribution")
        plt.fill_between(x, quantiles[1], quantiles[-2], alpha=0.5, facecolor="C17")
        plt.xlabel('Invariant mass [GeV]')
        plt.ylabel('Number of events')
        plt.legend()
        
    return chain_samples, model_predictive, posterior_predictive


def process_chain(chain, discard=0, thin=1, flat=False):
    """
    Process an MCMC chain array by discarding initial steps, thinning, and optionally flattening.

    Parameters:
        chain (np.ndarray): The MCMC chain array with shape (nsteps, nwalkers, ndim).
        discard (int): Number of initial steps to discard from each walker.
        thin (int): Interval to keep samples (every 'thin' steps).
        flat (bool): Whether to flatten the chain across all walkers.

    Returns:
        np.ndarray: The processed chain array.
    """
    # Discard the initial steps
    if discard > 0:
        chain = chain[discard:]
    
    # Apply thinning
    if thin > 1:
        chain = chain[::thin]

    # Check if flattening is needed
    if flat:
        # Reshape the chain to combine steps and walkers
        nsteps, nwalkers, ndim = chain.shape
        chain = chain.reshape(nsteps * nwalkers, ndim)

    return chain
                     
