import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.stats import rel_breitwigner
from scipy.special import erf
from scipy.signal import convolve

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

# def crystal_ball(x, a, mu, sigma, alpha, n):
#     z = (x - mu) / sigma
#     if alpha < 0:
#         z = -z
#     A = (n / np.abs(alpha)) ** n * np.exp(-alpha ** 2 / 2)
#     B = n / np.abs(alpha) - np.abs(alpha)
#     C = n / np.abs(alpha) * (1 / (n - 1)) * np.exp(-alpha ** 2 / 2)
#     D = np.sqrt(np.pi / 2) * (1 + erf(np.abs(alpha) / np.sqrt(2)))
#     N = 1.0 / (sigma * (C + D))
#     if z > -alpha:
#         return a * N * np.exp(-z ** 2 / 2)
#     else:
#         return a * N * A * (B - z) ** (-n)
    
# def breit_wigner(x, a, mu, gamma):
#     return a / ((x - mu) ** 2 + (gamma / 2) ** 2)

# Crystal Ball function
def crystal_ball(x, alpha, n, mean_cb, sigma_cb):
    """Crystal Ball function with Gaussian core and power-law tail."""
    # if sigma_cb <= 0 or n <= 0:
    #     raise ValueError("Sigma and n must be positive.")

    A = (n / abs(alpha)) ** n * np.exp(-alpha**2 / 2)
    B = n / abs(alpha) - abs(alpha)
    C = (n / abs(alpha)) * (1 / (n - 1)) * np.exp(-alpha**2 / 2)
    D = np.sqrt(np.pi / 2) * (1 + erf(alpha / np.sqrt(2)))
    N = 1 / (sigma_cb * (C + D))
    
    z = (x - mean_cb) / sigma_cb
    return np.where(z > -alpha, N * np.exp(-z**2 / 2), N * A * (B - z)**-n)

# Convolution of Breit-Wigner and Crystal Ball
def bw_cb_convolution(x, alpha, n, mean_cb, sigma_cb):
    """Convolve Breit-Wigner with Crystal Ball."""
    dx = x[1] - x[0]  # Step size
    x_fine = np.linspace(x[0] - 5 * sigma_cb, x[-1] + 5 * sigma_cb, len(x) * 5)
    
    # Relativistic Breit-Wigner (fixed mZ and GammaZ)
    bw = rel_breitwigner.pdf(x_fine, 91.188, 2.485)
    
    # Crystal Ball function
    cb = crystal_ball(x_fine, alpha, n, mean_cb, sigma_cb)
    
    # Perform convolution using scipy's convolve
    convolved = convolve(bw, cb, mode='same') * dx
    
    # Interpolate back to the original x grid
    return np.interp(x, x_fine, convolved)

# Combined signal + background model
def signal_model(x, alpha, n, mean_cb, sigma_cb, scale, exp_scale, exp_coeff):
    """Breit-Wigner ⊗ Crystal Ball + Exponential Background."""
    # Signal: Convolution of Breit-Wigner and Crystal Ball
    signal = scale * bw_cb_convolution(x, alpha, n, mean_cb, sigma_cb)
    
    # Background: Exponential falling background
    background = exp_scale * np.exp(-exp_coeff * x)
    
    return signal + background
    

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
                     
