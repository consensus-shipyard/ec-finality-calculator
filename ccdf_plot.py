import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


def plot_poisson_ccdf(lambda_val, x_max=None):
    """
    Plots the CCDF of a Poisson random variable with a given lambda value.

    Parameters:
    lambda_val (float): The lambda (mean) of the Poisson distribution.
    x_max (int, optional): The maximum x value for the plot. Defaults to None.
    """
    if x_max is None:
        x_max = int(lambda_val + 10 * np.sqrt(lambda_val))  # A default heuristic

    # Generate x values and calculate the CDF
    x_values = np.arange(0, x_max + 1)
    cdf_values = poisson.cdf(x_values, lambda_val)

    # Calculate the CCDF as 1 - CDF
    ccdf_values = 1 - cdf_values

    # Plot the CCDF
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, ccdf_values, marker='o', linestyle='-')
    plt.title(f'CCDF of a Poisson({lambda_val}) Random Variable')
    plt.xlabel('x')
    plt.ylabel('CCDF (log scale)')
    plt.yscale('log')
    plt.grid(True)
    plt.show()


# Example usage
lambda_val = 50*5  # Replace with your desired lambda value
plot_poisson_ccdf(lambda_val, lambda_val + 6.3 * np.sqrt(lambda_val))