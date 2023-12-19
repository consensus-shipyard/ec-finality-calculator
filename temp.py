import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf
from scipy.stats import poisson
from scipy.stats import norm
import validator_calc_finality as vf



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

def plot_normal_ccdf(mean, variance):
    # Standard deviation from variance
    std_dev = np.sqrt(variance)

    # Generate a range of values for which we will plot the CCDF
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)

    # Calculate the CCDF (1 - CDF)
    ccdf = 1 - norm.cdf(x, loc=mean, scale=std_dev)

    # Find where the CCDF reaches 10^-10
    critical_value = norm.ppf(1 - 10 ** -10, loc=mean, scale=std_dev)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, ccdf, label="CCDF")
    plt.axhline(y=10 ** -10, color='r', linestyle='--', label="CCDF = $10^{-10}$")
    plt.axvline(x=critical_value, color='g', linestyle='--', label=f"Critical value at {critical_value:.2f}")
    plt.yscale('log')  # Log scale for better visibility
    plt.title("Complementary Cumulative Distribution Function (CCDF) of a Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("CCDF (log scale)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_conditional_poisson_distribution(lambda_val, c):
    # Calculate the conditional probabilities for k >= c
    conditional_probs = {}
    k_values = range(int(c-1), int(max(lambda_val,c) * 2) + 10)
    expected_value = 0
    sanity = 0.0
    for k in k_values:
        prob = hf.conditional_poisson_probability(lambda_val, k, c)
        expected_value += k * prob
        conditional_probs[k] = prob
        sanity += prob

    print(f"The expected value of the conditional distribution: {expected_value}")
    print(f"sanity check: {sanity}")

    # Plotting
    plt.bar(conditional_probs.keys(), conditional_probs.values(), color='blue')
    plt.xlabel('k')
    plt.ylabel('P(X = k | X ≥ '+str(c)+')')
    plt.title(f'Conditional Poisson Distribution (λ = {lambda_val})')
    plt.xticks(list(k_values))
    plt.show()

# # Example usage plot_poisson_ccdf
# lambda_val = 30*2.5  # Replace with your desired lambda value
# plot_poisson_ccdf(lambda_val, 30 * 4.5)

# # Example usage plot_normal_ccdf
# mean=2.5*30
# variance=3*30
# print("expected: "+str(4.5*30))
# plot_normal_ccdf(mean, variance)  # mean (x) = 0, variance (y) = 1

# Example usage plot_conditional_poisson_distribution
# lambda_val = 5*30
# chain_health = 1.2
# c = chain_health * lambda_val
# plot_conditional_poisson_distribution(lambda_val, c)

import pandas as pd

# Load the CSV file
nov = r'C:\Users\sgore\Downloads\orphan_block_count_november.csv'
mar = r'C:\Users\sgore\Downloads\blocks_count_from_march.csv'
df = pd.read_csv(mar)  # Replace 'your_file.csv' with your actual file name

# Adjust 'height' to start from zero
average = df['block_counts'].mean()
print("length is: " + str(len(df['height'])))
print("average block count is: " + str(average))


# Parameters
subseq_length = 905

# Calculate the rolling mean with a window size of 900
rolling_means = df['block_counts'].rolling(window=subseq_length).mean()

# Find the start position of the subsequence with the lowest average
min_mean_index = rolling_means.idxmin()

# Extract the subsequence
lowest_avg_subsequence = df['block_counts'][min_mean_index:min_mean_index + subseq_length]

# Print or process the subsequence
min_average = lowest_avg_subsequence.mean()

print("min_average block count is: " + str(min_average) + " at position " + str(min_mean_index))



chain = lowest_avg_subsequence.to_numpy()
e = 5
f = 0.3
c = len(chain)-1 # current position (end of history)
s = c - 30


ans1 = vf.validator_calc_finality(e, f, chain, c, s)
print("the error probability1 is: " + str(ans1))

ans2 = vf.validator_calc_finality(e, f, chain, c, s-5)
print("the error probability2 is: " + str(ans2))


