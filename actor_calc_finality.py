import numpy as np
import helper_functions as hf
from scipy.stats import poisson
from scipy.stats import skellam

# Define parameters
e = 5
f = 0.25
num_variables = 905 # length of history
chain_health = 4.8/5 # mean precentage of blocks in an epoch compared to the expectation from a perfect network
c = 904 # current position (end of history)
s = c - 90 # slot in history for which finality is calculated

# Generate the chain of Poisson random variables
chain = np.random.poisson(chain_health*e, num_variables)

## Calculate Lf
# Initialize an array to store the probabilities of Lf
max_kL = 300 # Define the values of k for which you want to calculate Pr(Lf=k)  # Adjust the range as needed
values_of_kL = np.arange(0, max_kL+1)
probabilities_Lf = np.zeros(max_kL + 1)


# Calculate BpZ given chain for each of the relevant past subchains
cumulative_loc_i = 0
max_look_back = 25 # for running time purpose. Needs to be justified theoretically!
for i in range(s, s - max_look_back, -1):
#for i in range(s, c - 900, -1):
    print("i="+str(i))
    cumulative_loc_i += chain[i]
    max_relevant_BpZ = (s - i + 1) * e * 4 + 2 * e # more than this, the probability is negligible
    _, probabilities_based_on_BpZ = hf.probability_of_BpZ_given_chain(chain, i - 1, s, e, f, max_relevant_BpZ//2, max_relevant_BpZ//2)
    # Calculate Pr(Lf=k) for each value of k
    for k in values_of_kL:
        prob_Lf_i = 0 if k + cumulative_loc_i >= len(probabilities_based_on_BpZ) else probabilities_based_on_BpZ[k + cumulative_loc_i]
        probabilities_Lf[k] = max(probabilities_Lf[k], prob_Lf_i)
tot_Lf_prob = sum(probabilities_Lf)
probabilities_Lf[0] += 1 - tot_Lf_prob # The lead is never negative. Thus, we move all the weight of "negative lead" tp zero


## Calculate Bf
# Define the values of for which you want to calculate Pr(Bf=k)
max_kB = max(100, (c - s) * e * 2)  # Adjust the range as needed
[values_of_kB, probabilities_Bf] = hf.probability_of_BpZ_given_chain(chain, s, c, e, f, max_kB//2, max_kB//2)

## Calculate Mf
# lambda_Z is the expected rate of honest chain growth
lambda_Z = 0

# preperations for calculating the lambda_Z lower bound
# numerically calculate expected values of x=1/2**b[i-1] and y=b[i]/2**b[i-1]
# Initialize the expected value
expected_value_of_x = 0.0
expected_value_of_y = 0.0

# Calculate the expected value using a sum
for k in range(0, e+3*e):  # Adjust the range as needed
    pmf = poisson.pmf(k, e*f)
    expected_value_of_x += (1 / (2 ** k)) * pmf
    expected_value_of_y += (k / (2 ** k)) * pmf

# Calculate the probability Pr(h > 0)
lambda_h = e * (1-f)
Pr_h_gt_0 = 1 - poisson.cdf(0, lambda_h)

# calculate lambda_Z lower bound
lambda_Z = Pr_h_gt_0 * ( lambda_h*expected_value_of_x + expected_value_of_y )

max_kM = 300     # Maximum value of k for plot. k is the good advantage that the adversary needs to catch up with.
max_i = 100     # Maximum value of epochs for the calculation (after which the probabilities become negligible)

# Initialize an array to store the probabilities of Mf
probabilities_Mf = np.zeros(max_kM + 1)
values_of_kM = range(max_kM + 1)

# Calculate Pr(Mf_i = k) for each i and find the maximum
for i in range(1, max_i + 1):
    lambda_b_i = i * e * f
    lambda_Z_i = i * lambda_Z
    # Calculate Pr(Mf = k) for each value of k
    for k in values_of_kM:
        prob_Mf_i = skellam.pmf(k, lambda_b_i, lambda_Z_i)
        probabilities_Mf[k] = max(probabilities_Mf[k], prob_Mf_i)
tot_Mf_prob = sum(probabilities_Mf)
probabilities_Mf[0] += 1 - tot_Mf_prob # probabilities_Mf[0] sums the probability of the adversary never catching up in the future.

## Calculate error probability upper bound "BAD event: Pr(Lf + Bf + Mf > k)"
# Define the range of values for k you wish to consider. k is the good advantage and Pr(Lf + Bf + Mf > k) is the probability of error given the observation of this good advantage
max_k = 500
values_of_k = np.arange(0, max_k)  # Adjust the range as needed

# Initialize an array to store the probabilities of BAD given a k good-advantage
error_probabilities = np.zeros(len(values_of_k))

# Calculate cumulative sums for Lf, Bf and Mf
cumulative_sum_Lf = np.cumsum(probabilities_Lf)
cumulative_sum_Bf = np.cumsum(probabilities_Bf)
cumulative_sum_Mf = np.cumsum(probabilities_Mf)

# Calculate the bound according to the equation ??? in the doc (Final step)
for k in values_of_k:
    sum_Lf_ge_k = cumulative_sum_Lf[-1] if k <= 0 else cumulative_sum_Lf[-1] - cumulative_sum_Lf[min(k - 1, len(cumulative_sum_Lf)-1)]
    double_sum = 0.0
    for l in range(k):
        sum_Bf_ge_k_min_l = cumulative_sum_Bf[-1] if k-l-1 <= 0 else cumulative_sum_Bf[-1] - cumulative_sum_Bf[min(k - l - 1, len(cumulative_sum_Bf)-1)]
        double_sum += probabilities_Lf[min(l, len(probabilities_Lf)-1)] * sum_Bf_ge_k_min_l
        for b in range(k-l):
            sum_Mf_ge_k_min_l_min_b = cumulative_sum_Mf[-1] if k-l-b-1 <= 0 else cumulative_sum_Mf[-1] - cumulative_sum_Mf[min(k - l - b - 1, len(cumulative_sum_Mf)-1)]
            double_sum += probabilities_Lf[min(l, len(probabilities_Lf)-1)] * probabilities_Bf[min(b, len(cumulative_sum_Bf)-1)] * sum_Mf_ge_k_min_l_min_b
    error_probabilities[k] = sum_Lf_ge_k + double_sum


good_ADV = sum(chain[s:c])
print("observed chain from epoch-"+str(s)+" till epoch-"+str(c)+" contains "+str(good_ADV)+" blocks.")
print("sanity check: the sum of Lf="+str(sum(probabilities_Lf)))
print("sanity check: the sum of Bf="+str(sum(probabilities_Bf)))
print("sanity check: the sum of Mf="+str(sum(probabilities_Mf)))
print("the error probability is "+str(error_probabilities[good_ADV]))


## Plot the probabilities
import matplotlib.pyplot as plt

# Plot the probability of Lf
plt.figure(figsize=(10, 6))
plt.plot(values_of_kL, probabilities_Lf, marker='o', linestyle='-')
plt.yscale('log')  # Set the y-axis to log scale
plt.title('Probability of Lf vs. k (Log Scale)')
plt.xlabel('k')
plt.ylabel('Pr(Lf = k)')
plt.grid(True)
plt.xticks(values_of_kL[::20])
plt.show()

# Plot the probability of Bf
plt.figure(figsize=(10, 6))
plt.plot(values_of_kB, probabilities_Bf, marker='o', linestyle='-')
plt.yscale('log')  # Set the y-axis to log scale
plt.title('Probability of Bf vs. k (Log Scale)')
plt.xlabel('k')
plt.ylabel('Pr(Bf = k)')
plt.grid(True)
plt.xticks(values_of_kB[::20])
plt.show()

# Plot the probability of Mf
plt.figure(figsize=(10, 6))
plt.plot(values_of_kM, probabilities_Mf, marker='o', linestyle='-')
plt.yscale('log')  # Set the y-axis to log scale
plt.title('Probability of Mf vs. k (Log Scale)')
plt.xlabel('k')
plt.ylabel('Pr(Mf = k)')
plt.grid(True)
plt.xticks(values_of_kM[::20])
plt.show()

# Plot the probability of BAD event
plt.figure(figsize=(10, 6))
plt.plot(values_of_k, error_probabilities, marker='o', linestyle='-')
plt.yscale('log')  # Set the y-axis to log scale
plt.title('Probability of error vs. observed lead (Log Scale)')
plt.xlabel('k')
plt.ylabel('Pr(Lf + Bf + Mf >= k)')
plt.grid(True)
plt.xticks(values_of_k[::5])
plt.show()