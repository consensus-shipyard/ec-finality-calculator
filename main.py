import numpy as np
from scipy.stats import poisson
from scipy.stats import skellam


# Define parameters
e = 5
f = 0.3
num_variables = 905 # length of history
c = 904 # current position (end of history)
s = c - 30 # slot in history for which finality is calculated

# Generate the chain of Poisson random variables
chain = np.random.poisson(0.9*e, num_variables)

## Step 1
# helper function
def conditional_poisson_probability(lambda_T, t, c):
    """
    Calculate the conditional probability P(T = t | T >= c) for a Poisson random variable T.

    Args:
    lambda_T (float): The parameter of the Poisson distribution.
    c (int): The condition (T >= c).
    t (int): The value at which to evaluate the conditional probability.

    Returns:
    float: The conditional probability P(T = t | T >= c).
    """
    if t < c:
        return 0.0  # Probability is 0 if t < c
    prob_T_ge_c = 1.0 - poisson.cdf(c - 1, lambda_T)  # P(T >= c)
    if prob_T_ge_c == 0.0:
        return 0.0  # Avoid division by zero
    prob_T_equals_t_and_T_ge_c = poisson.pmf(t, lambda_T)  # P(T = t and T >= c)
    return prob_T_equals_t_and_T_ge_c / prob_T_ge_c


def probability_of_Z_given_chain(lambda_T, z, c):
    """
    Calculate the conditional probability P(Z = z | T >= c) for the random variable Z accounting for blocks outside the obsereved chain section.

    Args:
    lambda_T (float): The parameter of the Poisson distribution T. I.e., the expected number of total blocks production in the relevant chain section
    c (int): The condition (T >= c). I.e., the number of blocks in the relevant chain section
    z (int): The value at which to evaluate the conditional probability.

    Returns:
    float: The conditional probability P(Z = z | T >= c) which is equal to P(T = z+c | T >= c).
    """
    return conditional_poisson_probability(lambda_T, z+c, c)

# step 2: account for malicious blocks
# calculate the Pr(B|T) = Pr(H=T-B)
def probability_of_B_given_T(lambda_H, b, t):
    """
     Calculate the probability of B given T, that is, Pr(B = b | T = t).
     B is the amount of malicious blocks that were produced during the relevant chain section.
     T is the total number of  blocks that were produced during the relevant chain section.

     Args:
     lambda_H (float): The parameter of the Poisson distribution H. I.e., the expected number of total honest blocks produced in the relevant chain section
     t (int): The condition (T = t).
     b (int): The value at which to evaluate the conditional probability.

     Returns:
     float: The conditional probability P(B = b | T = t) which is equal to P(H = t - b).
     """
    return poisson.pmf(t - b,  lambda_H)

# calculate the probability of BpZ = B + Z base on the joint distribution of (B,Z | chain)
def probability_of_BpZ_given_chain(chain, start_epoch, end_epoch, e, f, max_z, max_b):
    max_BpZ = max_b+max_z
    values_of_BpZ = np.arange(0, max_BpZ+1)
    probabilities_BpZ = np.zeros(max_b + max_z + 1)

    num_epochs = (end_epoch - start_epoch)
    lambda_T = e * num_epochs
    lambda_H = (1-f) * lambda_T
    num_of_observed_blocks = sum(chain[start_epoch:end_epoch])

    for b in range(max_b + 1):
            for z in range(max_z + 1):
                sum_b_z = b + z
                joint_prob = probability_of_B_given_T(lambda_H, b, z + num_of_observed_blocks) * probability_of_Z_given_chain(lambda_T, z, num_of_observed_blocks)
                probabilities_BpZ[sum_b_z] += joint_prob
    return [values_of_BpZ, probabilities_BpZ]

# Define the values of Z and B for which we wish to calculate
max_z = 100  # Adjust the range as needed
max_b = 107  # Adjust the range as needed

# Define the section of interest
start_epoch = s
end_epoch = c

[values_of_BpZ, probabilities_BpZ] = probability_of_BpZ_given_chain(chain, start_epoch, end_epoch, e, f, max_z, max_b)


# Step 3: calculate Lf and Bf
# TODO: Calculate Lf
# Initialize an array to store the probabilities of Lf
probabilities_Lf = []

# Define the values of k for which you want to calculate Pr(Lf=k)
max_kL = 20   # Adjust the range as needed
values_of_kL = np.arange(0, max_kL)

# compute the distribution of Lf
# Define the values of Z and B for which we wish to calculate
# DELETE: max_z = 100  # Adjust the range as needed
# DELETE: max_b = 107  # Adjust the range as needed
# Define the section of interest
start_epoch = s-10 # for computation reasons we limit the amount of lookback
end_epoch = s

# Calculate Pr(Lf=k) for each value of k
for k in values_of_kL:
    max_probability = 0
    cumulative_sum_ef = 0
    cumulative_loc_i = 0

    # Calculate Pr(Lf_i = k_i) for each i and find the maximum
    prev_i = end_epoch
    for i in range(end_epoch, start_epoch, -1):
        print(i)
        cumulative_loc_i += chain[i]
        print("start = "+str(i-1))
        print("end = " + str(end_epoch))
        print("k+(end_epoch-i)+1 = " + str(k+(end_epoch-i)+1))
        _, probabilities_based_on_BpZ = probability_of_BpZ_given_chain(chain, i-1, end_epoch, e, f, e * (k+end_epoch-i+5), e * (k+end_epoch-i+5))
        prob_Lf_i = probabilities_based_on_BpZ[k + cumulative_loc_i]
        print("prob_Lf_i = "+str(prob_Lf_i))
        max_probability = max(max_probability, prob_Lf_i)
        prev_i = i

    probabilities_Lf.append(max_probability)
tot_Lf_prob = sum(probabilities_Lf)
probabilities_Lf[0] += 1 - tot_Lf_prob


## Calculate Bf
# Define the values of for which you want to calculate Pr(Bf=k)
max_Kb = 202  # Adjust the range as needed
values_of_kb = np.arange(0, max_Kb)

# compute the distribution of Bf
# Define the values of Z and B for which we wish to calculate
max_z = 100  # Adjust the range as needed
max_b = 107  # Adjust the range as needed
# Define the section of interest
start_epoch = s
end_epoch = c
_, probabilities_based_on_BpZ = probability_of_BpZ_given_chain(chain, start_epoch, end_epoch, e, f, max_z, max_b)
probabilities_Bf = [probabilities_based_on_BpZ[k] for k in values_of_kb]


## Calculate Mf
# the expected rate of honest chain growth
lambda_Z = 0

# preperations for calculating the lambda_Z lower bound
# numerically calculate expected values of x=1/2**b[i-1] and y=b[i]/2**b[i-1]
# Initialize the expected value
expected_value_of_x = 0.0
expected_value_of_y = 0.0

# Calculate the expected value using a sum
for k in range(0, 3*e):  # Adjust the range as needed
    pmf = poisson.pmf(k, e*f)
    expected_value_of_x += (1 / (2 ** k)) * pmf
    expected_value_of_y += (k / (2 ** k)) * pmf

# Calculate the probability Pr(h > 0)
lambda_h = e * (1-f)
Pr_h_gt_0 = 1 - poisson.cdf(0, lambda_h)

# calculate lambda_Z lower bound
lambda_Z = Pr_h_gt_0 * ( lambda_h*expected_value_of_x + expected_value_of_y )

max_kM = 80     # Maximum value of k for plot. k is the good advantage that the adversary needs to  catch up with.
max_i = 50     # Maximum value of epochs for the calculation (after which the probabilities become negligible)

# Initialize an array to store the probabilities of Mf
probabilities_Mf = np.zeros(max_kM + 1)
values_of_kM = range(max_kM + 1)

# Calculate Pr(Mf = k) for each value of k
for k in values_of_kM:
    max_probability = 0

    # Calculate Pr(Mf_i = k) for each i and find the maximum
    for i in range(1, max_i + 1):
        lambda_b_i = i * e * f
        lambda_Z_i = i * lambda_Z
        prob_Mf_i = skellam.pmf(k, lambda_b_i, lambda_Z_i)
        max_probability = max(max_probability, prob_Mf_i)

    probabilities_Mf[k] = max_probability
tot_Mf_prob = sum(probabilities_Mf)
probabilities_Mf[0] += 1 - tot_Mf_prob


## Calculate error probability upper bound "BAD event: Pr(Lf + Bf + Mf > k)"
# Define the range of values for k you wish to consider. k is the good advantage and Pr(Lf + Bf + Mf > k) is the probability of error given the observation of this good advantage
max_k = 200
values_of_k = np.arange(0, max_k)  # Adjust the range as needed

# Initialize an array to store the probabilities of BAD given a k good-advantage
error_probabilities = np.zeros(len(values_of_k))

# Calculate cumulative sums for Lf, Bf and Mf
cumulative_sum_Lf = np.cumsum(probabilities_Lf)
cumulative_sum_Bf = np.cumsum(probabilities_Bf)
cumulative_sum_Mf = np.cumsum(probabilities_Mf)

# Calculate the bound according to the equation ??? in the doc
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
# Plot the probability of BpZ
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(values_of_BpZ, probabilities_BpZ, marker='o', linestyle='-')
# plt.yscale('log')  # Set the y-axis to log scale
# plt.title('Probability of BpZ vs. k (Log Scale)')
plt.title('Probability of BpZ vs. k')
plt.xlabel('k')
plt.ylabel('Pr(BpZ = k)')
plt.grid(True)
plt.xticks(values_of_BpZ[::20])
plt.show()

# Plot the probability of Lf
plt.figure(figsize=(10, 6))
plt.plot(values_of_kL, probabilities_Lf, marker='o', linestyle='-')
plt.yscale('log')  # Set the y-axis to log scale
plt.title('Probability of Lf vs. k (Log Scale)')
plt.xlabel('k')
plt.ylabel('Pr(Lf = k)')
plt.grid(True)
plt.xticks(values_of_kL[::10])
plt.show()

# Plot the probability of Bf
plt.figure(figsize=(10, 6))
plt.plot(values_of_kb, probabilities_Bf, marker='o', linestyle='-')
plt.yscale('log')  # Set the y-axis to log scale
plt.title('Probability of Bf vs. k (Log Scale)')
plt.xlabel('k')
plt.ylabel('Pr(Bf = k)')
plt.grid(True)
plt.xticks(values_of_kb[::20])
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