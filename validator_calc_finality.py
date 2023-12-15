import numpy as np
from scipy.stats import poisson
from scipy.stats import skellam


# Define parameters
import helper_functions

e = 5 # Expected number of blocks per epoch
f = 0.3 # portion of malicious power
num_variables = 905 # length of history
chain_health = 4.5/5 # mean precentage of blocks in an epoch compared to the expectation from a perfect network
c = 904 # current position (end of history)
s = c - 30 # slot in history for which finality is calculated

chain = np.random.poisson(chain_health * e, num_variables)


def validator_calc_finality(e, f, chain, c, s):
    # Generate the chain of Poisson random variables

    ## Calculate Lf
    # Initialize an array to store the probabilities of Lf
    probabilities_Lf = []

    # Define the values of k for which you want to calculate Pr(Lf=k)
    max_kL = 50  # Adjust the range as needed
    values_of_kL = np.arange(0, max_kL)

    # Calculate Pr(Lf=k) for each value of k
    for k in values_of_kL:
        max_probability = 0
        cumulative_sum_ef = 0
        cumulative_loc_i = 0

        # Calculate Pr(Lf_i = k_i) for each i and find the maximum
        for i in range(s, c - 900, -1):
            cumulative_sum_ef += e * f
            cumulative_loc_i -= chain[i - 1]
            lambda_Lf_i = cumulative_sum_ef
            prob_Lf_i = poisson.pmf(k, lambda_Lf_i, cumulative_loc_i)
            max_probability = max(max_probability, prob_Lf_i)

        probabilities_Lf.append(max_probability)
    tot_Lf_prob = sum(probabilities_Lf)
    probabilities_Lf[
        0] += 1 - tot_Lf_prob  # The lead is never negative. Thus, we move all the weight of "negative lead" tp zero

    ## Calculate Bf
    # Define the values of for which you want to calculate Pr(Bf=k)
    max_Kb = 202  # Adjust the range as needed
    values_of_kb = np.arange(0, max_Kb)
    lambda_Bf = (c - s + 1) * e * f
    probabilities_Bf = poisson.pmf(values_of_kb, lambda_Bf)

    ## Calculate Mf
    # lambda_Z is the expected rate of honest chain growth
    lambda_Z = 0

    # preperations for calculating the lambda_Z lower bound
    # numerically calculate expected values of x=1/2**b[i-1] and y=b[i]/2**b[i-1]
    # Initialize the expected value
    expected_value_of_x = 0.0
    expected_value_of_y = 0.0

    # Calculate the expected value using a sum
    for k in range(0, e + 3 * e):  # Adjust the range as needed
        pmf = poisson.pmf(k, e * f)
        expected_value_of_x += (1 / (2 ** k)) * pmf
        expected_value_of_y += (k / (2 ** k)) * pmf

    # Calculate the probability Pr(h > 0)
    lambda_h = e * (1 - f)
    Pr_h_gt_0 = 1 - poisson.cdf(0, lambda_h)

    # calculate lambda_Z lower bound
    lambda_Z = Pr_h_gt_0 * (lambda_h * expected_value_of_x + expected_value_of_y)

    max_kM = 80  # Maximum value of k for plot. k is the good advantage that the adversary needs to  catch up with.
    max_i = 100  # Maximum value of epochs for the calculation (after which the probabilities become negligible)

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
    probabilities_Mf[
        0] += 1 - tot_Mf_prob  # probabilities_Mf[0] sums the probability of the adversary never catching up in the future.

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
        sum_Lf_ge_k = cumulative_sum_Lf[-1] if k <= 0 else cumulative_sum_Lf[-1] - cumulative_sum_Lf[
            min(k - 1, len(cumulative_sum_Lf) - 1)]
        double_sum = 0.0
        for l in range(k):
            sum_Bf_ge_k_min_l = cumulative_sum_Bf[-1] if k - l - 1 <= 0 else cumulative_sum_Bf[-1] - cumulative_sum_Bf[
                min(k - l - 1, len(cumulative_sum_Bf) - 1)]
            double_sum += probabilities_Lf[min(l, len(probabilities_Lf) - 1)] * sum_Bf_ge_k_min_l
            for b in range(k - l):
                sum_Mf_ge_k_min_l_min_b = cumulative_sum_Mf[-1] if k - l - b - 1 <= 0 else cumulative_sum_Mf[-1] - \
                                                                                           cumulative_sum_Mf[
                                                                                               min(k - l - b - 1,
                                                                                                   len(cumulative_sum_Mf) - 1)]
                double_sum += probabilities_Lf[min(l, len(probabilities_Lf) - 1)] * probabilities_Bf[
                    min(b, len(cumulative_sum_Bf) - 1)] * sum_Mf_ge_k_min_l_min_b
        error_probabilities[k] = sum_Lf_ge_k + double_sum

    good_ADV = sum(chain[s:c])
    # print("observed chain from epoch-" + str(s) + " till epoch-" + str(c) + " contains " + str(good_ADV) + " blocks.")
    # print("sanity check: the sum of Lf=" + str(sum(probabilities_Lf)))
    # print("sanity check: the sum of Bf=" + str(sum(probabilities_Bf)))
    # print("sanity check: the sum of Mf=" + str(sum(probabilities_Mf)))
    # print("the error probability is " + str(error_probabilities[good_ADV]))
    # blocks_needed = helper_functions.find_closest(error_probabilities, 2**(-30))
    # print("the error probability after "+str(blocks_needed)+" good build is " + str(error_probabilities[blocks_needed]))
    return error_probabilities[good_ADV]


# ans = validator_calc_finality(e, f, chain, c, s)
# print(ans)



#
# # Generate the chain of Poisson random variables
# chain = np.random.poisson(chain_health*e, num_variables)
#
# ## Calculate Lf
# # Initialize an array to store the probabilities of Lf
# probabilities_Lf = []
#
# # Define the values of k for which you want to calculate Pr(Lf=k)
# max_kL = 50   # Adjust the range as needed
# values_of_kL = np.arange(0, max_kL)
#
# # Calculate Pr(Lf=k) for each value of k
# for k in values_of_kL:
#     max_probability = 0
#     cumulative_sum_ef = 0
#     cumulative_loc_i = 0
#
#     # Calculate Pr(Lf_i = k_i) for each i and find the maximum
#     for i in range(s, c-900, -1):
#         cumulative_sum_ef += e * f
#         cumulative_loc_i -= chain[i-1]
#         lambda_Lf_i = cumulative_sum_ef
#         prob_Lf_i = poisson.pmf(k, lambda_Lf_i, cumulative_loc_i)
#         max_probability = max(max_probability, prob_Lf_i)
#
#     probabilities_Lf.append(max_probability)
# tot_Lf_prob = sum(probabilities_Lf)
# probabilities_Lf[0] += 1 - tot_Lf_prob # The lead is never negative. Thus, we move all the weight of "negative lead" tp zero
#
#
# ## Calculate Bf
# # Define the values of for which you want to calculate Pr(Bf=k)
# max_Kb = 202  # Adjust the range as needed
# values_of_kb = np.arange(0, max_Kb)
# lambda_Bf = (c - s + 1) * e * f
# probabilities_Bf = poisson.pmf(values_of_kb, lambda_Bf)
#
#
# ## Calculate Mf
# # lambda_Z is the expected rate of honest chain growth
# lambda_Z = 0
#
# # preperations for calculating the lambda_Z lower bound
# # numerically calculate expected values of x=1/2**b[i-1] and y=b[i]/2**b[i-1]
# # Initialize the expected value
# expected_value_of_x = 0.0
# expected_value_of_y = 0.0
#
# # Calculate the expected value using a sum
# for k in range(0, e+3*e):  # Adjust the range as needed
#     pmf = poisson.pmf(k, e*f)
#     expected_value_of_x += (1 / (2 ** k)) * pmf
#     expected_value_of_y += (k / (2 ** k)) * pmf
#
# # Calculate the probability Pr(h > 0)
# lambda_h = e * (1-f)
# Pr_h_gt_0 = 1 - poisson.cdf(0, lambda_h)
#
# # calculate lambda_Z lower bound
# lambda_Z = Pr_h_gt_0 * ( lambda_h*expected_value_of_x + expected_value_of_y )
#
# max_kM = 80     # Maximum value of k for plot. k is the good advantage that the adversary needs to  catch up with.
# max_i = 100     # Maximum value of epochs for the calculation (after which the probabilities become negligible)
#
# # Initialize an array to store the probabilities of Mf
# probabilities_Mf = np.zeros(max_kM + 1)
# values_of_kM = range(max_kM + 1)
#
# # # Calculate Pr(Mf = k) for each value of k
# # for k in values_of_kM:
# #     max_probability = 0
# #
# #     # Calculate Pr(Mf_i = k) for each i and find the maximum
# #     for i in range(1, max_i + 1):
# #         lambda_b_i = i * e * f
# #         lambda_Z_i = i * lambda_Z
# #         prob_Mf_i = skellam.pmf(k, lambda_b_i, lambda_Z_i)
# #         max_probability = max(max_probability, prob_Mf_i)
# #
# #     probabilities_Mf[k] = max_probability
# # tot_Mf_prob = sum(probabilities_Mf)
# # probabilities_Mf[0] += 1 - tot_Mf_prob
#
# # Calculate Pr(Mf_i = k) for each i and find the maximum
# for i in range(1, max_i + 1):
#     lambda_b_i = i * e * f
#     lambda_Z_i = i * lambda_Z
#     # Calculate Pr(Mf = k) for each value of k
#     for k in values_of_kM:
#         prob_Mf_i = skellam.pmf(k, lambda_b_i, lambda_Z_i)
#         probabilities_Mf[k] = max(probabilities_Mf[k], prob_Mf_i)
# tot_Mf_prob = sum(probabilities_Mf)
# probabilities_Mf[0] += 1 - tot_Mf_prob # probabilities_Mf[0] sums the probability of the adversary never catching up in the future.
#
# ## Calculate error probability upper bound "BAD event: Pr(Lf + Bf + Mf > k)"
# # Define the range of values for k you wish to consider. k is the good advantage and Pr(Lf + Bf + Mf > k) is the probability of error given the observation of this good advantage
# max_k = 200
# values_of_k = np.arange(0, max_k)  # Adjust the range as needed
#
# # Initialize an array to store the probabilities of BAD given a k good-advantage
# error_probabilities = np.zeros(len(values_of_k))
#
# # Calculate cumulative sums for Lf, Bf and Mf
# cumulative_sum_Lf = np.cumsum(probabilities_Lf)
# cumulative_sum_Bf = np.cumsum(probabilities_Bf)
# cumulative_sum_Mf = np.cumsum(probabilities_Mf)
#
# # Calculate the bound according to the equation ??? in the doc
# for k in values_of_k:
#     sum_Lf_ge_k = cumulative_sum_Lf[-1] if k <= 0 else cumulative_sum_Lf[-1] - cumulative_sum_Lf[min(k - 1, len(cumulative_sum_Lf)-1)]
#     double_sum = 0.0
#     for l in range(k):
#         sum_Bf_ge_k_min_l = cumulative_sum_Bf[-1] if k-l-1 <= 0 else cumulative_sum_Bf[-1] - cumulative_sum_Bf[min(k - l - 1, len(cumulative_sum_Bf)-1)]
#         double_sum += probabilities_Lf[min(l, len(probabilities_Lf)-1)] * sum_Bf_ge_k_min_l
#         for b in range(k-l):
#             sum_Mf_ge_k_min_l_min_b = cumulative_sum_Mf[-1] if k-l-b-1 <= 0 else cumulative_sum_Mf[-1] - cumulative_sum_Mf[min(k - l - b - 1, len(cumulative_sum_Mf)-1)]
#             double_sum += probabilities_Lf[min(l, len(probabilities_Lf)-1)] * probabilities_Bf[min(b, len(cumulative_sum_Bf)-1)] * sum_Mf_ge_k_min_l_min_b
#     error_probabilities[k] = sum_Lf_ge_k + double_sum
#
#
# good_ADV = sum(chain[s:c])
# print("observed chain from epoch-"+str(s)+" till epoch-"+str(c)+" contains "+str(good_ADV)+" blocks.")
# print("sanity check: the sum of Lf="+str(sum(probabilities_Lf)))
# print("sanity check: the sum of Bf="+str(sum(probabilities_Bf)))
# print("sanity check: the sum of Mf="+str(sum(probabilities_Mf)))
# print("the error probability is "+str(error_probabilities[good_ADV]))
#
# ## Plot the probabilities
# import matplotlib.pyplot as plt
#
# # Plot the probability of Lf
# plt.figure(figsize=(10, 6))
# plt.plot(values_of_kL, probabilities_Lf, marker='o', linestyle='-')
# plt.yscale('log')  # Set the y-axis to log scale
# plt.title('Probability of Lf vs. k (Log Scale)')
# plt.xlabel('k')
# plt.ylabel('Pr(Lf = k)')
# plt.grid(True)
# plt.xticks(values_of_kL[::20])
# plt.show()
#
# # Plot the probability of Bf
# plt.figure(figsize=(10, 6))
# plt.plot(values_of_kb, probabilities_Bf, marker='o', linestyle='-')
# plt.yscale('log')  # Set the y-axis to log scale
# plt.title('Probability of Bf vs. k (Log Scale)')
# plt.xlabel('k')
# plt.ylabel('Pr(Bf = k)')
# plt.grid(True)
# plt.xticks(values_of_kb[::20])
# plt.show()
#
# # Plot the probability of Mf
# plt.figure(figsize=(10, 6))
# plt.plot(values_of_kM, probabilities_Mf, marker='o', linestyle='-')
# plt.yscale('log')  # Set the y-axis to log scale
# plt.title('Probability of Mf vs. k (Log Scale)')
# plt.xlabel('k')
# plt.ylabel('Pr(Mf = k)')
# plt.grid(True)
# plt.xticks(values_of_kM[::20])
# plt.show()
#
# # Plot the probability of BAD event
# plt.figure(figsize=(10, 6))
# plt.plot(values_of_k, error_probabilities, marker='o', linestyle='-')
# plt.yscale('log')  # Set the y-axis to log scale
# plt.title('Probability of error vs. observed lead (Log Scale)')
# plt.xlabel('k')
# plt.ylabel('Pr(Lf + Bf + Mf >= k)')
# plt.grid(True)
# plt.xticks(values_of_k[::5])
# plt.show()