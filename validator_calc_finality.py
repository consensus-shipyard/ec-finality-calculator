import numpy as np
from scipy.stats import poisson
from scipy.stats import skellam

# Define parameters
import helper_functions

def validator_calc_finality(chain, blocks_per_epoch, byzantine_fraction, current_epoch, target_epoch, negligible_threshold):
    """
    Calculate the finality of a validator in a blockchain network.

    Parameters:
    - chain (list): List of block counts per epoch.
    - blocks_per_epoch (int): Expected number of blocks per epoch epoch.
    - byzantine_fraction (float): Upper bound on the fraction of malicious nodes in the network.
    - current_epoch (int): Current epoch.
    - target_epoch (int): Epoch for which finality is to be calculated.
    - negligible_threshold (float): Threshold at which the probability of an event is negligible.

    Returns:
    - error_probability (float): Probability of replacement of the tipset of the target_epoch
    """

    ####################
    # Parameters
    ####################

    # Max k for which to calculate Pr(Lf=k)
    max_k_L = 100
    # Max k for which to calculate Pr(Bf=k)
    max_k_B = (int) ((current_epoch - target_epoch) * blocks_per_epoch)
    # Max k for which to calculate Pr(Mf=k)
    max_k_M = 100
    # Maximum number of epochs for the calculation (after which the pr become negligible)
    max_i_M = 100    


    ####################
    # Preliminaries
    ####################
    rate_malicious_blocks = blocks_per_epoch * byzantine_fraction # upper bound
    rate_honest_blocks = blocks_per_epoch * (1-byzantine_fraction) # lower bound


    ####################
    # Compute Lf
    ####################

    # Q: We'll probably need to better motivate these parameters and provide them in the FIP
    # There are theoretical approaches here, could also heuristically look at decreasing returns 
    # There's no single answer, depends on chain conditions. 
    # Could stop on pr_Lf[k]<\epsilon (10^-10), but not monotonically decreasing in theory

    # Initialize an array to store Pr(Lf=k)
    pr_Lf = [0] * max_k_L

    # Calculate Pr(Lf=k) for each value of k
    for k in range(0, max_k_L):
        sum_expected_adversarial_blocks_i = 0
        sum_chain_blocks_i = 0

        # Calculate Pr(Lf_i = k_i) for each epoch i, starting from epoch `s` under evaluation
        # and walking backwards to the last final tipset
        for i in range(target_epoch, current_epoch - 900, -1):
            sum_expected_adversarial_blocks_i += rate_malicious_blocks
            sum_chain_blocks_i -= chain[i - 1]
            # Poisson(k=k, lambda=sum_expected_adversarial_blocks_i, location=sum_chain_blocks_i)
            pr_Lf_i = poisson.pmf(k, sum_expected_adversarial_blocks_i, sum_chain_blocks_i)
            # Take Pr(Lf=k) as the maximum over all i
            pr_Lf[k] = max(pr_Lf[k], pr_Lf_i)
        
        # Break if pr_Lf[k] becomes negligible
        if k > 1 and pr_Lf[k] < negligible_threshold and pr_Lf[k] < pr_Lf[k-1]:
            print("Breaking Lf at ", k)
            max_k_L = k
            pr_Lf = pr_Lf[:max_k_L+1]
            break

    # As the adversarial lead is never negative, the missing probability is added to k=0
    pr_Lf[0] += 1 - sum(pr_Lf)


    ####################
    # Compute Bf
    ####################

    # Initialize an array to store Pr(Bf=k)
    pr_Bf = [0] * max_k_B

    # Calculate Pr(Bf=k) for each value of k
    for k in range(0, max_k_B):
        # Poisson(k=k, lambda=sum_expected_adversarial_blocks, location=0)
        pr_Bf[k] = poisson.pmf(k, (current_epoch - target_epoch + 1) * rate_malicious_blocks, 0)

        # Break if pr_Bf[k] becomes negligible
        if k > 1 and pr_Bf[k] < negligible_threshold and pr_Bf[k] < pr_Bf[k-1]:
            print("Breaking Bf at ", k)
            max_k_B = k
            pr_Bf = pr_Bf[:max_k_B+1]
            break


    ####################
    # Compute Mf
    ####################

    # lambda_Z is the lower bound on the growth rate of the public chain
    lambda_Z = 0

    # Calculate the probability Pr(H > 0)
    # Poisson (k=0, lambda=rate_honest_blocks, location=0)
    Pr_H_gt_0 = 1 - poisson.pmf(0, rate_honest_blocks, 0)

    # Calculate the expected value of Z
    exp_Z = 0.0
    for k in range(0, 4 * blocks_per_epoch):  # Q: wtf? Heuristic looking for negligible numbers
        # Poisson(k=k, lambda=rate_adv_blocks, location=0)
        pmf = poisson.pmf(k, rate_malicious_blocks, 0)
        exp_Z += ((rate_honest_blocks + k) / (2 ** k)) * pmf

    # calculate lambda_Z lower bound
    # lambda_Z = Pr_h_gt_0 * ( lambda_h*expected_value_of_x + expected_value_of_y )
    # Q: Isn't this technically lambda_Z_prime?
    lambda_Z = Pr_H_gt_0 * exp_Z

    # Initialize an array to store Pr(Mf=k)
    pr_Mf = [0] * (max_k_M + 1)  # Q: Why all the +1s?

    # Calculate Pr(Mf = k) for each value of k
    for k in range(0, max_k_M + 1):
        # Calculate Pr(Mf_i = k) for each i and find the maximum
        for i in range(1, max_i_M + 1):
            lambda_b_i = i * rate_malicious_blocks
            lambda_Z_i = i * lambda_Z
            # Skellam(mu1=lambda_b_i, mu2=lambda_Z_i)
            prob_Mf_i = skellam.pmf(k, lambda_b_i, lambda_Z_i)
            # Take Pr(Mf=k) as the maximum over all i
            pr_Mf[k] = max(pr_Mf[k], prob_Mf_i)

        # Break if pr_Mf[k] becomes negligible
        if k > 1 and pr_Mf[k] < negligible_threshold and pr_Mf[k] < pr_Mf[k-1]:
            print("Breaking Mf at ", k)
            max_k_M = k
            pr_Mf = pr_Mf[:max_k_M+1]
            break

    # pr_Mf[0] collects the probability of the adversary never catching up in the future.
    pr_Mf[0] += 1 - sum(pr_Mf)


    ####################
    # Compute error probability upper bound "BAD event: Pr(Lf + Bf + Mf > k)"
    ####################

    # Define the range of values for k you wish to consider. 
    # k is the good advantage and Pr(Lf + Bf + Mf > k) is the probability of error given the observation of this good advantage
    max_k = max_k_L + max_k_B + max_k_M 

    # Initialize an array to store the pr of BAD given a k good-advantage
    pr_error = [0] * max_k

    # Calculate cumulative sums for Lf, Bf, and Mf
    cumsum_Lf = np.cumsum(pr_Lf)
    cumsum_Bf = np.cumsum(pr_Bf)
    cumsum_Mf = np.cumsum(pr_Mf)

    # Calculate the bound
    for k in range(0, max_k):
        sum_Lf_ge_k = cumsum_Lf[-1]
        if k > 0:
            sum_Lf_ge_k -= cumsum_Lf[min(k - 1, len(cumsum_Lf) - 1)] 
        double_sum = 0.0

        for l in range(0, k):
            sum_Bf_ge_k_min_l = cumsum_Bf[-1] 
            if k - l - 1 > 0:  # Q: Why all the -1s?
                sum_Bf_ge_k_min_l -= cumsum_Bf[min(k - l - 1, len(cumsum_Bf) - 1)]
            double_sum += pr_Lf[min(l, len(pr_Lf) - 1)] * sum_Bf_ge_k_min_l

            for b in range(0, k - l):
                sum_Mf_ge_k_min_l_min_b = cumsum_Mf[-1] 
                if k - l - b - 1 > 0:
                    sum_Mf_ge_k_min_l_min_b -= cumsum_Mf[min(k - l - b - 1, len(cumsum_Mf) - 1)]
                double_sum += pr_Lf[min(l, len(pr_Lf) - 1)] * pr_Bf[min(b, len(cumsum_Bf) - 1)] * sum_Mf_ge_k_min_l_min_b

        pr_error[k] = sum_Lf_ge_k + double_sum

    good_advantage = int(sum(chain[target_epoch:current_epoch]))
    return pr_error[good_advantage]
