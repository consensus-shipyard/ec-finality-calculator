import numpy as np
import scipy.stats as ss

def finality_calc_validator(chain: list[int], blocks_per_epoch: float, byzantine_fraction: float, 
                            current_epoch: int, target_epoch: int) -> float:
    """
    Compute the probability that a previous blockchain tipset gets replaced.

    Parameters:
    - chain (list[int]): List of block counts per epoch.
    - blocks_per_epoch (float): Expected number of blocks per epoch.
    - byzantine_fraction (float): Upper bound on the fraction of malicious nodes in the network.
    - current_epoch (int): Current epoch.
    - target_epoch (int): Epoch for which finality is to be calculated.

    Returns:
    - error_probability (float): Probability of replacement of the tipset of the target_epoch
    """

    ####################
    # Parameters
    ####################

    # Max k for which to calculate Pr(Lf=k)
    max_k_L = 100
    # Max k for which to calculate Pr(Bf=k)
    max_k_B = (int)((current_epoch - target_epoch) * blocks_per_epoch)
    # Max k for which to calculate Pr(Mf=k)
    max_k_M = 100
    # Maximum number of epochs for the calculation (after which the pr become negligible)
    max_i_M = 100
    # Threshold at which the probability of an event is considered negligible
    negligible_threshold = 10**-15


    ####################
    # Preliminaries
    ####################
    rate_malicious_blocks = blocks_per_epoch * byzantine_fraction # upper bound
    rate_honest_blocks = blocks_per_epoch - rate_malicious_blocks # lower bound


    ####################
    # Compute Lf
    ####################

    # Initialize an array to store Pr(Lf=k)
    pr_Lf = [0] * (max_k_L + 1)

    # Calculate Pr(Lf=k) for each value of k
    for k in range(0, max_k_L + 1):
        sum_expected_adversarial_blocks_i = 0
        sum_chain_blocks_i = 0

        # Calculate Pr(Lf_i = k_i) for each epoch i, starting from epoch `s` under evaluation
        # and walking backwards to the last final tipset
        for i in range(target_epoch, current_epoch - 900, -1):
            sum_expected_adversarial_blocks_i += rate_malicious_blocks
            sum_chain_blocks_i -= chain[i - 1]
            # Poisson(k=k, lambda=sum_expected_adversarial_blocks_i, location=sum_chain_blocks_i)
            pr_Lf_i = ss.poisson.pmf(k, sum_expected_adversarial_blocks_i, sum_chain_blocks_i)
            # Take Pr(Lf=k) as the maximum over all i
            pr_Lf[k] = max(pr_Lf[k], pr_Lf_i)
        
        # Break if pr_Lf[k] becomes negligible
        if k > 1 and pr_Lf[k] < negligible_threshold and pr_Lf[k] < pr_Lf[k-1]:
            pr_Lf = pr_Lf[:(max_k_L:=k)+1]
            break

    # As the adversarial lead is never negative, the missing probability is added to k=0
    pr_Lf[0] += 1 - sum(pr_Lf)


    ####################
    # Compute Bf
    ####################

    # Initialize an array to store Pr(Bf=k)
    pr_Bf = [0] * (max_k_B + 1)

    # Calculate Pr(Bf=k) for each value of k
    for k in range(0, max_k_B + 1):
        # Poisson(k=k, lambda=sum_expected_adversarial_blocks, location=0)
        pr_Bf[k] = ss.poisson.pmf(k, (current_epoch - target_epoch + 1) * rate_malicious_blocks, 0)

        # Break if pr_Bf[k] becomes negligible
        if k > 1 and pr_Bf[k] < negligible_threshold and pr_Bf[k] < pr_Bf[k-1]:
            pr_Bf = pr_Bf[:(max_k_B:=k)+1]
            break


    ####################
    # Compute Mf
    ####################

    # Calculate the probability Pr(H>0)
    # Poisson (k=0, lambda=rate_honest_blocks, location=0)
    Pr_H_gt_0 = 1 - ss.poisson.pmf(0, rate_honest_blocks, 0)

    # Calculate E[Z]
    exp_Z = 0.0
    for k in range(0, (int) (4 * blocks_per_epoch)):  # Range stems from the distribution's moments
        # Poisson(k=k, lambda=rate_adv_blocks, location=0)
        pmf = ss.poisson.pmf(k, rate_malicious_blocks, 0)
        exp_Z += ((rate_honest_blocks + k) / (2 ** k)) * pmf

    # Lower bound on the growth rate of the public chain
    rate_public_chain = Pr_H_gt_0 * exp_Z

    # Initialize an array to store Pr(Mf=k)
    pr_Mf = [0] * (max_k_M + 1)

    # Calculate Pr(Mf = k) for each value of k
    for k in range(0, max_k_M + 1):
        # Calculate Pr(Mf_i = k) for each i and find the maximum
        for i in range(max_i_M, 0, -1):
            lambda_B_i = i * rate_malicious_blocks
            lambda_Z_i = i * rate_public_chain
            # Skellam(k=k, mu1=lambda_b_i, mu2=lambda_Z_i)
            prob_Mf_i = ss.skellam.pmf(k, lambda_B_i, lambda_Z_i)

            # Break if prob_Mf_i becomes negligible
            if prob_Mf_i < negligible_threshold and prob_Mf_i < pr_Mf[k]:
                break # Note: to be checked, but breaking here didn't change output in simulation

            # Take Pr(Mf=k) as the maximum over all i
            pr_Mf[k] = max(pr_Mf[k], prob_Mf_i)

        # Break if pr_Mf[k] becomes negligible
        if k > 1 and pr_Mf[k] < negligible_threshold and pr_Mf[k] < pr_Mf[k-1]:
            pr_Mf = pr_Mf[:(max_k_M:=k)+1]
            break

    # pr_Mf[0] collects the probability of the adversary never catching up in the future.
    pr_Mf[0] += 1 - sum(pr_Mf)


    ####################
    # Compute error probability upper bound "BAD event: Pr(Lf + Bf + Mf > k)"
    ####################

    # Max k for which to calculate Pr(BAD)
    # The sum of each max_k provides a strict upper bound, but one could pick a fraction.
    max_k = max_k_L + max_k_B + max_k_M 

    # Calculate cumulative sums for Lf, Bf, and Mf
    cumsum_Lf = np.cumsum(pr_Lf)
    cumsum_Bf = np.cumsum(pr_Bf)
    cumsum_Mf = np.cumsum(pr_Mf)

    # The observed chain has added weight equal to number of blocks since added
    k = sum(chain[target_epoch:current_epoch])

    # Calculate pr_error[k] for the observed added weight
    # Performs a convolution over the step probability vectors
    sum_Lf_ge_k = cumsum_Lf[-1]
    if k > 0:
        sum_Lf_ge_k -= cumsum_Lf[min(k - 1, max_k_L)] 
    double_sum = 0.0

    for l in range(0, k):
        sum_Bf_ge_k_min_l = cumsum_Bf[-1] 
        if k - l - 1 > 0:  
            sum_Bf_ge_k_min_l -= cumsum_Bf[min(k - l - 1, max_k_B)]
        double_sum += pr_Lf[min(l, max_k_L)] * sum_Bf_ge_k_min_l

        for b in range(0, k - l):
            sum_Mf_ge_k_min_l_min_b = cumsum_Mf[-1] 
            if k - l - b - 1 > 0:
                sum_Mf_ge_k_min_l_min_b -= cumsum_Mf[min(k - l - b - 1, max_k_M)]
            double_sum += pr_Lf[min(l, max_k_L)] * pr_Bf[min(b, max_k_B)] * sum_Mf_ge_k_min_l_min_b

    pr_error = sum_Lf_ge_k + double_sum
    
    # Get the probability of the adversary overtaking the observed weight
    # The conservative upper may exceed 1 in limit cases, so we cap the output.
    return min(pr_error, 1.0)


# Run with example data if file executed
def main() -> None:
    # Set some default parameters
    e = 5 # Expected number of blocks per epoch
    num_epochs = 905 # Length of generated chain history
    chain_health = 4.5/5 # Mean fraction of blocks in an epoch compared to the expectation
    f = 0.3 # Upper bound on the fraction of malicious nodes in the network
    c = num_epochs - 1  # Current epoch (end of history)
    s = c - 30 # Target epoch for which finality is calculated

    # Generate random chain (with fixed seed)
    rng = np.random.default_rng(0)
    chain = rng.poisson(chain_health * e, num_epochs)

    # Run calculator and print error probability 
    print(finality_calc_validator(chain, e, f, c, s))

if __name__ == "__main__":
    main()