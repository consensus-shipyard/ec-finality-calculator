import numpy as np
import scipy.stats as ss

# Calculate the conditional probability P(T = t | T >= c) for a Poisson random variable T.
def pr_poisson_conditional(lambda_T, t, c):
    if t < c:
        return 0.0  # Probability is 0 if t < c
    
    prob_T_ge_c = 1.0 - ss.poisson.cdf(c - 1, lambda_T)  # P(T >= c)
    prob_T_eq_t_and_T_ge_c = ss.poisson.pmf(t, lambda_T)  # P(T = t and T >= c)    
    
    if prob_T_ge_c == 0.0:
        return 0.0  # Avoid division by zero
    
    return prob_T_eq_t_and_T_ge_c / prob_T_ge_c

# calculate the probability of BpZ = B + Z based on the joint distribution of (B,Z | chain)
def pr_BpZ_given_chain(chain, start_epoch, end_epoch, e, f, max_z, max_b):
    probabilities_BpZ = [0] * (max_b + max_z + 1)

    num_epochs = end_epoch - start_epoch
    lambda_T = e * num_epochs
    lambda_H = (1-f) * lambda_T
    num_of_observed_blocks = sum(chain[start_epoch:end_epoch])

    for z in range(max_z + 1):
        pr_of_z_given_chain = pr_poisson_conditional(lambda_T, z + num_of_observed_blocks, num_of_observed_blocks)
        for b in range(max_b + 1):
            # Sum the joint probability
            probabilities_BpZ[b + z] += ss.poisson.pmf(z + num_of_observed_blocks - b, lambda_H,) * pr_of_z_given_chain
    
    return probabilities_BpZ

def finality_calc_actor(chain: list[int], blocks_per_epoch: float, byzantine_fraction: float, 
                            current_epoch: int, target_epoch: int) -> float:
    """
    Compute the probability that a previous blockchain tipset gets replaced.

    This code is EXPERIMENTAL and extremely slow. It is not part of our FRC.

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

    # Max k for which to calculate Pr(L=k)
    max_k_L = 100
    # Max k for which to calculate Pr(B=k)
    max_k_B = (int) ((current_epoch - target_epoch) * blocks_per_epoch)
    # Max k for which to calculate Pr(M=k)
    max_k_M = 100
    # Maximum number of epochs for the L calculation (after which the pr become negligible)
    max_i_L = 25
    # Maximum number of epochs for the M calculation (after which the pr become negligible)
    max_i_M = 100
    # Threshold at which the probability of an event is considered negligible
    negligible_threshold = 10**-25


    ####################
    # Preliminaries
    ####################
    rate_malicious_blocks = blocks_per_epoch * byzantine_fraction # upper bound
    rate_honest_blocks = blocks_per_epoch - rate_malicious_blocks # lower bound

    ####################
    # Compute L
    ####################

    # Initialize an array to store the probabilities of L
    pr_L = [0] * (max_k_L + 1)

    # Calculate BpZ given chain for each of the relevant past subchains
    sum_chain_blocks_i = 0

    # Calculate Pr(L_i = k_i) for each epoch i, starting from epoch `s` under evaluation
    # and walking backwards to the last final tipset (but stop after max_i_L epochs for perf)
    for i in range(target_epoch, target_epoch - max_i_L, -1):
        sum_chain_blocks_i += chain[i]
        max_relevant_BpZ = (int) (((target_epoch - i + 1) * 4 + 2) * blocks_per_epoch) # more than this, pr is negligible
        probabilities_based_on_BpZ = pr_BpZ_given_chain(chain, i - 1, target_epoch, blocks_per_epoch, byzantine_fraction, max_relevant_BpZ//2, max_relevant_BpZ//2)
        
        # Calculate Pr(L=k) for each value of k
        for k in range(0, max_k_L + 1):
            prob_L_i = 0 if k + sum_chain_blocks_i >= len(probabilities_based_on_BpZ) else probabilities_based_on_BpZ[k + sum_chain_blocks_i]
            pr_L[k] = max(pr_L[k], prob_L_i)

    # As the adversarial lead is never negative, the missing probability is added to k=0
    pr_L[0] += 1 - sum(pr_L)


    ####################
    # Compute B
    ####################

    pr_B = pr_BpZ_given_chain(chain, target_epoch, current_epoch, blocks_per_epoch, byzantine_fraction, max_k_B//2, max_k_B//2)


    ####################
    # Compute M
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

    # Initialize an array to store Pr(M=k)
    pr_M = [0] * (max_k_M + 1)

    # Calculate Pr(M = k) for each value of k
    for k in range(0, max_k_M + 1):
        # Calculate Pr(M_i = k) for each i and find the maximum
        for i in range(max_i_M, 0, -1):
            lambda_B_i = i * rate_malicious_blocks
            lambda_Z_i = i * rate_public_chain
            # Skellam(k=k, mu1=lambda_b_i, mu2=lambda_Z_i)
            prob_M_i = ss.skellam.pmf(k, lambda_B_i, lambda_Z_i)

            # Take Pr(M=k) as the maximum over all i
            pr_M[k] = max(pr_M[k], prob_M_i)

        # Break if pr_M[k] becomes negligible
        if k > 1 and pr_M[k] < negligible_threshold and pr_M[k] < pr_M[k-1]:
            pr_M = pr_M[:(max_k_M:=k)+1]
            break

    # pr_M[0] collects the probability of the adversary never catching up in the future.
    pr_M[0] += 1 - sum(pr_M)


    ####################
    # Compute error probability upper bound
    ####################

    # Calculate cumulative sums for L, B and M
    cumsum_L = np.cumsum(pr_L)
    cumsum_B = np.cumsum(pr_B)
    cumsum_M = np.cumsum(pr_M)

    # The observed chain has added weight equal to number of blocks since added
    k = sum(chain[target_epoch:current_epoch])    

    # Calculate pr_error[k] for the observed added weight
    # Performs a convolution over the step probability vectors
    sum_L_ge_k = cumsum_L[-1]
    if k > 0:
        sum_L_ge_k -= cumsum_L[min(k - 1, max_k_L)] 
    double_sum = 0.0

    for l in range(0, k):
        sum_B_ge_k_min_l = cumsum_B[-1] 
        if k - l - 1 > 0:  
            sum_B_ge_k_min_l -= cumsum_B[min(k - l - 1, max_k_B)]
        double_sum += pr_L[min(l, max_k_L)] * sum_B_ge_k_min_l

        for b in range(0, k - l):
            sum_M_ge_k_min_l_min_b = cumsum_M[-1] 
            if k - l - b - 1 > 0:
                sum_M_ge_k_min_l_min_b -= cumsum_M[min(k - l - b - 1, max_k_M)]
            double_sum += pr_L[min(l, max_k_L)] * pr_B[min(b, max_k_B)] * sum_M_ge_k_min_l_min_b

    pr_error = sum_L_ge_k + double_sum

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
    print(finality_calc_actor(chain, e, f, c, s))

if __name__ == "__main__":
    main()
