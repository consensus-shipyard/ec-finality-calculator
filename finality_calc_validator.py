import numpy as np
import scipy.stats as ss

def finality_calc_validator(chain: list[int], blocks_per_epoch: float, byzantine_fraction: float, 
                            current_epoch: int, target_epoch: int) -> float:
    """
    Compute the probability that a previous blockchain tipset gets replaced from the
    perspective of a validator.

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
    max_k_L = 400
    # Max k for which to calculate Pr(B=k)
    max_k_B = (int)((current_epoch - target_epoch) * blocks_per_epoch)
    # Max k for which to calculate Pr(M=k)
    max_k_M = 400
    # Maximum number of epochs for the calculation (after which the pr become negligible)
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

    # Initialize an array to store Pr(L=k)
    pr_L = [0] * (max_k_L + 1)

    # Calculate Pr(L=k) for each value of k
    for k in range(0, max_k_L + 1):
        sum_expected_adversarial_blocks_i = 0
        sum_chain_blocks_i = 0

        # Calculate Pr(L_i = k_i) for each epoch i, starting from epoch `s` under evaluation
        # and walking backwards to the last final tipset
        for i in range(target_epoch, current_epoch - 900, -1):
            sum_expected_adversarial_blocks_i += rate_malicious_blocks
            sum_chain_blocks_i += chain[i - 1]
            # Poisson(k=k, lambda=sum(f*e))
            pr_L_i = ss.poisson.pmf(k + sum_chain_blocks_i, sum_expected_adversarial_blocks_i)
            # Take Pr(L=k) as the maximum over all i
            pr_L[k] = max(pr_L[k], pr_L_i)
        
        # Break if pr_L[k] becomes negligible
        if k > 1 and pr_L[k] < negligible_threshold and pr_L[k] < pr_L[k-1]:
            pr_L = pr_L[:(max_k_L:=k)+1]
            break

    # As the adversarial lead is never negative, the missing probability is added to k=0
    pr_L[0] += 1 - sum(pr_L)


    ####################
    # Compute B
    ####################

    # Initialize an array to store Pr(B=k)
    pr_B = [0] * (max_k_B + 1)

    # Calculate Pr(B=k) for each value of k
    for k in range(0, max_k_B + 1):
        # Poisson(k=k, lambda=sum(f*e))
        pr_B[k] = ss.poisson.pmf(k, (current_epoch - target_epoch) * rate_malicious_blocks)

        # Break if pr_B[k] becomes negligible
        if k > 1 and pr_B[k] < negligible_threshold and pr_B[k] < pr_B[k-1]:
            pr_B = pr_B[:(max_k_B:=k)+1]
            break


    ####################
    # Compute M
    ####################

    # Calculate the probability Pr(H>0)
    # Poisson (k=0, lambda=h*e)
    Pr_H_gt_0 = 1 - ss.poisson.pmf(0, rate_honest_blocks)

    # Calculate E[Z]
    exp_Z = 0.0
    for k in range(0, (int) (4 * blocks_per_epoch)):  # Range stems from the distribution's moments
        # Poisson(k=k, lambda=f*e)
        pmf = ss.poisson.pmf(k, rate_malicious_blocks)
        exp_Z += ((rate_honest_blocks + k) / (2 ** k)) * pmf

    # Lower bound on the growth rate of the public chain
    rate_public_chain = Pr_H_gt_0 * exp_Z

    # Initialize an array to store Pr(M=k)
    pr_M = [0] * (max_k_M + 1)

    # Calculate Pr(M = k) for each value of k
    for k in range(0, max_k_M + 1):
        # Calculate Pr(M_i = k) for each i and find the maximum
        for i in range(max_i_M, 0, -1):
            # Skellam(k=k, mu1=n*e*f, mu2=n*E[Z])
            prob_M_i = ss.skellam.pmf(k, i * rate_malicious_blocks, i * rate_public_chain)

            # Break if prob_M_i becomes negligible
            if prob_M_i < negligible_threshold and prob_M_i < pr_M[k]:
                break

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

    # Calculate cumulative sums for L, B, and M
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
    print(finality_calc_validator(chain, e, f, c, s))

if __name__ == "__main__":
    main()