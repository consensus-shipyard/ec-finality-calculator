import numpy as np
from scipy.stats import poisson

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

## Step 2: account for malicious blocks
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

## Step 3
# calculate the probability of BpZ = B + Z base on the joint distribution of (B,Z | chain)
def probability_of_BpZ_given_chain(chain, start_epoch, end_epoch, e, f, max_z, max_b):
    max_BpZ = max_b+max_z
    values_of_BpZ = np.arange(0, max_BpZ+1)
    probabilities_BpZ = np.zeros(max_b + max_z + 1)

    num_epochs = (end_epoch - start_epoch)
    lambda_T = e * num_epochs
    lambda_H = (1-f) * lambda_T
    num_of_observed_blocks = sum(chain[start_epoch:end_epoch])

    for z in range(max_z + 1):
        pr_of_z_given_chain = probability_of_Z_given_chain(lambda_T, z, num_of_observed_blocks)
        for b in range(max_b + 1):
            b_p_z = b + z
            joint_prob = probability_of_B_given_T(lambda_H, b, z + num_of_observed_blocks) * pr_of_z_given_chain
            probabilities_BpZ[b_p_z] += joint_prob
    return [values_of_BpZ, probabilities_BpZ]



def find_closest(array, target_val):
    # Calculate the absolute difference with the target value
    differences = np.abs(array - target_val)

    # Find the index of the smallest difference
    index_of_closest = np.argmin(differences)

    # Return the closest value
    return index_of_closest