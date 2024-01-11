import numpy as np
from scipy.stats import poisson
from scipy.stats import skellam


# Define parameters
import helper_functions
import validator_calc_finality as vcf

e = 5 # Expected number of blocks per epoch
f = 0.3 # portion of malicious power
num_variables = 905 # length of history
chain_health = 4.5/5 # mean precentage of blocks in an epoch compared to the expectation from a perfect network
c = 904 # current position (end of history)
s = c - 30 # slot in history for which finality is calculated
negligible_threshold = 10 ** -15

rng = np.random.default_rng(0)
chain = rng.poisson(chain_health * e, num_variables)

print(vcf.validator_calc_finality(chain, e, f, c, s, negligible_threshold))
