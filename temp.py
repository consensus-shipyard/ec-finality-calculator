import numpy as np
import helper_functions as hf

# Define parameters
e = 5
f = 0.3
num_variables = 905 # length of history
c = 904 # current position (end of history)
s = c - 20 # slot in history for which finality is calculated

# Generate the chain of Poisson random variables
chain = np.random.poisson(0.9*e, num_variables)


values_of_BpZ = []
probabilities_BpZ = []
for i in range(5):
    # Define the values of Z and B for which we wish to calculate
    max_z = e * 5 * (i+1) + 20  # Adjust the range as needed
    max_b = e * 5 * (i+1) + 20  # Adjust the range as needed

    # Define the section of interest
    start_epoch = s - 10 * i - 1
    end_epoch = s

    [tmp_values_of_BpZ, tmp_probabilities_BpZ] = hf.probability_of_BpZ_given_chain(chain, start_epoch, end_epoch, e, f, max_z, max_b)
    values_of_BpZ.append(tmp_values_of_BpZ)
    probabilities_BpZ.append(tmp_probabilities_BpZ)
    print(i)

## Plot the probabilities
# Plot the probability of BpZ
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Example lists with 10 arrays

# Create 10 subplots
fig, axs = plt.subplots(1, 5, figsize=(15, 6))
fig.tight_layout()

for i in range(5):
    ax = axs[i]

    # Plot probabilities_BpZ[i] as a function of values_of_BpZ[i]
    ax.plot(values_of_BpZ[i], probabilities_BpZ[i])
    ax.set_title(f'Plot {i + 1}')

plt.show()
