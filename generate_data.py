import csv
import numpy as np

quality_range = range(50, 101, 10) # Range for quality parameter
instance_range = range(0, 6) # Number of instances for each parameter
length = 10000 # How many epochs to simulate
blocks_per_epoch = 5  # Expected number of blocks per epoch

# Generate chain simulations for each quality parameter
for lambda_param in quality_range:
    for instance in instance_range:
        filename = f'./simulation/data/{lambda_param}_{instance}.csv'
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['height', 'block_counts'])
            for height in range(length):
                block_count = np.random.poisson(lambda_param / 100 * blocks_per_epoch) 
                writer.writerow([height, block_count])