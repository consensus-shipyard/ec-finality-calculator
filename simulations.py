import matplotlib.pyplot as plt
import validator_calc_finality as vf
import csv

# Define your simulation function here
def run_simulation(chain_health, depth):
    # Your simulation code
    result = vf.validator_calc_finality(5, 0.3, 905, chain_health, 904, 904-depth)
    return result

# Define your parameter sets
parameter_sets = [
    {'chain_health': 3.5/5, 'depth': 30},  # Set 1
    {'chain_health': 3.6/5, 'depth': 30},
    {'chain_health': 3.7/5, 'depth': 30},
    {'chain_health': 3.8/5, 'depth': 30},
    {'chain_health': 3.9/5, 'depth': 30},  # Set 5
    {'chain_health': 4.0/5, 'depth': 30},
    {'chain_health': 4.1/5, 'depth': 30},
    {'chain_health': 4.2/5, 'depth': 30},
    {'chain_health': 4.3/5, 'depth': 30},
    {'chain_health': 4.4 / 5, 'depth': 30},  # Set 10
    {'chain_health': 4.5 / 5, 'depth': 30},
    {'chain_health': 4.6 / 5, 'depth': 30},
    {'chain_health': 4.7 / 5, 'depth': 30},
    {'chain_health': 4.8 / 5, 'depth': 30},
    {'chain_health': 4.9 / 5, 'depth': 30},
    {'chain_health': 5.0/5, 'depth': 30}  # Set 16
]

# Run simulation and save individual results
with open('individual_simulation_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['chain_health', 'Result'])
    for params in parameter_sets:
        print(params)
        for _ in range(7):
            result = run_simulation(params['chain_health'], params['depth'])
            writer.writerow([params['chain_health'], result])

# Plotting the results directly from saved data
def plot_results(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        x_values, y_values = [], []
        for row in reader:
            x_values.append(float(row[0]))  # Assuming param1 is convertible to float
            y_values.append(float(row[1]))

    plt.scatter(x_values, y_values, alpha=0.7)  # alpha for better visibility if points overlap
    plt.xscale('linear')  # or 'log' if param1 should be on a logarithmic scale
    plt.yscale('log')
    plt.xlabel('param1')
    plt.ylabel('Simulation Result')
    plt.title('Simulation Results')
    plt.grid(True)
    plt.show()

# Plot the results
plot_results('individual_simulation_results.csv')