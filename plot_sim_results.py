import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_results_with_trendline(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        x_values, y_values = [], []
        for row in reader:
            x_values.append(float(row[0]))  # Assuming param1 is convertible to float
            y_values.append(float(row[1]))

    # Convert lists to numpy arrays for linear regression
    x_array = np.array(x_values)
    y_array = np.array(y_values)

    # Linear regression for trend line
    slope, intercept = np.polyfit(x_array, np.log(y_array), 1)  # Using log of y for linear fit if y is on log scale
    trendline = np.exp(intercept + slope * x_array)

    plt.scatter(x_values, y_values, alpha=0.7)  # alpha for better visibility if points overlap
    plt.plot(x_array, trendline, color='red', label='Trend Line')  # Add trend line to the plot
    plt.xscale('linear')  # or 'log' if param1 should be on a logarithmic scale
    plt.yscale('log')
    plt.xlabel('Chain fill rate')
    plt.ylabel('Error Probability')
    plt.title('Finality values for different fill-rates after 30 epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the results with trend line
plot_results_with_trendline('individual_simulation_results.csv')

