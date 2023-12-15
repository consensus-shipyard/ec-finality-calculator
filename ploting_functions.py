import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_results_with_trendline(file_path):
    with open(file_path, 'r') as file:
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


def plot_error_probability(file_path1, file_path2=None):
    """
    Plots Error Probability vs Height from one or two given CSV files on the same graph.

    Parameters:
    file_path1 (str): Path to the first CSV file containing 'Height' and 'Error Probability' columns.
    file_path2 (str, optional): Path to the second CSV file containing 'Height' and 'Error Probability' columns. Default is None.
    """
    # Read the first CSV file
    df1 = pd.read_csv(file_path1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df1['Height'], df1['Error Probability'], marker='o', linestyle='-', label='30 deep')

    # If a second file path is provided, plot its data
    if file_path2 is not None:
        df2 = pd.read_csv(file_path2)
        plt.plot(df2['Height'], df2['Error Probability'], marker='x', linestyle='--', label='20 deep')

    plt.xlabel('Height')
    plt.ylabel('Error Probability')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('Error Probability vs Height')
    plt.legend()
    plt.grid(True)
    plt.show()

# def plot_error_probability(file_path):
#     """
#     Plots Error Probability vs Height from a given CSV file.
#
#     Parameters:
#     file_path (str): Path to the CSV file containing 'Height' and 'Error Probability' columns.
#     """
#     # Read the CSV file
#     df = pd.read_csv(file_path)
#
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(df['Height'], df['Error Probability'], marker='o', linestyle='-')
#     plt.xlabel('Height')
#     plt.ylabel('Error Probability')
#     plt.yscale('log')  # Set y-axis to logarithmic scale
#     plt.title('Error Probability vs Height')
#     plt.grid(True)
#     plt.show()


# Plot the results with trend line
# file_path = r".\individual_simulation_results.csv"
# plot_results_with_trendline(file_path)

# Plot the error probabilities vs. tipset heights
file_path1 = r".\Evaluation_results\results_files\evaluation_of_results_mar_from_0_to_2000_depth_30.csv"
file_path2 = r".\Evaluation_results\results_files\evaluation_of_results_mar_from_0_to_2000_depth_20.csv"
plot_error_probability(file_path1, file_path2)
