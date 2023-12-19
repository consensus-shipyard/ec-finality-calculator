import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def plot_err_prob_and_blck_cnt(file_path_for_blck_cnt, file_path_for_err_prob, delay):
    # Load the data from the evaluation results file
    # file_path_for_err_prob = r".\Evaluation_results\results_files\evaluation_of_results_mar_chunk_9_depth_30.csv"
    df1 = pd.read_csv(file_path_for_err_prob)  # Replace with your first file path
    # Load the data from the block count file
    df2 = pd.read_csv(file_path_for_blck_cnt) # Replace with your second file path

    # # Shortening the data
    # midpoint = len(df1) // 2
    # df1 = df1.iloc[:1000]

    # Calculate the moving average for the block count
    window_size = delay
    # df2['Moving Average'] = df2['block_counts'].rolling(window=window_size).mean()
    df2['Moving Average'] = df2['block_counts'].rolling(window=window_size, min_periods=1).mean().shift(-(window_size-1))


    # Define the height range you're interested in
    min_height = df1['Height'].iloc[0]  # Replace with your minimum height
    max_height = df1['Height'].iloc[-1]  # Replace with your maximum height

    # Filter the DataFrame for the height range
    df2 = df2[(df2['height'] >= min_height) & (df2['height'] <= max_height)]

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Format x-axis to show full numbers
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Plot the first dataset with ax1 for the y-axis on the left
    ax1.plot(df1['Height'], df1['Error Probability'], color='blue', marker='o', linestyle='-')
    ax1.set_xlabel('Height')
    ax1.set_ylabel('Error Probability after 30 epochs delay', color='blue')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for the second dataset
    ax2 = ax1.twinx()
    ax2.plot(df2['height'], df2['block_counts'], color='green', marker='x', linestyle='--')
    ax2.set_ylabel('# of blocks at tipset', color='green')
    ax2.plot(df2['height'], df2['Moving Average'], color='red', marker='', linestyle='-', label='30-Slot Moving Average')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    plt.title('Error Probabilities and # blocks per tipset')
    plt.grid(True)
    plt.show()



# Usage:

# Plot the results with trend line
# file_path = r".\individual_simulation_results.csv"
# plot_results_with_trendline(file_path)

# Plot the error probabilities vs. tipset heights
# file_path1 = r".\Evaluation_results\results_files\evaluation_of_results_mar_from_0_to_2000_depth_30.csv"
# file_path2 = r".\Evaluation_results\results_files\evaluation_of_results_mar_from_0_to_2000_depth_20.csv"
# file_path1 = r".\Evaluation_results\results_files\evaluation_of_results_mar_chunk_9_depth_30.csv"
# plot_error_probability(file_path1) #, file_path2)

# Plot the error probabilities, tipset block-count and a moving average (delay) vs. tipset heights
# file_path_err_prob = r".\Evaluation_results\results_files\evaluation_of_results_mar_chunk_53_depth_30.csv"
# file_path_blck_cnt = r".\Evaluation_results\raw_data\blocks_count_from_march.csv"
file_path_err_prob = r".\Evaluation_results\results_files\evaluation_of_results_nov_chunk_4_depth_30.csv"
file_path_blck_cnt = r".\Evaluation_results\raw_data\orphan_block_count_november.csv"
delay = 30
plot_err_prob_and_blck_cnt(file_path_blck_cnt, file_path_err_prob, delay)


# # Load the data from the first file
# # file1 = r".\Evaluation_results\results_files\evaluation_of_results_mar_chunk_9_depth_30.csv"
# file1 = r".\Evaluation_results\results_files\evaluation_of_results_mar_chunk_53_depth_30.csv"
# df1 = pd.read_csv(file1)  # Replace with your first file path
# # Shortening the data
# # midpoint = len(df1) // 2
# # df1 = df1.iloc[:1000]
# # Load the data from the second file
# file2 = r".\Evaluation_results\raw_data\blocks_count_from_march.csv"
# df2 = pd.read_csv(file2) # Replace with your second file path
#
# # Calculate the moving average for the block count
# window_size = 30
# # df2['Moving Average'] = df2['block_counts'].rolling(window=window_size).mean()
# df2['Moving Average'] = df2['block_counts'].rolling(window=window_size, min_periods=1).mean().shift(-(window_size-1))
#
#
# # Define the height range you're interested in
# min_height = df1['Height'].iloc[0]  # Replace with your minimum height
# max_height = df1['Height'].iloc[-1]  # Replace with your maximum height
#
# # Filter the DataFrame for the height range
# df2 = df2[(df2['height'] >= min_height) & (df2['height'] <= max_height)]
#
# # Create the plot
# fig, ax1 = plt.subplots(figsize=(10, 6))
#
# # Format x-axis to show full numbers
# ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
#
# # Plot the first dataset with ax1 for the y-axis on the left
# ax1.plot(df1['Height'], df1['Error Probability'], color='blue', marker='o', linestyle='-')
# ax1.set_xlabel('Height')
# ax1.set_ylabel('Error Probability after 30 epochs delay', color='blue')
# ax1.set_yscale('log')
# ax1.tick_params(axis='y', labelcolor='blue')
#
# # Create a second y-axis for the second dataset
# ax2 = ax1.twinx()
# ax2.plot(df2['height'], df2['block_counts'], color='green', marker='x', linestyle='--')
# ax2.set_ylabel('# of blocks at tipset', color='green')
# ax2.plot(df2['height'], df2['Moving Average'], color='red', marker='', linestyle='-', label='30-Slot Moving Average')
# ax2.tick_params(axis='y', labelcolor='green')
# ax2.legend(loc='upper right')
#
# plt.title('Error Probabilities and # blocks per tipset')
# plt.grid(True)
# plt.show()







#
# # Load the data
# data_file = r".\Evaluation_results\raw_data\blocks_count_from_march.csv"
# df = pd.read_csv(data_file)  # Replace with your file path
#
# # Define the height range you're interested in
# min_height = 2710586  # Replace with your minimum height
# max_height = 2713608  # Replace with your maximum height
#
# # Filter the DataFrame for the height range
# filtered_df = df[(df['height'] >= min_height) & (df['height'] <= max_height)]
#
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(filtered_df['height'], filtered_df['block_counts'], marker='o', linestyle='-')
# plt.xlabel('Height')
# plt.ylabel('Block count')
# plt.title('# blocks at Height')
# plt.grid(True)
# plt.show()
