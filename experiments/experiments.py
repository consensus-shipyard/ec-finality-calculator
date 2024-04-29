import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
from multiprocessing import Pool

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import finality_calc_validator as vf
import finality_calc_actor as af


####################
# Parameters
####################

# Default calculator options
history_length = 900  # length of relevant history
blocks_per_epoch = 5  # expected number of blocks per epoch
byzantine_fraction = 0.3  # portion of adversary to tolerate
settlement_epochs = 30  # number of delay (settlement) epochs

# Processing options
sampling_step = 1000 # skip step size for iteration
process_count = 12 # number of concurrent processes to use

# Visualisation options
plotting_step = 200 # skip epochs in plotting; does not affect error plotting
averaging_window = 30 # window size for moving average

# Bundled dataset: Evaluation
evaluation_dataset = ['march', 'november']
evaluation_path = './experiments/evaluation'
evaluation_params = {
    "history_length": history_length,
    "blocks_per_epoch": blocks_per_epoch,
    "byzantine_fraction": byzantine_fraction,
    "settlement_epochs": settlement_epochs,
    "sampling_step": sampling_step,
    "path": evaluation_path,
    "dataset": evaluation_dataset
}

# Bundled dataset: Simulation
simulation_quality_range = range(80, 101, 2)
simulation_instance_range = range(0, 6)
simulation_settlement_range = [20, 30, 40] 
simulation_epoch_count = 40000 
simulation_dataset = [f"{quality}_{instance}" for quality in simulation_quality_range for instance in simulation_instance_range]
simulation_path = './experiments/simulation'
simulation_params = {
    "history_length": history_length,
    "blocks_per_epoch": blocks_per_epoch,
    "byzantine_fraction": byzantine_fraction,
    "sampling_step": sampling_step,
    "path": simulation_path,
    "dataset": simulation_dataset
}

####################
# Data generation
####################

# Generates simulated chain traces for given parameters and exports to csv
def generate_chain_history(quality_range, instance_range, epoch_count, blocks_per_epoch, path):
    for lambda_param in quality_range:
        for instance in instance_range:
            filename = f'{path}/data/{lambda_param}_{instance}.csv'
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['height', 'block_counts'])
                for height in range(epoch_count):
                    block_count = np.random.poisson(lambda_param / 100 * blocks_per_epoch) 
                    writer.writerow([height, block_count])


####################
# Data processing
####################

# Process the dataset in parallel
def process_dataset(history_length, blocks_per_epoch, byzantine_fraction, settlement_epochs, sampling_step, path, dataset):
    pool = Pool(processes=process_count)
    args = [(history_length, blocks_per_epoch, byzantine_fraction, settlement_epochs, sampling_step, path, table) for table in dataset]
    pool.starmap(process_table, args)

# Run the actor and validator error probability calculation for each table in the dataset
# Imports the raw data from csv files, and exports the processed data to csv files
def process_table(history_length, blocks_per_epoch, byzantine_fraction, settlement_epochs, sampling_step, path, dataset):
    # Set paths
    raw_path = f'{path}/data/{dataset}.csv'
    chain_path = f'{path}/results/{dataset}_chain.csv'
    result_path = f'{path}/results/{dataset}_error_{settlement_epochs}.csv'

    # Load and process raw data
    df_chain = pd.read_csv(raw_path)
    df_chain.fillna(0, inplace=True)
    df_chain = df_chain.convert_dtypes()
    df_chain.to_csv(chain_path, index=False)

    # Compute probabilities
    sample_indices = range(history_length + 1, df_chain.shape[0] - history_length, sampling_step)
    pr_err_v = []
    pr_err_a = []
    for start_index in sample_indices:
        print(dataset + ": " + str(start_index) + "/" + str(sample_indices[-1]))

        # Extract the subsequence
        end_index = start_index + history_length
        subchain = df_chain['block_counts'][start_index:end_index].to_numpy()

        # Height of evaluated tipset
        height = df_chain['height'][end_index - settlement_epochs]
        target_epoch = len(subchain) - 1 - settlement_epochs

        # Calculate error probability
        pr_err_v.append(vf.finality_calc_validator(subchain, blocks_per_epoch, byzantine_fraction, target_epoch + settlement_epochs, target_epoch))
        pr_err_a.append(af.finality_calc_actor(subchain, blocks_per_epoch, byzantine_fraction, target_epoch + settlement_epochs, target_epoch))

    # Create dataframe and export to csv
    df_results = pd.DataFrame({'Height': df_chain['height'][sample_indices], 'Error (Validator)': pr_err_v, 'Error (Actor)': pr_err_a})
        
    # Export to csv
    df_chain.to_csv(chain_path, index=False)
    df_results.to_csv(result_path, index=False)
    return None

####################
# Data visualisation
####################

# Processes dataset and generates error plots for each table
def generate_error_plots(settlement_epochs, plotting_step, path, dataset):

    df_chain = dict()
    df_results= dict()

    for table in dataset:
        print("Table: " + table)
        chain_path = f'{path}/results/{table}_chain.csv'
        result_path = f'{path}/results/{table}_error_{settlement_epochs}.csv'
        df_chain[table] = pd.read_csv(chain_path)
        df_results[table] = pd.read_csv(result_path)

    # Find limits for plotting
    block_count_min = 0
    block_count_max = max(df_chain[dataset]['block_counts'][::plotting_step].max() for dataset in dataset) * 1.1
    error_min_v = min(df_results[dataset]['Error (Validator)'].min() for dataset in dataset) 
    error_max_v = max(df_results[dataset]['Error (Validator)'].max() for dataset in dataset) 
    error_min_a = min(df_results[dataset]['Error (Actor)'].min() for dataset in dataset) 
    error_max_a = max(df_results[dataset]['Error (Actor)'].max() for dataset in dataset) 
    error_min = min(error_min_v, error_min_a) * 0.9
    error_max = max(error_max_v, error_max_a) * 1.1

    for table in dataset:
    # Plot and export results
        figure_path = f'{path}/figures/{table}_{settlement_epochs}.png'
        fig = plot_err_prob_and_block_cnt(df_chain[table], df_results[table], settlement_epochs, plotting_step, (block_count_min, block_count_max), (error_min, error_max))
        fig.savefig(figure_path)
        plt.close(fig)

# Plot the error probabilities vs epoch for a given table
def plot_err_prob_and_block_cnt(chain, errors, settlement_epochs, plotting_step, block_limits=False, error_limits=False):
    # Calculate the moving average for the block count
    chain['Moving Average'] = chain['block_counts'].rolling(window=averaging_window, min_periods=1).mean().shift(-(averaging_window-1))

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Format x-axis to show full numbers
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Plot the block counts
    ax2 = ax1.twinx()
    ax2.plot(chain['height'][::plotting_step], chain['block_counts'][::plotting_step], color='green', marker='x', linestyle='')
    ax2.set_ylabel('# blocks at tipset', color='green')
    ax2.plot(chain['height'][::plotting_step], chain['Moving Average'][::plotting_step], color='green', marker='', linestyle='-', label='30-slot moving average')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')
    if block_limits:
        ax2.set_ylim(block_limits)        

    # Plot the error probabilities
    ax1.plot(errors['Height'], errors['Error (Validator)'], color='blue', marker='o', linestyle='-')
    ax1.plot(errors['Height'], errors['Error (Actor)'], color='purple', marker='o', linestyle='-')
    ax1.set_xlabel('Height')
    ax1.set_ylabel(f'Error probability after {settlement_epochs} epochs', color='blue')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(['Validator', 'Actor'], loc='upper left')
    if error_limits:
        ax1.set_ylim(error_limits)

    plt.title('Error probabilities and # blocks per tipset')
    plt.grid(True)
    return fig

# Processes dataset and generates scatter plot over the different fill rates
def generate_scatter_plots(path, dataset, settlement_epochs):
    df_results = dict()
    for table in dataset:
        result_path = f'{path}/results/{table}_error_{settlement_epochs}.csv'
        df_results[table] = pd.read_csv(result_path)
        print(table + "/min: " + str(min(df_results[table]['Error (Validator)'])))
        print(table + "/mean: " + str(np.mean(df_results[table]['Error (Validator)'])))
        print(table + "/0: " + str(df_results[table]['Error (Validator)'][0]))
    
    error_min_v = min(df_results[dataset]['Error (Validator)'].min() for dataset in dataset) 
    error_max_v = max(df_results[dataset]['Error (Validator)'].max() for dataset in dataset) 
    error_min_a = min(df_results[dataset]['Error (Actor)'].min() for dataset in dataset) 
    error_max_a = max(df_results[dataset]['Error (Actor)'].max() for dataset in dataset) 
    error_min = min(error_min_v, error_min_a) * 0.8
    error_max = max(error_max_v, error_max_a) * 1.2
    error_limits = (error_min, error_max)

    x_values = np.array([float(table.split('_')[0])/100 for table in dataset])
 
    y_values = np.array([df_results[table]['Error (Validator)'][0] for table in dataset])
    plot_scatter(x_values, y_values, settlement_epochs).savefig(f'{path}/figures/scatter_validator_{settlement_epochs}_sample.png')

    y_values = np.array([df_results[table]['Error (Actor)'][0] for table in dataset])
    plot_scatter(x_values, y_values, settlement_epochs).savefig(f'{path}/figures/scatter_actor_{settlement_epochs}_sample.png')    

# Scatter plot for the given x and y values
def plot_scatter(x_values, y_values, settlement_epochs=30, error_limits=False):
    fig = plt.figure()

    slope, intercept = np.polyfit(x_values, np.log(y_values), 1)  # Using log of y for linear fit if y is on log scale
    trendline = np.exp(intercept + slope * x_values)

    plt.scatter(x_values, y_values, alpha=0.7)
    plt.plot(x_values, trendline, color='red', label='Trend Line')
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('Chain fill rate')
    plt.ylabel('Error Probability')
    plt.title(f'Finality values for different fill-rates after {settlement_epochs} epochs')
    plt.legend()
    if error_limits:
        plt.ylim(error_limits)
    plt.grid(True)
    return fig

####################
# If run from console, process the bundled datasets
####################

if __name__ == "__main__":
    # Simulation
    generate_chain_history(simulation_quality_range, simulation_instance_range, simulation_epoch_count, blocks_per_epoch, simulation_path)
    for settlement_epochs in simulation_settlement_range:
       process_dataset(**simulation_params, settlement_epochs=settlement_epochs)
    for settlement_epochs in simulation_settlement_range:
       generate_error_plots(settlement_epochs, plotting_step, simulation_params['path'], simulation_params['dataset'])
       generate_scatter_plots(simulation_params['path'], simulation_params['dataset'], settlement_epochs)

    # Evaluation
    process_dataset(**evaluation_params)
    generate_error_plots(evaluation_params['settlement_epochs'], plotting_step, evaluation_params['path'], evaluation_params['dataset'])