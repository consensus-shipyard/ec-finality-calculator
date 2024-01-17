import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
 
import finality_calc_validator as vf

####################
# Parameters
####################

# Calculator
history_length = 900  # length of relevant history
blocks_per_epoch = 5  # expected number of blocks per epoch
byzantine_fraction = 0.3  # portion of adversary to tolerate

# Evaluation
settlement_epochs = 30  # number of delay (settlement) epochs
sampling_step = 1000 # skip step size for iteration
plotting_step = 200 # skip epochs in plotting; does not affect error plotting
datasets = ['march', 'november']

####################
# Helper function for generating plots
####################

def plot_err_prob_and_block_cnt(chain, errors, delay, block_limits=False, error_limits=False):
    # Calculate the moving average for the block count
    chain['Moving Average'] = chain['block_counts'].rolling(window=delay, min_periods=1).mean().shift(-(delay-1))

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
    ax1.plot(errors['Height'], errors['Error'], color='blue', marker='o', linestyle='-')
    ax1.set_xlabel('Height')
    ax1.set_ylabel('Error probability after 30 epochs', color='blue')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='blue')
    if error_limits:
        ax1.set_ylim(error_limits)

    plt.title('Error probabilities and # blocks per tipset')
    plt.grid(True)
    return fig

####################
# Run through each dataset, estimate error for sampled points, and export csv
####################

df_chain = dict()
df_results= dict()

for dataset in datasets:
    print("Dataset: " + dataset)

    # Set paths
    data_path = f'./evaluation/data/{dataset}.csv'
    result_path = f'./evaluation/results/{dataset}.csv'

    # Load raw data
    df_chain[dataset] = pd.read_csv(data_path)
    df_chain[dataset].fillna(0, inplace=True)
    df_chain[dataset] = df_chain[dataset].convert_dtypes()

    # Compute probabilities
    sample_indices = range(history_length + 1, df_chain[dataset].shape[0] - history_length, sampling_step)
    pr_err = []
    for start_index in sample_indices:
        print("i: " + str(start_index) + "/" + str(sample_indices[-1]))

        # Extract the subsequence
        end_index = start_index + history_length
        subchain = df_chain[dataset]['block_counts'][start_index:end_index].to_numpy()

        # Height of evaluated tipset
        height = df_chain[dataset]['height'][end_index - settlement_epochs]
        target_epoch = len(subchain) - 1 - settlement_epochs

        # Calculate error probability
        pr_err.append(vf.finality_calc_validator(subchain, blocks_per_epoch, byzantine_fraction, target_epoch + settlement_epochs, target_epoch))

    # Create dataframe and export to csv
    df_results[dataset] = pd.DataFrame({'Height': df_chain[dataset]['height'][sample_indices], 'Error': pr_err})
    df_results[dataset].to_csv(result_path, index=False)

####################
# Rerun through each dataset and generate plots with consistent axes
####################

# Find limits for plotting
block_count_min = min(df_chain[dataset]['block_counts'].min() for dataset in datasets) 
block_count_max = max(df_chain[dataset]['block_counts'].max() for dataset in datasets) 
error_min = min(df_results[dataset]['Error'].min() for dataset in datasets) 
error_max = max(df_results[dataset]['Error'].max() for dataset in datasets) 

for dataset in datasets:
    # Plot and export results
    figure_path = f'./evaluation/figures/{dataset}.png'
    fig = plot_err_prob_and_block_cnt(df_chain[dataset], df_results[dataset], settlement_epochs, (block_count_min, block_count_max), (error_min, error_max))
    fig.savefig(figure_path)
