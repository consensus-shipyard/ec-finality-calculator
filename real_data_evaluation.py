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
plotting_step = 100 # skip epochs in plotting; does not affect error plotting
datasets = ['march', 'november']

####################
# Helper function for generating plots
####################

def plot_err_prob_and_block_cnt(chain, errors, delay):
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

    # Plot the error probabilities
    ax1.plot(errors['Height'], errors['Error'], color='blue', marker='o', linestyle='-')
    ax1.set_xlabel('Height')
    ax1.set_ylabel('Error probability after 30 epochs', color='blue')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='blue')

    plt.title('Error probabilities and # blocks per tipset')
    plt.grid(True)
    return fig

####################
# Run through each dataset, estimate error for sampled points, and export csv and plot
####################

for dataset in datasets:
    print("Dataset: " + dataset)

    # Set paths
    data_path = f'./evaluation/data/{dataset}.csv'
    result_path = f'./evaluation/results/{dataset}.csv'
    figure_path = f'./evaluation/figures/{dataset}.png'

    # Load raw data
    df_chain = pd.read_csv(data_path)
    df_chain.fillna(0, inplace=True)
    df_chain = df_chain.convert_dtypes()

    # Compute probabilities
    sample_indices = range(history_length + 1, df_chain.shape[0] - history_length, sampling_step)
    pr_err = []
    for start_index in sample_indices:
        print("i: " + str(start_index))

        # Extract the subsequence
        end_index = start_index + history_length
        subchain = df_chain['block_counts'][start_index:end_index].to_numpy()

        # Height of evaluated tipset
        height = df_chain['height'][end_index - settlement_epochs]
        target_epoch = len(subchain) - 1 - settlement_epochs

        # Calculate error probability
        pr_err.append(vf.finality_calc_validator(subchain, blocks_per_epoch, byzantine_fraction, target_epoch + settlement_epochs, target_epoch))

    # Create dataframe and export to csv
    df_results = pd.DataFrame({'Height': df_chain['height'][sample_indices], 'Error': pr_err})
    df_results.to_csv(result_path, index=False)

    # Plot and export results
    fig = plot_err_prob_and_block_cnt(df_chain, df_results, settlement_epochs)
    fig.savefig(figure_path)

