import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

####################
# Helper function for generating plots
####################

def plot_err_prob_and_block_cnt(chain, errors, delay, plotting_step=200, block_limits=False, error_limits=False):
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