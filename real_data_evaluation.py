import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
 
import finality_calc_validator as vf
import finality_calc_actor as af
import helper_functions as hf

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
    pr_err_v = []
    pr_err_a = []
    for start_index in sample_indices:
        print("i: " + str(start_index) + "/" + str(sample_indices[-1]))

        # Extract the subsequence
        end_index = start_index + history_length
        subchain = df_chain[dataset]['block_counts'][start_index:end_index].to_numpy()

        # Height of evaluated tipset
        height = df_chain[dataset]['height'][end_index - settlement_epochs]
        target_epoch = len(subchain) - 1 - settlement_epochs

        # Calculate error probability
        pr_err_v.append(vf.finality_calc_validator(subchain, blocks_per_epoch, byzantine_fraction, target_epoch + settlement_epochs, target_epoch))
        pr_err_a.append(af.finality_calc_actor(subchain, blocks_per_epoch, byzantine_fraction, target_epoch + settlement_epochs, target_epoch))

    # Create dataframe and export to csv
    df_results[dataset] = pd.DataFrame({'Height': df_chain[dataset]['height'][sample_indices], 'Error (Validator)': pr_err_v, 'Error (Actor)': pr_err_a})
    df_results[dataset].to_csv(result_path, index=False)

####################
# Rerun through each dataset and generate plots with consistent axes
####################

# Find limits for plotting
block_count_min = min(df_chain[dataset]['block_counts'].min() for dataset in datasets) 
block_count_max = max(df_chain[dataset]['block_counts'].max() for dataset in datasets) 
error_min_v = min(df_results[dataset]['Error (Validator)'].min() for dataset in datasets) 
error_max_v = max(df_results[dataset]['Error (Validator)'].max() for dataset in datasets) 
error_min_a = min(df_results[dataset]['Error (Actor)'].min() for dataset in datasets) 
error_max_a = max(df_results[dataset]['Error (Actor)'].max() for dataset in datasets) 
error_min = min(error_min_v, error_min_a)
error_max = max(error_max_v, error_max_a)

for dataset in datasets:
    # Plot and export results
    figure_path = f'./evaluation/figures/{dataset}.png'
    fig = hf.plot_err_prob_and_block_cnt2(df_chain[dataset], df_results[dataset], settlement_epochs, plotting_step, (block_count_min, block_count_max), (error_min, error_max))
    fig.savefig(figure_path)
    plt.close(fig)
