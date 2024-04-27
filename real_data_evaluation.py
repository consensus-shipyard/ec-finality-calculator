import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from multiprocessing import Pool


import finality_calc_validator as vf
import finality_calc_actor as af
import helper_functions as hf
import numpy as np

####################
# Parameters
####################

# Calculator options
history_length = 900  # length of relevant history
blocks_per_epoch = 5  # expected number of blocks per epoch
byzantine_fraction = 0.3  # portion of adversary to tolerate
settlement_epochs = 30  # number of delay (settlement) epochs

# Processing options
process_count = 12 # number of concurrent processes to use

# Visualisation options
sampling_step = 1000 # skip step size for iteration
plotting_step = 200 # skip epochs in plotting; does not affect error plotting

# Bundled dataset: Evaluation
dataset_evaluation = ['march', 'november']
path_evaluation = './evaluation'

# Bundled dataset: Simulation
quality_range = range(80, 101, 2)
instance_range = range(0, 6)
epoch_count = 80000 
dataset_simulation = [f"{quality}_{instance}" for quality in quality_range for instance in instance_range]
path_simulation = './simulation'


####################
# Process the dataset in parallel
####################
def process_dataset(history_length, blocks_per_epoch, byzantine_fraction, settlement_epochs, sampling_step, path, dataset):
    pool = Pool(processes=process_count)
    args = [(history_length, blocks_per_epoch, byzantine_fraction, settlement_epochs, sampling_step, path, table) for table in dataset]
    pool.starmap(compute_error, args)

####################
# Run the actor and validator error probability calculation for each table in the dataset
# Imports the raw data from csv files, and exports the processed data to csv files
####################
def compute_error(history_length, blocks_per_epoch, byzantine_fraction, settlement_epochs, sampling_step, path, dataset):
    # Set paths
    raw_path = f'{path}/data/{dataset}.csv'
    chain_path = f'{path}/results/{dataset}_chain.csv'
    result_path = f'{path}/results/{dataset}_error.csv'

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
# Generates simulated chain traces for given parameters and exports to csv
####################
def generate_chain_history(quality_range, instance_range, epoch_count, blocks_per_epoch):
    for lambda_param in quality_range:
        for instance in instance_range:
            filename = f'./simulation/data/{lambda_param}_{instance}.csv'
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['height', 'block_counts'])
                for height in range(epoch_count):
                    block_count = np.random.poisson(lambda_param / 100 * blocks_per_epoch) 
                    writer.writerow([height, block_count])

####################
# Import all tables in the dataset and plot error over time for each
####################
def generate_error_plots2(settlement_epochs, plotting_step, path, dataset):

    df_chain = dict()
    df_results= dict()

    for table in dataset:
        print("Table: " + table)
        chain_path = f'{path}/results/{table}_chain.csv'
        result_path = f'{path}/results/{table}_error.csv'
        df_chain[table] = pd.read_csv(chain_path)
        df_results[table] = pd.read_csv(result_path)

    # Find limits for plotting
    block_count_min = min(df_chain[dataset]['block_counts'].min() for dataset in dataset) * 0.9
    block_count_max = max(df_chain[dataset]['block_counts'].max() for dataset in dataset) * 1.1
    error_min_v = min(df_results[dataset]['Error (Validator)'].min() for dataset in dataset) 
    error_max_v = max(df_results[dataset]['Error (Validator)'].max() for dataset in dataset) 
    error_min_a = min(df_results[dataset]['Error (Actor)'].min() for dataset in dataset) 
    error_max_a = max(df_results[dataset]['Error (Actor)'].max() for dataset in dataset) 
    error_min = min(error_min_v, error_min_a) * 0.9
    error_max = max(error_max_v, error_max_a) * 1.1

    for table in dataset:
    # Plot and export results
        figure_path = f'{path}/figures/{table}.png'
        fig = hf.plot_err_prob_and_block_cnt2(df_chain[table], df_results[table], settlement_epochs, plotting_step, (block_count_min, block_count_max), (error_min, error_max))
        fig.savefig(figure_path)
        plt.close(fig)

# Generates a scatter plot where x entries are the elements of dataset and y entries are the first entry of the corresponding df_results
def generate_scatter_plots(path, dataset):
    df_results = dict()
    for table in dataset:
        result_path = f'{path}/results/{table}_error.csv'
        df_results[table] = pd.read_csv(result_path)
    x_values = np.array([float(table.split('_')[0])/100 for table in dataset])

    y_values = np.array([df_results[table]['Error (Validator)'][0] for table in dataset])
    fig = scatter_plot(x_values, y_values)    
    fig.savefig(f'{path}/figures/scatter_validator.png')

    for table in dataset:
        print(table + ": " + str(min(df_results[table]['Error (Validator)'])))

    y_values = np.array([df_results[table]['Error (Actor)'][0] for table in dataset])
    fig = scatter_plot(x_values, y_values)    
    fig.savefig(f'{path}/figures/scatter_actor.png')

def scatter_plot(x_values, y_values):
    fig = plt.figure()

    slope, intercept = np.polyfit(x_values, np.log(y_values), 1)  # Using log of y for linear fit if y is on log scale
    trendline = np.exp(intercept + slope * x_values)    

    plt.scatter(x_values, y_values, alpha=0.7)
    plt.plot(x_values, trendline, color='red', label='Trend Line')
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('Chain fill rate')
    plt.ylabel('Error Probability')
    plt.title('Finality values for different fill-rates after 30 epochs')
    plt.legend()
    plt.grid(True)
    return fig

####################
# If run from console, process the bundled datasets
####################
if __name__ == "__main__":
    # Simulation
    #generate_chain_history(quality_range, instance_range, epoch_count, blocks_per_epoch)
    process_dataset(history_length, blocks_per_epoch, byzantine_fraction, settlement_epochs, sampling_step, path_simulation, dataset_simulation)
    generate_error_plots2(settlement_epochs, plotting_step, path_simulation, dataset_simulation)
    generate_scatter_plots(path_simulation, dataset_simulation)

    # Evaluation
    process_dataset(history_length, blocks_per_epoch, byzantine_fraction, settlement_epochs, sampling_step, path_evaluation, dataset_evaluation)
    generate_error_plots2(settlement_epochs, plotting_step, path_evaluation, dataset_evaluation)