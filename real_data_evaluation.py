import pandas as pd
import numpy as np
import validator_calc_finality as vf

# evaluation function to be applied to each entry
def evaluate_finality_of_s_after_dd_epochs(sub_chain, s, dd=30):
    e = 5  # expected number of blocks per epoch
    f = 0.3  # portion of adversary to tolerate
    c = s + dd  # position of evaluation (dd epochs after the tipset of interest)

    err_prob = vf.validator_calc_finality(sub_chain, e, f, c, s)
    return err_prob



# Load the dataset
# mar = r'C:\Users\sgore\Downloads\blocks_count_from_march.csv' # Replace with your actual file path
nov = r".\Evaluation_results\raw_data\orphan_block_count_november.csv"
df = pd.read_csv(nov)
df.fillna(0, inplace=True)
df = df.convert_dtypes()

# Parameters
subseq_length = 904  # (maximal) length of relevant history
chunk_size = 1500  # Number of entries to process before saving to a new file
dd = 30  # number of delay (settlement) epochs
data_name = "nov"  # where is the data coming from


# Iterate over the dataset and process in chunks
for chunk_start in range(6500 - subseq_length, len(df), chunk_size):
    output_file = f"./Evaluation_results/results_files/evaluation_of_results_{data_name}_chunk_{chunk_start // chunk_size + 1}_depth_{str(dd)}.csv"
    with open(output_file, 'w') as file:
        # Write header
        file.write('Height,Error Probability\n')
        for start_index in range(chunk_start, min(chunk_start + chunk_size, len(df) - subseq_length + 1)):
            # Extract the sub-sequence
            end_index = start_index + subseq_length
            subsequence = df['block_counts'][start_index:end_index]
            # Height of evaluated tipset
            height = df['height'][end_index - dd]
            sub_chain = subsequence.to_numpy()
            s = len(sub_chain) - 1 - dd
            # Calculate error probability
            err_prob = evaluate_finality_of_s_after_dd_epochs(sub_chain, s, dd)
            file.write(f'{height},{err_prob}\n')
            print("iteration at index: " + str(start_index))

        print(f"Evaluation of chunk {chunk_start // chunk_size + 1} complete and results saved.")
print('Simulation complete and results saved in chunks.')
