# Expected Consensus Finality Calculator

Python demonstrator of an algorithm to determine the probability bound for replacement of a Filecoin tipset given chain history.

The algorithm allows users to make decisions on transaction confirmation based on the actual content of the chain rather than on a fixed number of epochs. It can reduce the time to (soft) finality by one order of magnitude under normal operating conditions.

## Background

Link to specification forthcoming.

## Status

Work in progress. This provides the prototype implementation for an upcoming Filecoin Improvement Proposal (FIP).

## Requirements

This project uses numpy, scipy, pandas, and matplotlib.

## Content
 - `validator_calc_finality.py`: **the EC Finality Calculator**
 - `read_data_evaluation.py`: evaluation script that runs the EC finality calculator on chain data
 - `evaluation`: evaluation data and results using default settings
 - `actor_calc_finality.py`: *experimental* calculator in more restrictive actor context

## Usage

* Run `validator_calc_finality.py` to try the calculator on generated data
* Run `real_data_evaluation.py` to try the calculator on Filecoin chain traces
* Call `validator_calc_finality()` using your chosen data and parameters

## License

Dual-licensed under MIT + Apache 2.0