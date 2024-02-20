# Expected Consensus Finality Calculator

Python demonstrator of an algorithm to determine the probability bound for replacement of a Filecoin tipset given chain history. 

The algorithm allows users to make decisions on transaction confirmation based on the actual content of the chain rather than on a fixed number of epochs. It can reduce the time to (soft) finality by one order of magnitude under normal operating conditions.

## Background

FRC-0089: https://github.com/filecoin-project/FIPs/pull/941

(to be replaced with link to final FRC)

## Requirements

Uses `numpy`, `scipy`, `pandas`, and `matplotlib`

## Content
 - `finality_calc_validator.py`: **the EC Finality Calculator** given full information available to nodes
 - `real_data_evaluation.py`: evaluation script that runs the EC finality calculator on chain data
 - `evaluation/`: evaluation data and results using default settings
 - `finality_calc_actor.py`: *experimental* calculator in more restrictive actor context

## Usage

* Run `finality_calc_validator.py` to try the calculator on generated data
* Run `real_data_evaluation.py` to try the calculator on Filecoin chain traces
* Call `finality_calc_validator()` using your chosen data and parameters

## License

Dual-licensed under MIT + Apache 2.0
