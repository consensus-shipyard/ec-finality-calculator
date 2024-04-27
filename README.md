# Expected Consensus Finality Calculator

Python demonstrator of an algorithm to determine the probability bound for replacement of a Filecoin tipset given chain history. 

The algorithm allows users to make decisions on transaction confirmation based on the actual content of the chain rather than on a fixed number of epochs. It can reduce the time to (soft) finality by one order of magnitude under normal operating conditions.

## Background

FRC-0089: https://github.com/filecoin-project/FIPs/blob/master/FRCs/frc-0089.md

## Requirements

Uses `numpy`, `scipy`, `pandas`, and `matplotlib`

## Content
 - `finality_calc_validator.py`: **the EC Finality Calculator** given full information available to nodes
 - `finality_calc_actor.py`: *experimental* calculator in more restrictive actor context
 - `experiments/`: evaluation code, data, and results

## Usage

* Run `finality_calc_validator.py` to try the calculator on generated data
* Run `experiments/evaluation.py` to try the calculator on simulation data and Filecoin chain traces
* Call `finality_calc_validator()` using your chosen data and parameters

## License

Dual-licensed under MIT + Apache 2.0
