#!/bin/bash

#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

inv validate --preset=release --target=cusparse2
inv benchmark --target=cusparse2 --results-dir=results/fit-cluster
