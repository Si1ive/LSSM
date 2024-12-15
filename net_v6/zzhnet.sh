#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1

1120241577
module load anaconda/anaconda3-2022.10
module load cuda/11.8.0

source activate zzhnet

python train.py |tee train.out
