#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus 1
1120241486
module load anaconda3/envs
module load cuda/11.6.0

source activate dsca
python train.py --cfg configs/cuhk_sysu_da.yaml
