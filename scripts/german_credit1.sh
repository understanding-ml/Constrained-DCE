#!/bin/bash
#BSUB -q hpc
#BSUB -J german_credit1
#BSUB -n 4
#BSUB -W 4:00
#BSUB -R "rusage[mem=10GB]"
#BSUB -oo data/german_credit/logs/std.out
#BSUB -eo data/german_credit/logs/std.err

# 加载环境
module load python3/3.10.16
module load cuda/11.6

# 设置相对路径，不写死任何 /zhome/...
READ_DATA_PATH="$PWD/data/german_credit"
WRITE_DATA_PATH="$PWD/data/german_credit"
export READ_DATA_PATH WRITE_DATA_PATH

echo "Using READ_DATA_PATH=$READ_DATA_PATH"
echo "Using WRITE_DATA_PATH=$WRITE_DATA_PATH"
ls -l "$READ_DATA_PATH"

# 注意调用的是 german_credit1.py
python3 -m experiments.german_credit1
