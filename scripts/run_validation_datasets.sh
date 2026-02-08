#!/bin/bash
#BSUB -q gpuv100              
#BSUB -gpu "num=1"             
#BSUB -n 4                     
#BSUB -R "rusage[mem=4096]"     
#BSUB -W 6:00                  
#BSUB -o output_%J.log         

source ~/myenv/bin/activate

# python HELOC_new.py
# python cardio_new.py
python market_new.py



