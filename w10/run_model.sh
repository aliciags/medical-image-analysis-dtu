#!/bin/bash
#BSUB -q gpuv100
#BSUB -J train_seg_model_mia     # Job name
#BSUB -n 4                       # Number of cores
#BSUB -W 14:00                   # Wall-clock time (14 hours here)
#BSUB -R "rusage[mem=8GB]"       # Memory requirements (8 GB here)
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o test.out  # Standard output file
#BSUB -e test.err  # Standard error file

# Activate the environment
source /zhome/2b/8/212341/medical-image-analysis-dtu/w10/.venv/bin/activate

# Run the Python script
python3 exec.py > output.txt