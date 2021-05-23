#!/bin/bash

# run training with rendering enabled
# also performs cleanup
./clean.sh
conda activate drlnd
python3 main.py
#xvfb-run -s "-screen 0 600x400x24" ~/anaconda3/envs/drlnd/bin/python3 main.py

echo "execute ./run_tensorboard.sh to view results"
