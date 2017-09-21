#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --time 5:0
#SBATCH --qos bbshort
#SBATCH --mail-type ALL

set -e

module purge; module load bluebear
module load apps/python2/2.7.11
module load apps/tensorflow/1.3.0-python-2.7.11
module load apps/keras/2.0.6-python-2.7.11

pip install --user pandas, fancyimpute
