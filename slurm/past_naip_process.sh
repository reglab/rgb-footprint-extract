#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-footprint-extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB

cd ../

singularity exec $GROUP_HOME/singularity/rgb-building1.sif python3 superres_helper.py \
	--oak-fp=/oak/stanford/groups/deho/building_compliance/ --year=2016