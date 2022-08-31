#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-footprint-extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --dependency=afterany:61162958

cd ../

singularity exec $GROUP_HOME/singularity/rgb-building1.sif python3 superres_helper.py --partition=1 \
	--oak-fp=/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/ --year=2018