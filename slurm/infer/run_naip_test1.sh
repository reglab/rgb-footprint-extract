#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-footprint-extract
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --time=00:05:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

cd ../../

singularity exec --nv $GROUP_HOME/singularity/rgb-building1.sif python3 run_deeplab.py --inference --backbone=drn_c42 --out-stride=8 \
    --workers=2 --epochs=1 --test-batch-size=1 --gpu-ids=0 --resume=SJ_0.2_True_0.0005_0.0001_1.03_8_superresx2_sharp \
    --window-size=256 --stride=256 --best-miou \
    --input-filename='/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/train/images/m_3712141_se_10_060_20200525_134.npy' \
    --output-filename='p2_sj_adu_0.2_1.03_superresx2_sharp.npy'

