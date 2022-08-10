#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=rgb-hypertune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=avaimar@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --mem=50GB
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00
#SBATCH --array=0-17%1

cd ../../

backbone_list="drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42 drn_c42
"
out_stride_list="8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8
"
dataset_list="combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip combined_naip
"
loss_type_list="wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice wce_dice
"
workers_list="4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
"
fbeta_list="0.15 0.15 0.15 0.15 0.15 0.15 0.2 0.2 0.2 0.2 0.2 0.2 0.25 0.25 0.25 0.25 0.25 0.25
"
epochs_list="70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70
"
batch_size_list="24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24
"
test_batch_size_list="4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
"
weight_decay_list="0.0009821 7.444e-05 0.00054426 3.582e-05 0.00037878 2.053e-05 0.00030879 4.949e-05 0.00086776 2.156e-05 0.00016199 8.93e-05 0.00012133 8.403e-05 0.00053704 3.486e-05 0.00099283 9.158e-05
"
loss_weights_list="1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0 1.0,1.0
"
lr_list="0.0006996 0.0094572 0.0007212 0.0097927 0.0004032 0.0098519 0.0002535 0.0074523 0.000861 0.0017896 0.0008872 0.0019201 0.0005785 0.0032615 0.0002779 0.0034187 0.0003585 0.00621
"
dropout_list="0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5 0.3,0.5
"
gpu_ids_list="0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2 0,1,2
"
resume_list="crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI crowdAI
"
data_root_list="/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/ /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/
"
loss_weights_param_list="1.025 1.025 1.03 1.03 1.035 1.035 1.025 1.025 1.03 1.03 1.035 1.035 1.025 1.025 1.03 1.03 1.035 1.035
"
superres_list="2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
"
backbone_list=($backbone_list)
out_stride_list=($out_stride_list)
dataset_list=($dataset_list)
loss_type_list=($loss_type_list)
workers_list=($workers_list)
fbeta_list=($fbeta_list)
epochs_list=($epochs_list)
batch_size_list=($batch_size_list)
test_batch_size_list=($test_batch_size_list)
weight_decay_list=($weight_decay_list)
loss_weights_list=($loss_weights_list)
lr_list=($lr_list)
dropout_list=($dropout_list)
gpu_ids_list=($gpu_ids_list)
resume_list=($resume_list)
data_root_list=($data_root_list)
loss_weights_param_list=($loss_weights_param_list)
superres_list=($superres_list)


singularity exec --nv $GROUP_HOME/singularity/rgb-building1.sif python run_deeplab.py --incl-bounds --best-miou --freeze-bn --preempt-robust --use-wandb --backbone ${backbone_list[$SLURM_ARRAY_TASK_ID]} --out-stride ${out_stride_list[$SLURM_ARRAY_TASK_ID]} --dataset ${dataset_list[$SLURM_ARRAY_TASK_ID]} --loss-type ${loss_type_list[$SLURM_ARRAY_TASK_ID]} --fbeta ${fbeta_list[$SLURM_ARRAY_TASK_ID]} --workers ${workers_list[$SLURM_ARRAY_TASK_ID]} --epochs ${epochs_list[$SLURM_ARRAY_TASK_ID]} --batch-size ${batch_size_list[$SLURM_ARRAY_TASK_ID]} --weight-decay ${weight_decay_list[$SLURM_ARRAY_TASK_ID]} --lr ${lr_list[$SLURM_ARRAY_TASK_ID]} --loss-weights ${loss_weights_list[$SLURM_ARRAY_TASK_ID]} --dropout ${dropout_list[$SLURM_ARRAY_TASK_ID]} --gpu-ids ${gpu_ids_list[$SLURM_ARRAY_TASK_ID]} --resume ${resume_list[$SLURM_ARRAY_TASK_ID]} --test-batch-size ${test_batch_size_list[$SLURM_ARRAY_TASK_ID]} --data-root ${data_root_list[$SLURM_ARRAY_TASK_ID]} --loss-weights-param ${loss_weights_param_list[$SLURM_ARRAY_TASK_ID]} --superres ${superres_list[$SLURM_ARRAY_TASK_ID]}
