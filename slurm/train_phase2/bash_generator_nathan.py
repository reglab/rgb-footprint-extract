import os
import sys


def put_qmark(s):
    s = "\"" + s +"\""
    return s


def generate(backbone, out_stride, dataset, workers, loss_type, fbeta, epochs, batch_size, test_batch_size, weight_decay,
        gpu_ids, lr, loss_weights, dropout, resume, data_root, loss_weights_param, superres, array):

    backbone_list = []
    out_stride_list = []
    dataset_list = []
    workers_list = []
    loss_type_list = []
    fbeta_list = []
    epochs_list = []
    batch_size_list = []
    test_batch_size_list = []
    weight_decay_list = weight_decay
    gpu_ids_list = []
    lr_list = lr
    loss_weights_list = []
    dropout_list = []
    resume_list = []
    data_root_list = []
    loss_weights_param_list = []
    superres_list = []
    for b in backbone:
        for o in out_stride:
            for d in dataset:
                for l in loss_type:
                    for f in fbeta:
                        for w in workers:
                            for e in epochs:
                                for bs in batch_size:
                                    for tb in test_batch_size:
                                        for lw in loss_weights:
                                            for g in gpu_ids:
                                                for da in data_root:
                                                    for dr in dropout:
                                                        for r in resume:
                                                            for lo in loss_weights_param:
                                                                for s in superres:
                                                                    backbone_list.extend([b]*2)
                                                                    out_stride_list.extend([o]*2)
                                                                    dataset_list.extend([d]*2)
                                                                    loss_type_list.extend([l]*2)
                                                                    fbeta_list.extend([f]*2)
                                                                    workers_list.extend([w]*2)
                                                                    epochs_list.extend([e]*2)
                                                                    batch_size_list.extend([bs]*2)
                                                                    test_batch_size_list.extend([tb]*2)
                                                                    gpu_ids_list.extend([g]*2)
                                                                    loss_weights_list.extend([lw]*2)
                                                                    dropout_list.extend([dr]*2)
                                                                    data_root_list.extend([da]*2)
                                                                    resume_list.extend([r]*2)
                                                                    loss_weights_param_list.extend([lo]*2)
                                                                    superres_list.extend([s]*2)



    S = "#!/bin/bash\n"
    S += "#SBATCH --begin=now\n"
    S += "#SBATCH --job-name=rgb-hypertune\n"
    S += "#SBATCH --mail-type=ALL\n"
    S += "#SBATCH --mail-user=nathanjo@law.stanford.edu\n"
    S += "#SBATCH --partition=owners\n"
    S += "#SBATCH --mem=50GB\n"
    S += "#SBATCH --gres=gpu:3\n"
    S += "#SBATCH --time=12:00:00\n"
    S += "#SBATCH --array=0-"
    S += f'{array}%1'

    S += "\n"
    S += "\n"

    S += "cd ../../"

    S += "\n"
    S += "\n"

    S += "backbone_list=" + put_qmark(" ".join(str(item) for item in backbone_list) + "\n")
    S += "\n"
    S += "out_stride_list=" + put_qmark(" ".join(str(item) for item in out_stride_list) + "\n")
    S += "\n"
    S += "dataset_list=" + put_qmark(" ".join(str(item) for item in dataset_list) + "\n")
    S += "\n"
    S += "loss_type_list=" + put_qmark(" ".join(str(item) for item in loss_type_list) + "\n")
    S += "\n"
    S += "workers_list=" + put_qmark(" ".join(str(item) for item in workers_list) + "\n")
    S += "\n"
    S += "fbeta_list=" + put_qmark(" ".join(str(item) for item in fbeta_list) + "\n")
    S += "\n"
    S += "epochs_list=" + put_qmark(" ".join(str(item) for item in epochs_list) + "\n")
    S += "\n"
    S += "batch_size_list=" + put_qmark(" ".join(str(item) for item in batch_size_list) + "\n")
    S += "\n"
    S += "test_batch_size_list=" + put_qmark(" ".join(str(item) for item in test_batch_size_list) + "\n")
    S += "\n"
    S += "weight_decay_list=" + put_qmark(" ".join(str(item) for item in weight_decay_list) + "\n")
    S += "\n"
    S += "loss_weights_list=" + put_qmark(" ".join(str(item) for item in loss_weights_list) + "\n")
    S += "\n"
    S += "lr_list=" + put_qmark(" ".join(str(item) for item in lr_list) + "\n")
    S += "\n"
    S += "dropout_list=" + put_qmark(" ".join(str(item) for item in dropout_list) + "\n")
    S += "\n"
    S += "gpu_ids_list=" + put_qmark(" ".join(str(item) for item in gpu_ids_list) + "\n")
    S += "\n"
    S += "resume_list=" + put_qmark(" ".join(str(item) for item in resume_list) + "\n")
    S += "\n"
    S += "data_root_list=" + put_qmark(" ".join(str(item) for item in data_root_list) + "\n")
    S += "\n"
    S += "loss_weights_param_list=" + put_qmark(" ".join(str(item) for item in loss_weights_param_list) + "\n")
    S += "\n"
    S += "superres_list=" + put_qmark(" ".join(str(item) for item in superres_list) + "\n")
    S += "\n"
    S += 'backbone_list=($backbone_list)' + "\n"
    S += 'out_stride_list=($out_stride_list)' + "\n"
    S += 'dataset_list=($dataset_list)' + "\n"
    S += 'loss_type_list=($loss_type_list)' + "\n"
    S += 'workers_list=($workers_list)' + "\n"
    S += 'fbeta_list=($fbeta_list)' + "\n"
    S += 'epochs_list=($epochs_list)' + "\n"
    S += 'batch_size_list=($batch_size_list)' + "\n"
    S += 'test_batch_size_list=($test_batch_size_list)' + "\n"
    S += 'weight_decay_list=($weight_decay_list)' + "\n"
    S += 'loss_weights_list=($loss_weights_list)' + "\n"
    S += 'lr_list=($lr_list)' + "\n"
    S += 'dropout_list=($dropout_list)' + "\n"
    S += 'gpu_ids_list=($gpu_ids_list)' + "\n"
    S += 'resume_list=($resume_list)' + "\n"
    S += 'data_root_list=($data_root_list)' + "\n"
    S += 'loss_weights_param_list=($loss_weights_param_list)' + "\n"
    S += 'superres_list=($superres_list)' + "\n"

    S += "\n"
    S += "\n"
    command = 'singularity exec --nv $GROUP_HOME/singularity/rgb-building1.sif python run_deeplab.py --incl-bounds --best-miou --freeze-bn ' + \
                '--preempt-robust --use-wandb --backbone ' + \
              '${backbone_list[$SLURM_ARRAY_TASK_ID]}' + ' --out-stride '  + '${out_stride_list[$SLURM_ARRAY_TASK_ID]}' + \
                ' --dataset '  + '${dataset_list[$SLURM_ARRAY_TASK_ID]}' + \
                ' --loss-type '  + '${loss_type_list[$SLURM_ARRAY_TASK_ID]}' + \
                ' --fbeta '  + '${fbeta_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --workers ' + '${workers_list[$SLURM_ARRAY_TASK_ID]}' + ' --epochs ' + '${epochs_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --batch-size '  + '${batch_size_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --weight-decay '  + '${weight_decay_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --lr '  + '${lr_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --loss-weights '  + '${loss_weights_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --dropout '  + '${dropout_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --gpu-ids ' + '${gpu_ids_list[$SLURM_ARRAY_TASK_ID]}'  + \
              ' --resume ' + '${resume_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --test-batch-size ' + '${test_batch_size_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --data-root ' + '${data_root_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --loss-weights-param ' + '${loss_weights_param_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --superres ' + '${superres_list[$SLURM_ARRAY_TASK_ID]}'

    S += command
    S += "\n"

    slurm_file = 'hypertune_nathan.sh'
    f = open(slurm_file, "w+")
    f.write(S)
    f.close()
    # print(slurm_file)


def main():

    backbone = ['drn_c42']
    out_stride = [8]
    dataset = ['OSM']
    workers = [4]
    loss_type = ['wce_dice']
    fbeta = [0.15, 0.2, 0.25]
    epochs = [70]
    batch_size = [24]
    test_batch_size = [4]
    weight_decay = [2.403e-05,
  0.00060935,
  8.505e-05,
  0.00011687,
  2.699e-05,
  0.00088518,
  2.948e-05,
  0.00026511,
  4.118e-05,
  0.00010202,
  7.917e-05,
  0.00075264,
  5.478e-05,
  0.00029785,
  9.643e-05,
  0.00031588,
  3.521e-05,
  0.00069474]
    # weight_decay = [1e-4, 1e-2]
    gpu_ids = ['0,1,2']
    # lr = [5e-4, 5e-3]
    lr = [0.0002403,
  0.0063974,
  0.0007751,
  0.0028153,
  0.0005799,
  0.0084325,
  0.000856,
  0.0052854,
  0.0007032,
  0.0033791,
  0.000995,
  0.0052322,
  0.0001711,
  0.0021573,
  0.0001908,
  0.0061425,
  0.0002035,
  0.0054401]
    loss_weights = ['1.0,1.0']
    dropout = ['0.3,0.5']
    resume = ['crowdAI']
    data_root = ['/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/']
    loss_weights_param = [1.025, 1.03, 1.035]
    superres = [2]
    

    array = len(fbeta) * 2 * len(loss_weights_param)- 1

    generate(backbone, out_stride, dataset, workers, loss_type, fbeta, epochs, batch_size, test_batch_size, weight_decay,
        gpu_ids, lr, loss_weights, dropout, resume, data_root, loss_weights_param, superres, array)

if __name__ == "__main__":
    main()