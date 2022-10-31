#!/bin/sh
#SBATCH -p dgx2q

#SBATCH -c 6

#SBATCH --gres=gpu:1
#SBATCH -t 336:00:00
#SBATCH --mail-user=joakimje@hiof.no

module load cuda11.0/toolkit/11.0.3
module load cudnn8.1-cuda11.2/8.1.1.33
module load ex3-modules
module load slurm/20.02.7
module load python-3.7.4


if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  omp_threads=$SLURM_CPUS_PER_TASK
else
  omp_threads=1
fi


export OMP_NUM_THREADS=$omp_thread          # OpenMP, Numpy
export MKL_NUM_THREADS=$omp_thread          # Intel MKL
export NUMEXPR_NUM_THREADS=$omp_thread      # Python3 Multiproc

export OMP_STACKSIZE=1G

export KMP_AFFINITY=scatter 

echo "Phosc"


python3 main.py --name ResNet34PretrainDrop1Dense --mode train --model Resnet34_temporalpooling --epochs 100 --train_csv /global/D1/homes/aniket/data/IAM_Data1/iamSplit_Aspect_1024#10_05_2011#/train.csv --train_folder /global/D1/homes/aniket/data/IAM_Data1/iamSplit_Aspect_1024#10_05_2011#/train --valid_csv /global/D1/homes/aniket/data/IAM_Data1/iamSplit_Aspect_1024#10_05_2011#/val1.csv --valid_folder /global/D1/homes/aniket/data/IAM_Data1/iamSplit_Aspect_1024#10_05_2011#/val


#python3 main.py --name ResNet34PretrainDrop1Dense --mode train --model Resnet34_temporalpooling --epochs 100 --train_csv /global/D1/projects/ZeroShot_Word_Recognition/Transformer_ZeroShot_Word_Recogniti#on/joakims_work/myphosc/image_data/IAM_Data/IAM_train.csv --train_folder /global/D1/projects/ZeroShot_Word_Recognition/Transformer_ZeroShot_Word_Recognition/joakims_work/myphosc/image_data/IAM_Data/IAM_t#rain --valid_csv /global/D1/projects/ZeroShot_Word_Recognition/Transformer_ZeroShot_Word_Recognition/joakims_work/myphosc/image_data/IAM_Data/IAM_valid_seen.csv --valid_folder /global/D1/projects/ZeroS#hot_Word_Recognition/Transformer_ZeroShot_Word_Recognition/joakims_work/myphosc/image_data/IAM_Data/IAM_valid

### ResNet18Phosc
# python main.py --name ResNet18Phosc --mode train --model ResNet18Phosc --epochs 100 --train_csv image_data/IAM_Data/IAM_train_extra_aug.csv --train_folder image_data/IAM_Data/IAM_train_extra_aug --valid_csv image_data/IAM_Data/IAM_valid_seen.csv --valid_folder image_data/IAM_Data/IAM_valid
# python main.py --name RPnet --mode train --model RPnet --epochs 100 --train_csv image_data/IAM_Data/IAM_train.csv --train_folder image_data/IAM_Data/IAM_train --valid_csv image_data/IAM_Data/IAM_valid_seen.csv --valid_folder image_data/IAM_Data/IAM_valid
# python main.py --name PHOSCnet_temporalpooling_more_data --mode train --model PHOSCnet_temporalpooling --epochs 100 --train_csv image_data/IAM_Data/IAM_train_extra_aug.csv --train_folder image_data/IAM_Data/IAM_train_extra_aug --valid_csv image_data/IAM_Data/IAM_valid_seen.csv --valid_folder image_data/IAM_Data/IAM_valid
# python main.py --name train_resmodel --mode train --model PHOSCnet_residual --epochs 500 --train_csv image_data/IAM_Data/IAM_train.csv --train_folder image_data/IAM_Data/IAM_train --valid_csv image_data/IAM_Data/IAM_valid_seen.csv --valid_folder image_data/IAM_Data/IAM_valid
# python main.py --name new_IAM_split_PHOSCnet_temporalpooling --mode train --model PHOSCnet_temporalpooling --train_csv image_data/IamSplit/augmented_data/train.csv --train_folder image_data/IamSplit/augmented_data/train --epochs 100
# python main.py --name train_model --mode train --model PHOSCnet_temporalpooling --train_csv image_data/IAM_Data/IAM_train.csv --train_folder image_data/IAM_Data/IAM_train --valid_csv image_data/IAM_Data/IAM_valid_seen.csv --valid_folder image_data/IAM_Data/IAM_valid --epochs 50

exit 0
