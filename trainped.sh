pip install torch==1.8.0
pip install tensorpack
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
python scripts/ped_train.py --dataset_path /data/pedestrian --rho-reg --batch_size 4 --val_batch_size 4 --model_name ped_nodyna --batches_per_epoch 600 --val_batches 100 --epochs 100 --loss nll --train
python scripts/ped_train_dyna.py --dataset_path ../pedestrian --rho-reg --batch_size 16 --batch_divide 2 --val_batch_size 8 --model_name new_ped_dyna --batches_per_epoch 300 --val_batches 10 --epochs 100 --loss nll --train

python scripts/ped_train_dyna.py --dataset_path ../pedestrian/processed --rho-reg --batch_size 32 --batch_divide 8 --val_batch_size 8 --model_name ped_dyna_32 --batches_per_epoch 150 --val_batches 50 --epochs 100 --loss nll --train --cuda_visible_devices 0,1,2,3

#python scripts/train_dynamics.py --dataset_path ../argoverse_data --rho-reg --batch_size 32 --batch_divide 32 --val_batch_size 1 --model_name newdynatest --batches_per_epoch 3 --val_batches 5 --epochs 100 --loss nll --use_lane --train
#python scripts/train_dynamics.py --dataset_path /data/argoverse --rho-reg --batch_size 1 --val_batch_size 1 --model_name newdyna --batches_per_epoch 600 --val_batches 20 --epochs 100 --loss nll --use_lane --train
#python scripts/train_dynamics.py --dataset_path /data/argoverse --rho-reg --batch_size 1 --val_batch_size 1 --model_name newdyna --batches_per_epoch 600 --val_batches 20 --epochs 100 --loss nll --use_lane --train