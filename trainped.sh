pip install torch==1.8.0
pip install tensorpack
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
python scripts/ped_train.py --dataset_path /data/pedestrian --rho-reg --batch_size 4 --val_batch_size 4 --model_name ped_nodyna --batches_per_epoch 600 --val_batches 100 --epochs 100 --loss nll --train
#python scripts/train_dynamics.py --dataset_path ../argoverse_data --rho-reg --batch_size 32 --batch_divide 32 --val_batch_size 1 --model_name newdynatest --batches_per_epoch 3 --val_batches 5 --epochs 100 --loss nll --use_lane --train
#python scripts/train_dynamics.py --dataset_path /data/argoverse --rho-reg --batch_size 1 --val_batch_size 1 --model_name newdyna --batches_per_epoch 600 --val_batches 20 --epochs 100 --loss nll --use_lane --train
#python scripts/train_dynamics.py --dataset_path /data/argoverse --rho-reg --batch_size 1 --val_batch_size 1 --model_name newdyna --batches_per_epoch 600 --val_batches 20 --epochs 100 --loss nll --use_lane --train