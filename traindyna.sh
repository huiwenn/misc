pip install torch==1.8.0 &&
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
python scripts/train_dynamics.py --dataset_path ../argoverse --rho-reg --batch_size 16 --batch_divide 4 --val_batch_size 4 --use_lane --train --model_name dyna_south --batches_per_epoch 300 --val_batches 50 --loss nll --cuda_visible_devices '1,5,6,7'

python scripts/train_dynamics.py --dataset_path /data/argoverse_data --rho-reg --batch_size 16 --batch_divide 4 --val_batch_size 8 --use_lane --train --model_name dyna_mis_naut --batches_per_epoch 300 --val_batches 50 --epochs 100 --loss mis
#python scripts/train_dynamics.py --dataset_path /data/argoverse_data --rho-reg --batch_size 32 --batch_divide 8 --val_batch_size 4 --model_name newdyna --batches_per_epoch 600 --val_batches 50 --epochs 100 --loss nll --use_lane --train
#python scripts/train_dynamics.py --dataset_path ../argoverse_data --rho-reg --batch_size 32 --batch_divide 32 --val_batch_size 1 --model_name newdynatest --batches_per_epoch 3 --val_batches 5 --epochs 100 --loss nll --use_lane --train
#python scripts/train_dynamics.py --dataset_path /data/argoverse --rho-reg --batch_size 1 --val_batch_size 1 --model_name newdyna --batches_per_epoch 600 --val_batches 20 --epochs 100 --loss nll --use_lane --train
#python scripts/train_dynamics.py --dataset_path /data/argoverse --rho-reg --batch_size 1 --val_batch_size 1 --model_name newdyna --batches_per_epoch 600 --val_batches 20 --epochs 100 --loss nll --use_lane --train