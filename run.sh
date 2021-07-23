install torch
python scripts/train.py --dataset_path /data/argoverse --rho-reg --model_name rho_reg_pecco --batch_size 16 --val_batch_size 32 --train --model_name cloud --batches_per_epoch 600 --val_batches 50 --epochs 100 --loss ecco
