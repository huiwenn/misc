## PECCO
python scripts/train.py --dataset_path ../argoverse_data --rho-reg --model_name rho_reg_pecco --batch_size 1 --val_batch_size 1 --train

python scripts/train.py --dataset_path ../argoverse_data/rose --rho-reg --model_name rho_reg_pecco --batch_size 1 --val_batch_size 1 --train --batches_per_epoch 1000 --val_batches 200 --model_name test

python scripts/train.py --dataset_path ../argoverse_data --rho-reg --model_name rho_reg_pecco --batch_size 2 --val_batch_size 1 --train --model_name debug2 --batches_per_epoch 1000 --val_batches 80 --epochs 100 --loss ecco --load_model_path debug

python scripts/train.py --dataset_path ../argoverse_data/rose --rho-reg --model_name test --batch_size 1 --train --batches_per_epoch 1 --val_batches 1 --epochs 50

