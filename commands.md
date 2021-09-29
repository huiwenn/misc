## PECCO
python scripts/train.py --dataset_path ../argoverse_data --rho-reg --model_name rho_reg_ecco --batch_size 1 --val_batch_size 1 --train --loss ecco

python scripts/ablation.py --dataset_path ../argoverse_data --rho-reg --batch_size 1 --val_batch_size 1 --train --model_name ablation --batches_per_epoch 1000 --val_batches 60 --epochs 100 --loss nll

python scripts/train.py --dataset_path ../argoverse_data --rho-reg --model_name rho_reg_pecco --batch_size 2 --val_batch_size 1 --train --model_name nll --batches_per_epoch 1000 --val_batches 80 --epochs 100 --loss ecco --load_model_path debug

python scripts/train.py --dataset_path ../argoverse_one --rho-reg --model_name one --batch_size 1  --batches_per_epoch 1 --val_batches 1 --epochs 50 --loss ecco --train --use_lane

python scripts/train.py --dataset_path ../argoverse_data --rho-reg --batch_size 1 --val_batch_size 1 --train --model_name lane_local_profile --batches_per_epoch 1000 --val_batches 80 --epochs 100 --loss nll 

python scripts/train.py --dataset_path ../argoverse_one --rho-reg --batch_size 1 --val_batch_size 1  --batches_per_epoch 1 --val_batches 1 --epochs 100 --loss mis --val_window 4 --train --model_name mis_local

python scripts/train.py --dataset_path /data/argoverse --rho-reg --batch_size 12 --val_batch_size 12 --train --model_name nll_nautilus --batches_per_epoch 600 --val_batches 50 --epochs 100 --loss nll --load_model_path nll_nautilus --use_lane

python scripts/train.py --dataset_path /data/argoverse --rho-reg --batch_size 16 --val_batch_size 16 --train --model_name mis_lane --batches_per_epoch 600 --val_batches 50 --epochs 100 --loss ecco --use_lane

python scripts/train.py --dataset_path ../argoverse_one --rho-reg --model_name nll_nautilus --batch_size 1  --batches_per_epoch 1 --val_batches 1 --epochs 50 --val_batches 10  --loss nll --evaluation --use_lane


python scripts/train_mod.py --dataset_path ../argoverse_data --rho-reg --batch_size 1 --val_batch_size 1 --model_name mod --batches_per_epoch 1200 --val_batches 45 --epoch 200 --train


 python scripts/train_mod.py --dataset_path /data/argoverse --rho-reg --batch_size 14 --val_batch_size 12 --train --model_name ecco_naut --batches_per_epoch 600 --val_batches 30 --epochs 100 --loss ecco --use_lane --load_model_path lane_nll

 python scripts/train_dyna.py --dataset_path /data/argoverse --rho-reg --batch_size 12 --val_batch_size 12 --model_name nll_dyna_local --batches_per_epoch 600 --val_batches 20 --epochs 100 --loss nll --use_lane --train 


python scripts/train_dynamics.py --dataset_path ../argoverse_data --rho-reg --batch_size 1 --val_batch_size 1 --batches_per_epoch 1200 --val_batches 150 --epoch 200 --model_name dyna_local_2 --loss nll --use_lane --train --load_model_path nll_dyna_local --base_lr 0.0005

python scripts/train_dynamics.py --dataset_path /data/argoverse --rho-reg --batch_size 14 --val_batch_size 12 --model_name lane_nll_scaled --batches_per_epoch 600 --val_batches 20 --epochs 100  --loss nll --use_lane --train 

python scripts/ped_train_dyna.py --dataset_path ../ped_one --rho-reg --batch_size 1 --val_batch_size 1 --model_name ped_dyna_local --batches_per_epoch 1 --val_batches 1 --epochs 100 --val_window 6 --loss nll --train 

python scripts/ped_train_dyna.py --dataset_path ../pedestrian --rho-reg --batch_size 1 --val_batch_size 1 --model_name ped_dyna_local --batches_per_epoch 1200 --val_batches 30 --epochs 100 --loss nll --train 

{3.84 2.09 0.967}



python scripts/ped_train.py --dataset_path ../pedestrian --rho-reg --batch_size 1 --val_batch_size 1 --model_name ped_local --batches_per_epoch 1200 --val_batches 30 --epochs 100 --loss nll --train

python scripts/ped_train.py --dataset_path ../ped_one --rho-reg --batch_size 1 --val_batch_size 1 --model_name ped_local --batches_per_epoch 1 --val_batches 1 --epochs 100 --loss nll --val_window 6 --train


## LSTM

python scripts/lstm_train_test.py --train_features ../pedestrian/train_.pkl  --val_features ../pedestrian/val_.pkl  --obs_len 6 --pred_len 12 --name ped_trans 
