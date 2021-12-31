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


 python scripts/train_dyna.py --dataset_path /data/argoverse --rho-reg --batch_size 12 --val_batch_size 12 --model_name nll_dyna_local --batches_per_epoch 600 --val_batches 20 --epochs 100 --loss nll --use_lane --train

python scripts/train.py --dataset_path /data/argoverse --rho-reg --batch_size 14 --val_batch_size 12 --model_name lane_nll_scaled --batches_per_epoch 600 --val_batches 20 --epochs 100  --loss nll --use_lane --train 

{3.84 2.09 0.967}

#reproduction scripts
python scripts/train.py --dataset_path ../argoverse_data --rho-reg --batch_size 1 --val_batch_size 1 --use_lane --train --model_name pecco_run1 --batches_per_epoch 1200 --val_batches 50 --epochs 100 --loss nll
python scripts/train_dynamics.py --dataset_path ../argoverse_data --rho-reg --batch_size 1 --val_batch_size 1 --use_lane --train --model_name peccodyna_scaled --batches_per_epoch 1200 --val_batches 50 --epochs 100 --loss nll

python scripts/ped_train_dyna.py --dataset_path ../pedestrian --rho-reg --batch_size 1 --val_batch_size 1 --model_name ped_local --batches_per_epoch 1200 --val_batches 30 --epochs 100 --loss nll --train
python scripts/ped_train.py --dataset_path ../pedestrian --rho-reg --batch_size 1 --val_batch_size 1 --model_name ped_nodyna_local --batches_per_epoch 1400 --val_batches 40 --epochs 100 --loss nll --train

python scripts/ped_train.py --dataset_path ../pedestrian --rho-reg --batch_size 4 --val_batch_size 4 --model_name ped_nodyna --batches_per_epoch 600 --val_batches 100 --epochs 100 --loss nll --train

python scripts/ped_train_dyna.py --dataset_path ../ped_one --rho-reg --batch_size 1 --val_batch_size 1 --model_name ped_local --batches_per_epoch 1 --val_batches 1 --epochs 100 --loss nll --val_window 6 --train

python scripts/cstconv_ped.py --dataset_path ../pedestrian --rho-reg --batch_size 4 read_pkl_data

# nri
python scripts/train_nri.py --dataset_path ../nridata/one --rho-reg --batch_size 1 --val_batch_size 1  --train --model_name nri_one --batches_per_epoch 1 --val_batches 1 --epochs 100 --loss nll

## LSTM
python scripts/lstm_train_test.py --train_features ../argoverse_data/train_rose.pkl  --val_features ../argoverse_data/val_rose.pkl --model_path ./checkpoints/lstm/LSTM_rollout30.pth.tar --test

python scripts/lstm_train_test_ped.py --train_features ../pedestrian/train  --val_features ../pedestrian/val  --name ped_lstm
python scripts/lstm_train_test_ped.py --train_features ../pedestrian/train  --val_features ../pedestrian/val  --name ped_mis_no_rot --mis_loss

python scripts/lstm_train_test_ped.py --train_features ../pedestrian/train_.pkl  --val_features ../pedestrian/val_.pkl --name ped_lstm


python scripts/lstm_train_test_nri.py --train_features ../nridata/train.pkl  --val_features ../nridata/val.pkl --name nri5_lstm 
python scripts/lstm_train_test_nri.py --train_features ../nridata/train10.pkl  --val_features ../nridata/val10.pkl --name nri10_lstm


models:

## argoverse

lstm: ./checkpoints/lstmfixed/LSTM_rollout30.pth.tar
lstmmis:  ./checkpoints/mis_no_rot/LSTM_rollout30.pth.tar 
cstconv: ablation
nll: lane_nll_naut
dyna: nll_dyna_local 
good dyna: newdyna

## pedestrian

python scripts/ped_train.py --dataset_path ../pedestrian --rho-reg --batch_size 1 --val_batch_size 1 --model_name ped_nodyna_hotel --batches_per_epoch 1400 --val_batches 40 --epochs 100 --loss nll --train #--evaluation 
python scripts/ped_train_dyna.py --dataset_path ../pedestrian --rho-reg --batch_size 1 --val_batch_size 1 --model_name ped_nodyna_hotel --batches_per_epoch 1400 --val_batches 40 --epochs 100 --loss nll --train #--evaluation 


python3.6 train.py --eval_every 10 --vis_every 1 --train_data_dict trajnet_train.pkl --eval_data_dict trajnet_val.pkl --offline_scene_graph no --preprocess_workers 5 --log_dir ../experiments/pedestrians/models --log_tag _trajnet_first --train_epochs 100 --augment --conf ../experiments/pedestrians/models/eth_vel/config.json" 


python scripts/ped_train_dyna.py --dataset_path ../pedestrian/ped_original/ --rho-reg --batch_size 1 --val_batch_size 1 --model_name peddyna_new --batches_per_epoch 1400 --val_batches 40 --epochs 100 --loss nll --evaluation

scttcov: ped_cstconv
nll: ped_nodyna
dyna: ped_local
