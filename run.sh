echo yes | conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
python scripts/train.py --dataset_path /data/argoverse --rho-reg --model_name rho_reg_pecco --batch_size 12 --val_batch_size 32 --train --model_name cloud --batches_per_epoch 600 --val_batches 50 --epochs 100 --loss ecco
