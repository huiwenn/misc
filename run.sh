echo yes | conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
python scripts/train.py --dataset_path /data/argoverse --rho-reg --batch_size 12 --val_batch_size 12 --train --model_name nll --batches_per_epoch 500 --val_batches 50 --epochs 100 --loss nll
