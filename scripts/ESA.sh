export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 1 --num_epochs 2   --batch_size 256  --mode train --dataset 3_months  --data_path dataset/ESA   --input_c 6 --output_c 6
python main.py --anormly_ratio 1 --num_epochs 10   --batch_size 256     --mode test    --dataset 3_months   --data_path dataset/ESA     --input_c 6     --pretrained_model 20