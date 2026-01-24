export CUDA_VISIBLE_DEVICES=0

python main_esa.py --anormly_ratio 1 --num_epochs 10   --batch_size 256  --mode train --dataset 84_months  --data_path dataset/ESA --train_length 84_months --test_length 84_months --input_c 6 --output_c 6
python main_esa.py --anormly_ratio 1 --num_epochs 1   --batch_size 256     --mode test    --dataset 84_months   --data_path dataset/ESA --train_length 84_months --test_length 84_months --input_c 6 --output_c 6 #--pretrained_model 20