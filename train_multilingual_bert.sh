export CUDA_VISIBLE_DEVICES=2

python train_xnli.py \
       --model_name_or_path bert-base-multilingual-cased \
       --push_to_hub_model_id multi-bert-xnli \
       --push_to_hub True \
       --output_dir ./multi \
       --language en \
       --train_language en \
       --do_train \
       --do_eval \
       --per_device_train_batch_size 32 \
       --learning_rate 5e-5 \
       --num_train_epochs 2.0 \
       --max_seq_length 128 \
       --save_steps -1
