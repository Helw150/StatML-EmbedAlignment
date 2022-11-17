export CUDA_VISIBLE_DEVICES=1

python train_xnli.py \
       --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
       --push_to_hub_model_id  WillHeld/es-bert-xnli \
       --language es \
       --train_language es \
       --do_train \
       --do_eval \
       --per_device_train_batch_size 32 \
       --learning_rate 5e-5 \
       --num_train_epochs 2.0 \
       --max_seq_length 128 \
       --output_dir /tmp/debug_xnli/ \
       --save_steps -1
