python qlora.py \
    --model_name_or_path HuggingFaceH4/zephyr-7b-alpha \
    --output_dir ./output/zephyr-7b-2 \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 4 \
    --max_new_tokens 32 \
    --dataloader_num_workers 8 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 8 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset oasst1 \
    --source_max_len 600 \
    --target_max_len 1200 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 1000 \
    --eval_steps 200 \
    --save_steps 200 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0
