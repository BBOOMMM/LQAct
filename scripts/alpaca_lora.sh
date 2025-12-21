python finetune_meft_alpaca.py --data_path 'yahma/alpaca-cleaned' \
--lora_weights_output_dir 'output/alpaca_lora_weights/original_lora_weights_lr3e-5' \
--num_epoch 1 \
--learning_rate 3e-5 \
--using_compress \
--compress_rank 0.25 \
--lora_r 64 \
--patch_locations 1 \
--cutoff_len 256 \
--device_map "cuda:0"