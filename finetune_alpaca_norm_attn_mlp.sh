export CUDA_VISIBLE_DEVICES=1
python finetune_meft_alpaca.py --data_path 'yahma/alpaca-cleaned' \
--lora_weights_output_dir 'output_lr3e-5/alpaca_lora_weights/meft_norm_attn_mlp_0.015625_compress_lora_weights' \
--num_epoch 1 \
--learning_rate 3e-5 \
--using_meft \
--using_compress \
--compress_rank 0.015625 \
--lora_r 64 \
--patch_locations 2 \
--cutoff_len 256 \
--device_map "cuda:0" \

python finetune_meft_alpaca.py --data_path 'yahma/alpaca-cleaned' \
--lora_weights_output_dir 'output/alpaca_lora_weights/meft_norm_attn_mlp_0.125_compress_lora_weights_lr1e-5' \
--num_epoch 1 \
--learning_rate 1e-5 \
--using_meft \
--using_compress \
--compress_rank 0.125 \
--lora_r 64 \
--patch_locations 2 \
--cutoff_len 256 \
--device_map "cuda:0" \


python finetune_meft_alpaca.py --data_path 'yahma/alpaca-cleaned' \
--lora_weights_output_dir 'output/alpaca_lora_weights/meft_norm_attn_mlp_0.0625_compress_lora_weights_lr1e-5' \
--num_epoch 1 \
--learning_rate 1e-5 \
--using_meft \
--using_compress \
--compress_rank 0.0625 \
--lora_r 64 \
--patch_locations 2 \
--cutoff_len 256 \
--device_map "cuda:0" \