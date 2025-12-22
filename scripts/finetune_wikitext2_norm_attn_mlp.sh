export CUDA_VISIBLE_DEVICES=3
python finetune_meft_wikitext2.py --lora_weights_output_dir 'output/wikitext2_lora_weights/meft_norm_attn_mlp_0.5_compress_lora_weights' \
--num_epoch 1 \
--learning_rate 3e-5 \
--using_meft \
--using_compress \
--compress_rank 0.5 \
--lora_r 64 \
--patch_locations 2 \
--max_length 1024 \
--device_map "cuda:0" \

CUDA_VISIBLE_DEVICES=1 python finetune_meft_wikitext2.py --lora_weights_output_dir 'output/wikitext2_lora_weights/meft_norm_attn_mlp_0.125_compress_lora_weights' \
--num_epoch 1 \
--learning_rate 4e-5 \
--using_meft \
--using_compress \
--compress_rank 0.125 \
--lora_r 64 \
--patch_locations 2 \
--max_length 1024 \
--device_map "cuda:0" \

python finetune_meft_wikitext2.py --lora_weights_output_dir 'output/wikitext2_lora_weights/meft_norm_attn_mlp_0.25_compress_lora_weights' \
--num_epoch 1 \
--learning_rate 4e-5 \
--using_meft \
--using_compress \
--compress_rank 0.25 \
--lora_r 64 \
--patch_locations 2 \
--max_length 1024 \
--device_map "cuda:0" \

python finetune_meft_wikitext2.py --lora_weights_output_dir 'output/wikitext2_lora_weights/meft_norm_attn_mlp_0.0625_compress_lora_weights' \
--num_epoch 1 \
--learning_rate 4e-5 \
--using_meft \
--using_compress \
--compress_rank 0.0625 \
--lora_r 64 \
--patch_locations 2 \
--max_length 1024 \
--device_map "cuda:0" \

