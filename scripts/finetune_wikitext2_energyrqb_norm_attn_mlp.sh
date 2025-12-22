export CUDA_VISIBLE_DEVICES=0
python finetune_meft_wikitext2.py --lora_weights_output_dir 'output/wikitext2_lora_weights/meft_energyqb_norm_attn_mlp_0.5_compress_lora_weights' \
--num_epoch 1 \
--learning_rate 3e-5 \
--using_meft \
--using_compress \
--compress_rank 0.125 \
--compress_method probing_rqb \
--lora_r 64 \
--patch_locations 2 \
--max_length 1024 \
--device_map "cuda:0"