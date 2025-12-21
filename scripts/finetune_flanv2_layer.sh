export CUDA_VISIBLE_DEVICES=2
python finetune_meft_flanv2.py --data_path 'chiayewken/flan-v2' \
--lora_weights_output_dir 'output_lr3e-5/flanv2/meft_layer_0.015625_compress_lora_weights' \
--num_epoch 1 \
--learning_rate 3e-5 \
--using_meft \
--using_compress \
--compress_rank 0.015625 \
--lora_r 64 \
--patch_locations 1 \
--cutoff_len 256 \
--device_map "cuda:0" \

python finetune_meft_flanv2.py --data_path 'chiayewken/flan-v2' \
--lora_weights_output_dir 'output/flan_v2_lora_weights/meft_layer_0.0625_compress_lora_weights_256_lr1e-5' \
--num_epoch 1 \
--learning_rate 1e-5 \
--using_meft \
--using_compress \
--compress_rank 0.0625 \
--lora_r 64 \
--patch_locations 1 \
--cutoff_len 256 \
--device_map "cuda:0" \

python finetune_meft_flanv2.py --data_path 'chiayewken/flan-v2' \
--lora_weights_output_dir 'output/flan_v2_lora_weights/meft_layer_0.25_compress_lora_weights_256_lr1e-5' \
--num_epoch 1 \
--learning_rate 1e-5 \
--using_meft \
--using_compress \
--compress_rank 0.25 \
--lora_r 64 \
--patch_locations 1 \
--cutoff_len 256 \
--device_map "cuda:0" \



