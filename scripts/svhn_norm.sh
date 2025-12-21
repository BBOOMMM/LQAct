export CUDA_VISIBLE_DEVICES=3
python finetune_image.py --dataset_path "ufldl-stanford/svhn" \
--subset "cropped_digits" \
--num_train_epochs 100 \
--lora_weights_output_dir 'output/svhn/meft_norm_attn_mlp_0.25_compress_lora_weights' \
--num_labels 10 \
--using_meft \
--using_compress \
--compress_rank 0.25 \
--lora_r 64 \
--patch_locations 2 \
--device_map "cuda:0" \

python finetune_image.py --dataset_path "ufldl-stanford/svhn" \
--num_train_epochs 100 \
--subset "cropped_digits" \
--lora_weights_output_dir 'output/svhn/meft_norm_attn_mlp_0.125_compress_lora_weights' \
--num_labels 10 \
--using_meft \
--using_compress \
--compress_rank 0.125 \
--lora_r 64 \
--patch_locations 2 \
--device_map "cuda:0" \

python finetune_image.py --dataset_path "ufldl-stanford/svhn" \
--num_train_epochs 10 \
--subset "cropped_digits" \
--lora_weights_output_dir 'output/svhn/meft_norm_attn_mlp_0.0625_compress_lora_weights' \
--num_labels 10 \
--using_meft \
--using_compress \
--compress_rank 0.0625 \
--lora_r 64 \
--patch_locations 2 \
--device_map "cuda:0" \