export CUDA_VISIBLE_DEVICES=1
python finetune_image.py --dataset_path "uoft-cs/cifar100" \
--num_train_epochs 100 \
--lora_weights_output_dir 'output/cifar100/meft_norm_attn_mlp_0.5_compress_lora_weights' \
--num_labels 100 \
--using_meft \
--using_compress \
--compress_rank 0.5 \
--lora_r 64 \
--patch_locations 2 \
--device_map "cuda:0" \

python finetune_image.py --dataset_path "uoft-cs/cifar100" \
--num_train_epochs 100 \
--lora_weights_output_dir 'output/cifar100/meft_norm_attn_mlp_0.125_compress_lora_weights' \
--num_labels 100 \
--using_meft \
--using_compress \
--compress_rank 0.125 \
--lora_r 64 \
--patch_locations 2 \
--device_map "cuda:0" \

python finetune_image.py --dataset_path "uoft-cs/cifar100" \
--num_train_epochs 100 \
--lora_weights_output_dir 'output/cifar100/meft_norm_attn_mlp_0.0625_compress_lora_weights' \
--num_labels 100 \
--using_meft \
--using_compress \
--compress_rank 0.0625 \
--lora_r 64 \
--patch_locations 2 \
--device_map "cuda:0" \