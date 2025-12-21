python test1.py --lora_model "./output/svhn/meft_layer_0.5_compress_lora_weights" --data_path "ufldl-stanford/svhn"
python test1.py --lora_model "./output/svhn/meft_layer_0.25_compress_lora_weights" --data_path "ufldl-stanford/svhn"    
python test1.py --lora_model "./output/svhn/meft_layer_0.125_compress_lora_weights" --data_path "ufldl-stanford/svhn"
python test1.py --lora_model "./output/svhn/meft_layer_0.0625_compress_lora_weights" --data_path "ufldl-stanford/svhn"


python test1.py --lora_model "./output/svhn/meft_norm_attn_mlp_0.5_compress_lora_weights" --data_path "ufldl-stanford/svhn"
python test1.py --lora_model "./output/svhn/meft_norm_attn_mlp_0.25_compress_lora_weights"  --data_path "ufldl-stanford/svhn"
python test1.py --lora_model "./output/svhn/meft_norm_attn_mlp_0.125_compress_lora_weights"  --data_path "ufldl-stanford/svhn"
python test1.py --lora_model "./output/svhn/meft_norm_attn_mlp_0.0625_compress_lora_weights" --data_path "ufldl-stanford/svhn"
