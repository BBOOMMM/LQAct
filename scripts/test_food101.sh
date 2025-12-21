python test1.py --lora_model "./output/food101/meft_layer_0.5_compress_lora_weights" --data_path ethz/food101
python test1.py --lora_model "./output/food101/meft_layer_0.25_compress_lora_weights" --data_path ethz/food101
python test1.py --lora_model "./output/food101/meft_layer_0.125_compress_lora_weights" --data_path ethz/food101
python test1.py --lora_model "./output/food101/meft_layer_0.0625_compress_lora_weights" --data_path ethz/food101

python test1.py --lora_model "./output/food101/meft_norm_attn_mlp_0.5_compress_lora_weights" --data_path ethz/food101
python test1.py --lora_model "./output/food101/meft_norm_attn_mlp_0.25_compress_lora_weights" --data_path ethz/food101  
python test1.py --lora_model "./output/food101/meft_norm_attn_mlp_0.125_compress_lora_weights" --data_path ethz/food101
python test1.py --lora_model "./output/food101/meft_norm_attn_mlp_0.0625_compress_lora_weights" --data_path ethz/food101
