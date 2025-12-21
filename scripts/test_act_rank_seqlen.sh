export CUDA_VISIBLE_DEVICES=3
python act_rank.py --layer 22 --batch_size 16 --seq_len 256 --device_map "cuda:0"
python act_rank.py --layer 22 --batch_size 16 --seq_len 512 --device_map "cuda:0"
python act_rank.py --layer 22 --batch_size 16 --seq_len 1024 --device_map "cuda:0"
python act_rank.py --layer 22 --batch_size 16 --seq_len 2048 --device_map "cuda:0"
python act_rank.py --layer 22 --batch_size 16 --seq_len 4096 --device_map "cuda:0"