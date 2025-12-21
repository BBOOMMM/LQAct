export CUDA_VISIBLE_DEVICES=0
python act_rank.py --layer 22 --batch_size 2 --seq_len 256 --device_map "cuda:0"
python act_rank.py --layer 22 --batch_size 4 --seq_len 256 --device_map "cuda:0"
python act_rank.py --layer 22 --batch_size 8 --seq_len 256 --device_map "cuda:0"
python act_rank.py --layer 22 --batch_size 16 --seq_len 256 --device_map "cuda:0"
python act_rank.py --layer 22 --batch_size 32 --seq_len 256 --device_map "cuda:0"
python act_rank.py --layer 22 --batch_size 128 --seq_len 256 --device_map "cuda:0"
python act_rank.py --layer 22 --batch_size 256 --seq_len 256 --device_map "cuda:0"

