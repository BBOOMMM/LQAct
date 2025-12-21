export CUDA_VISIBLE_DEVICES=1
python wikitext2_memory_record.py --device_map "cuda:0" --using_meft --compress_rank 0.5 --patch_locations 2 --micro_batch_size 16 --max_length 1024
python wikitext2_memory_record.py --device_map "cuda:0" --using_meft --compress_rank 0.25 --patch_locations 2 --micro_batch_size 16 --max_length 1024
python wikitext2_memory_record.py --device_map "cuda:0" --using_meft --compress_rank 0.125 --patch_locations 2 --micro_batch_size 16 --max_length 1024
python wikitext2_memory_record.py --device_map "cuda:0" --using_meft --compress_rank 0.0625 --patch_locations 2 --micro_batch_size 16 --max_length 1024
