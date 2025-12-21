export CUDA_VISIBLE_DEVICES=3
python memory_record.py --compress_rank 1 --using_meft --patch_locations 1 --device_map "cuda:0"
python memory_record.py --compress_rank 0.5 --using_meft --patch_locations 1 --device_map "cuda:0"
python memory_record.py --compress_rank 0.25 --using_meft --patch_locations 1 --device_map "cuda:0"
python memory_record.py --compress_rank 0.125 --using_meft --patch_locations 1 --device_map "cuda:0"
python memory_record.py --compress_rank 0.0625 --using_meft --patch_locations 1 --device_map "cuda:0"
python memory_record.py --compress_rank 0.03125 --using_meft --patch_locations 1 --device_map "cuda:0"
python memory_record.py --compress_rank 0.015625 --using_meft --patch_locations 1 --device_map "cuda:0"
