export CUDA_VISIBLE_DEVICES=0
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --using_meft --compress_rank 0.125 --patch_locations 1 --micro_batch_size 1 --cutoff_len 256
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --using_meft --compress_rank 0.125 --patch_locations 1 --micro_batch_size 2 --cutoff_len 256
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --using_meft --compress_rank 0.125 --patch_locations 1 --micro_batch_size 4 --cutoff_len 256
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --using_meft --compress_rank 0.125 --patch_locations 1 --micro_batch_size 8 --cutoff_len 256
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --using_meft --compress_rank 0.125 --patch_locations 1 --micro_batch_size 16 --cutoff_len 256
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --using_meft --compress_rank 0.125 --patch_locations 1 --micro_batch_size 32 --cutoff_len 256
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --using_meft --compress_rank 0.125 --patch_locations 1 --micro_batch_size 64 --cutoff_len 256