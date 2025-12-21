export CUDA_VISIBLE_DEVICES=0
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 1 --cutoff_len 256
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 1 --cutoff_len 512
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 1 --cutoff_len 1024
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 1 --cutoff_len 2048
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 1 --cutoff_len 4096

python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 2 --cutoff_len 256
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 2 --cutoff_len 512
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 2 --cutoff_len 1024
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 2 --cutoff_len 2048
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 2 --cutoff_len 4096

