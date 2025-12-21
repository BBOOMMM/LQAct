export CUDA_VISIBLE_DEVICES=1
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 4 --cutoff_len 256
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 4 --cutoff_len 512
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 4 --cutoff_len 1024
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 4 --cutoff_len 2048
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 4 --cutoff_len 4096

python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 8 --cutoff_len 256
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 8 --cutoff_len 512
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 8 --cutoff_len 1024
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 8 --cutoff_len 2048
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 8 --cutoff_len 4096

python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 16 --cutoff_len 256
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 16 --cutoff_len 512
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 16 --cutoff_len 1024
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 16 --cutoff_len 2048
python bsz_seqlen_mem_record.py  --device_map "cuda:0" --micro_batch_size 16 --cutoff_len 4096