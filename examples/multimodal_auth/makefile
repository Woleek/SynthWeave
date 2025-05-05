train_binary:
	uv run train.py --encoded --data_dir './encoded_data/DeepSpeak_v1_1' --batch_size 128 --max_epochs 15 \
	--task binary --fusion CAFF --emb_dim 512 --proj_dim 1024

train_fine_grained:
	uv run train.py --encoded --data_dir './encoded_data/DeepSpeak_v1_1' --task 'fine-grained' --batch_size 128
# --resume

encode_data:
	uv run encode_dataset.py --save_dir './encoded_data' --resume

evaluate:
	uv run evaluate.py