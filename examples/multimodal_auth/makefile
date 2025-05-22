train:
	uv run train.py --encoded --data_dir './encoded_data/DeepSpeak_v1_1' --dataset DeepSpeak_v1_1 --batch_size 128 --max_epochs 20 \
	--task binary --fusion CAFF --emb_dim 512 --proj_dim 1024 \
	--contr_loss_mod_w 0.1 \
	--contr_loss_fused_w 0.1
# --resume

evaluate:
	uv run evaluate.py --data_dir './encoded_data/DeepSpeak_v1_1' --dataset DeepSpeak_v1_1 \
	--task binary --fusion CAFF --trained_on DeepSpeak_v1_1

# datasets: DeepSpeak_v1_1 | SWAN_DF

encode_data:
	uv run encode_dataset.py --save_dir './encoded_data' --resume