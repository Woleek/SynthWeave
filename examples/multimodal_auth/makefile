train:
	uv run train.py --encoded --data_dir './encoded_data/DeepSpeak_v1_1' --batch_size 128 --max_epochs 30 \
	--task binary --fusion CAFF --emb_dim 512 --proj_dim 1024 \
	--contr_loss_mod_w 0.05 \
	--contr_loss_fused_w 0.1 \
	--aam_loss_w 0.0 --learnable_loss_weights
# --resume

evaluate:
	uv run evaluate.py --data_dir './encoded_data/DeepSpeak_v1_1' \
	--task binary --fusion MMD

encode_data:
	uv run encode_dataset.py --save_dir './encoded_data' --resume