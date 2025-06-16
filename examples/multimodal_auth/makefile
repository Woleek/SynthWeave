train:
	uv run train.py --encoded --data_dir './encoded_data/DeepSpeak_v1_1' --dataset DeepSpeak_v1_1 --batch_size 512 --max_epochs 100 \
	--task binary --fusion CFF --emb_dim 256 --proj_dim 512 --dropout 0.3 --lr 1e-5 \
	--iil_mode whitening --train_strategy FRADE
# --resume
# fusion: "CFF" "AFF" "IAFF" "CAFF" "GFF" "MMD"
# iil_mode: "none", "crossdf", "friday", "whitening"
# train_strategy: "default", "FRADE", "MRDF_ce", "MRDF_marg", "JSD"

evaluate:
	uv run evaluate.py --data_dir './encoded_data/DeepSpeak_v1_1' --dataset DeepSpeak_v1_1 \
	--task binary --fusion CAFF --trained_on DeepSpeak_v1_1

# datasets: DeepSpeak_v1_1 | SWAN_DF

encode_data:
	uv run encode_dataset.py --save_dir './encoded_data'