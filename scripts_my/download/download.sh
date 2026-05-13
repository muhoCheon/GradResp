# sh ./scripts_my/download/download.sh

# download the up-to-date benchmarks and checkpoints
# provided by OpenOOD v1.5
python ./scripts_my/download/download.py \
	--contents 'datasets' 'checkpoints' \
	--datasets 'ood_v1.5' \
	--checkpoints 'ood_v1.5' \
	--save_dir './data' './results' \
	--dataset_mode 'benchmark'
