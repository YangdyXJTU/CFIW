CUDA_VISIBLE_DEVICES=1 \
python main.py \
--model 'CFIW' \
--batch-size 32 \
--epochs 200 \
--output_dir './trained_results/' \
--data-path '/your_dataset_path/'
