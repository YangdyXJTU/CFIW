python predict_class.py \
    --model 'CFIW' \
    --resume './trained_weights/CFIW_best_checkpoint.pth' \
    --test-dir './sample_imgs' \
    --output-dir './output/' \
    --device 'cuda:0'