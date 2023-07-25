python3 train.py --lr_scheduler cosine --optimizer adam --weight_decay 1e-5 \
--CUDA_VISIBLE_DEVICES 0,1 --model resnest \
--crop_size 32,48,48 --num_workers 16 \
--save_path /data/dk/exp/luna16_fp_reduction \
--data_path /data/dk/luna16_crops --batch 160 \
--exp_name fp_reduction_exp01