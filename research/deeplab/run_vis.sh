cd ..

python deeplab/vis.py --checkpoint_dir 'deeplab/ckpts/deeplabv3_pascal_trainval' \
--vis_logdir 'deeplab/logs' \
--dataset_dir '/home/ubuntu/datasets/footballers1' \
--model_variant 'xception_65' \
--output_stride 8 \
--atrous_rates 12 \
--atrous_rates 24 \
--atrous_rates 36


