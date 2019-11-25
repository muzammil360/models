# cd ..
# python deeplab/vis.py --checkpoint_dir 'deeplab/ckpts/deeplabv3_pascal_trainval' \
# --vis_logdir 'deeplab/logs' \
# --dataset_dir '/home/ubuntu/datasets/footballers1' \
# --model_variant 'xception_65' \
# --output_stride 8 \
# --atrous_rates 12 \
# --atrous_rates 24 \
# --atrous_rates 36


python infer.py --input_dir '/home/ubuntu/datasets/footballers1' \
--output_dir '/home/ubuntu/datasets/footballers1_out' \
--model_path '/home/ubuntu/github/tf-models/research/deeplab/ckpts/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz'


