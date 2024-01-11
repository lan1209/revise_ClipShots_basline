log_dir=results
gpu_num=1
python main_cls.py \
--root_dir data/ClipShots/Videos \
--image_list_path data/data_list/choose_deepSBD.txt \
--result_dir results \
--model resnet \
--n_classes 3 --batch_size 20 --n_threads 16 \
--sample_duration 16 \
--learning_rate 0.001 \
--gpu_num $gpu_num \
--manual_seed 16 \
--shuffle \
--spatial_size 112 \
--pretrain_path kinetics_pretrained_model/Alexnet-finals.pth \
--gt_dir data/ClipShots/Annotations/test.json \
--test_list_path data/ClipShots/Video_lists/choose_test.txt \
--total_iter 300000 \
--auto_resume |tee $log_dir/test.log