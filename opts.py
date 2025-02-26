import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='data/ClipShots/Videos', required=False, type=str,
                        help='Root directory path of data')
    parser.add_argument('--image_list_path', default='data/data_list/choose_deepSBD.txt', type=str)
    parser.add_argument('--result_dir', default='results', type=str, help='Result directory path')
    parser.add_argument('--total_iter', default=200, type=int)
    parser.add_argument('--n_classes', default=3, type=int, help='Number of classes')
    parser.add_argument('--shuffle', action='store_true', help="shuffle the dataset")

    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')

    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal durationof inputs')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
    parser.add_argument('--lr_patience', default=10, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_val=False)
    parser.set_defaults(test=False)
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--norm_value', default=1, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--model', default='alexnet', type=str, help='(resnet-18 | alexnet')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--manual_seed', default=16, type=int)

    parser.add_argument('--spatial_size', type=int, default=128)

    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--test_list_path', default='data/ClipShots/Video_lists/choose_test.txt', type=str,
                        help='test list path')
    parser.add_argument('--gt_dir', default='data/ClipShots/Annotations/test.json', type=str,
                        help='directory contains ground truth for test set')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--test_subdir', type=str, default='test', help='subdirectroy for training set')
    parser.add_argument('--train_subdir', type=str, default='choose_train', help='subdirectory for testing set')
    args = parser.parse_args()

    return args


def parse_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='data/ClipShots/Videos', required=False, type=str, help='Root directory path of data')
    parser.add_argument('--test_list_path', default='data/ClipShots/Video_lists/choose_test.txt', type=str, help='test list path')
    parser.add_argument('--model', default='alexnet', type=str)
    parser.add_argument('--weights', default='results/model_final.pth', type=str)
    parser.add_argument('--result_dir', default='results', type=str)
    parser.add_argument('--n_classes', default=3, type=int,
                        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--norm_value', default=1, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--sample_duration', default=16, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--gt_dir',default='data/ClipShots/Annotations/test.json', type=str)
    parser.add_argument('--spatial_size', type=int, default=112)
    parser.add_argument('--test_subdir', type=str, default='test')
    args = parser.parse_args()
    return args


def parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path', type=str)
    parser.add_argument('--gt_dir', type=str)
    args = parser.parse_args()
    return args
