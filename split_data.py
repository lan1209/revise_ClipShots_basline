import os
import numpy as np
import shutil


def random_choose_video(video_path):
    video_name = os.listdir(video_path)
    video_num = np.random.choice(3657, 100, replace=False)

    choose_video_name = []
    for num in video_num:
        choose_video_name.append(video_name[num])

    for name in choose_video_name:
        path = os.path.join(video_path, name)
        shutil.copyfile(path, os.path.join('data/ClipShots/Videos/choose_train', name))


def random_choose_test_video(test_path):
    video_name = []
    with open(test_path, 'r') as f:
        for line in f.readlines():
            video_name.append(line)

    choose_video_name = []
    video_num = np.random.choice(500, 20, replace=False)
    for num in video_num:
        choose_video_name.append(video_name[num])

    with open('data/ClipShots/Video_lists/choose_test.txt', 'w') as f:
        for name in choose_video_name:
            f.writelines(name)


def spilt_data(video_path):
    video_name = os.listdir(video_path)

    with open('./data/data_list/deepSBD.txt', 'r') as f:
        video_name_label = f.readlines()

    choose_video_name_label = []
    for video_information in video_name_label:
        name = video_information.split(' ')[0]
        if name in video_name:
            choose_video_name_label.append(video_information)

    with open('./data/data_list/choose_deepSBD.txt', 'w') as f:
        for name_label in choose_video_name_label:
            f.writelines(name_label)


if __name__ == '__main__':
    all_video_path = 'data/ClipShots/Videos/train'
    choose_train_path = 'data/ClipShots/Videos/choose_train'
    tst_path = 'data/ClipShots/Video_lists/test.txt'

    """
        代码不健壮，勿运行
    """
    # random_choose_video(all_video_path)
    # spilt_data(choose_train_path)
    # random_choose_test_video(tst_path)
    spilt_data(choose_train_path)
