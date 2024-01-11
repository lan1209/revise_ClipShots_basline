import json

import pandas as pd

from opts import parse_eval_args


def if_overlap(begin1, end1, begin2, end2):
    if begin1 > begin2:
        begin1, end1, begin2, end2 = begin2, end2, begin1, end1

    return end1 >= begin2


def get_union_cnt(set1, set2):
    cnt = 0
    for begin, end in set1:
        for _begin, _end in set2:
            if if_overlap(begin, end, _begin, _end):
                cnt += 1
                break
    return cnt


def recall_pre_f1(a, b, c):
    recall = a / b if b != 0 else 0
    precison = a / c if c != 0 else 0
    if recall == 0 or precison == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precison / (recall + precison)
    return recall, precison, f1


def eval(predict_path, gt_path):
    predicts = json.load(open(predict_path))
    gts = json.load(open(gt_path))

    cut_correct_sum = 0
    gradual_correct_sum = 0
    all_correct_sum = 0
    gt_cut_sum = 0
    gt_gradual_sum = 0
    predict_cut_sum = 0
    predict_gradual_sum = 0
    for videoname, labels in gts.items():
        if videoname in predicts:
            _gts = gts[videoname]['transitions']
            gt_cuts = [(begin, end) for begin, end in _gts if end - begin == 1]
            gt_graduals = [(begin, end) for begin, end in _gts if end - begin > 1]

            _predicts = predicts[videoname]
            predicts_cut = _predicts['cut']
            predicts_gradual = _predicts['gradual']

            cut_correct = get_union_cnt(gt_cuts, predicts_cut)
            gradual_correct = get_union_cnt(gt_graduals, predicts_gradual)
            all_correct = get_union_cnt(_gts, predicts_cut + predicts_gradual)

            cut_correct_sum += cut_correct
            gradual_correct_sum += gradual_correct
            all_correct_sum += all_correct

            gt_cut_sum += len(gt_cuts)
            gt_gradual_sum += len(gt_graduals)

            predict_cut_sum += len(predicts_cut)
            predict_gradual_sum += len(predicts_gradual)
        # else:
        #     print(videoname)
        #     raise Exception()

    # print("group\trecall\tprecision\tf1score")
    # print("cut\t\t{}\t\t{}\t{}".format(*recall_pre_f1(cut_correct_sum, gt_cut_sum, predict_cut_sum)))
    # print("gradual\t{}\t{}\t{}".format(*recall_pre_f1(gradual_correct_sum, gt_gradual_sum, predict_gradual_sum)))
    # print("all\t\t{}\t\t{}\t{}".format(
    #     *recall_pre_f1(all_correct_sum, gt_cut_sum + gt_gradual_sum, predict_cut_sum + predict_gradual_sum)))

    data = [
        [*recall_pre_f1(cut_correct_sum, gt_cut_sum, predict_cut_sum)],
        [*recall_pre_f1(gradual_correct_sum, gt_gradual_sum, predict_gradual_sum)],
        [*recall_pre_f1(all_correct_sum, gt_cut_sum + gt_gradual_sum, predict_cut_sum + predict_gradual_sum)]
    ]
    columns = ['recall', 'precision', 'f1score']
    index = ['cut', 'gradual', 'all']
    df = pd.DataFrame(data = data, index = index, columns = columns)
    print(df)


if __name__ == "__main__":
    opt = parse_eval_args()
    eval(opt.predict_path, opt.gt_dir)
