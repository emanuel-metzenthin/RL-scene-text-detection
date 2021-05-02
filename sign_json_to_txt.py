

import json

# sign_gt = json.loads(open('../data/600_3_signs_3_words/validation.json').read())
#
# for i, gt in enumerate(sign_gt):
#     if i < 1000:
#         gt_file = open(f'../data/600_3_signs_3_words/gt/gt_img_{i}.txt', 'w+')
#
#         for bbox in gt['bounding_boxes']:
#             gt_file.write(f"{','.join(map(str, bbox))},\n")
#
#         gt_file.close()

for i in range(1000):
    file = open(f'sign_res/res_img_{i}.txt', 'r')
    w_file = open(f'sign_res2/res_img_{i}.txt', 'w+')

    for bb in file.readlines():
        float_bb = list(map(float, bb.split(',')))
        int_bb = list(map(int, float_bb))
        str_bb = list(map(str, int_bb))
        w_file.write(f"{','.join(str_bb)}\n")
    file.close()
    w_file.close()