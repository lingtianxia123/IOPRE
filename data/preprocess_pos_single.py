import argparse
import csv
import os
from collections import defaultdict
import json
from data.preprocess import get_catnm_list


def preprocess(data_root, input_file, output_file, fg_id_to_catnm, bg_catnm_idset):
    # Find all combinations of foreground and background.
    fg_bg_id_list = []
    fg_id_list = []
    bg_id_list = []
    catnm_list = []
    position_list = []
    label_list = []
    img_name_list = []
    mask_name_list = []
    csv_data = csv.DictReader(open(os.path.join(data_root, input_file), 'r'))
    for i, row in enumerate(csv_data):
        fg_id = row['fg_id']
        bg_id = row['bg_id']
        fg_bg_id = fg_id + '_' + bg_id

        assert (fg_id in fg_id_to_catnm)
        catnm = fg_id_to_catnm[fg_id]
        assert (catnm in bg_catnm_idset and bg_id in bg_catnm_idset[catnm])
        catnm_list.append(catnm)

        fg_bg_id_list.append(fg_bg_id)
        fg_id_list.append(fg_id)
        bg_id_list.append(bg_id)
        position_list.append([row['position']])
        label_list.append([row['label']])
        img_name_list.append([row['img_name'][8:]])
        mask_name_list.append([row['mask_name'][8:]])

    # Save
    anno_total = defaultdict(list)
    for i in range(len(fg_bg_id_list)):
        dict = defaultdict(list)
        dict["id"] = str(i)
        dict["fg_id"] = fg_id_list[i]
        dict["bg_id"] = bg_id_list[i]
        dict["catnm"] = catnm_list[i]
        dict["position"] = position_list[i]
        dict["label"] = label_list[i]
        dict["img_name"] = img_name_list[i]
        dict["mask_name"] = mask_name_list[i]

        if '1' in dict['label']:
            anno_total["annotation"].append(dict)

    train_dict = {
        'version': "1.0",
        'results': anno_total,
        'explain': {
            'used': True,
            'details': "OPA dataset",
        }
    }
    json_str = json.dumps(train_dict, indent=4)
    with open(os.path.join(data_root, output_file), 'w') as json_file:
        json_file.write(json_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/media/lingtianxia/Data/Dataset/ObjectPlacement/OPA", help="dataset root")
    opt = parser.parse_args()
    fg_id_to_catnm, bg_catnm_idset = get_catnm_list(opt.data_root)
    #preprocess(opt.data_root, 'train_set.csv', 'train_set_pos_group.json', fg_id_to_catnm, bg_catnm_idset)
    preprocess(opt.data_root, 'test_set.csv', 'test_set_pos_single.json', fg_id_to_catnm, bg_catnm_idset)