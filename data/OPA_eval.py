import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from .OPA import gen_blend
import cv2
import matplotlib.pyplot as plt

def gen_composite_images(bg_img, fg_img, fg_msk, trans_list):
    def modify(x, y, w, h):
        if x < 0:
            x = 0
        if x >= bg_img.size[0]:
            x = bg_img.size[0] - 1
        if y < 0:
            y = 0
        if y >= bg_img.size[1]:
            y = bg_img.size[1] - 1
        if w <= 0:
            w = 1
        if h <= 0:
            h = 1
        return x, y, w, h
    bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
    comp_img_list = list()
    comp_msk_list = list()
    bbox_list = list()
    for i in range(trans_list.shape[0]):
        trans = trans_list[i]
        relative_scale, relative_x, relative_y = trans[0], trans[1], trans[2]
        if bg_w / bg_h > fg_w / fg_h:
            fg_w_new, fg_h_new = bg_h * relative_scale * fg_w / fg_h, bg_h * relative_scale
        else:
            fg_w_new, fg_h_new = bg_w * relative_scale, bg_w * relative_scale * fg_h / fg_w
        start_x, start_y, width, height = round((bg_w - fg_w_new) * relative_x), round((bg_h - fg_h_new) * relative_y), round(fg_w_new), round(fg_h_new)
        start_x, start_y, width, height = modify(start_x, start_y, width, height)

        resize_func = transforms.Resize((height, width), interpolation=Image.BILINEAR)
        fg_img_new, fg_msk_new = resize_func(fg_img), resize_func(fg_msk)
        comp_img_arr, bg_img_arr, fg_img_arr, fg_msk_arr = np.array(bg_img), np.array(bg_img), np.array(fg_img_new), np.array(fg_msk_new)
        fg_msk_arr_norm = fg_msk_arr[:, :, np.newaxis].repeat(3, axis=2) / 255.0
        comp_img_arr[start_y:start_y+height, start_x:start_x+width, :] = fg_msk_arr_norm * fg_img_arr + (1.0 - fg_msk_arr_norm) * bg_img_arr[start_y:start_y+height, start_x:start_x+width, :]
        comp_img = Image.fromarray(comp_img_arr.astype(np.uint8)).convert('RGB')
        comp_msk_arr = np.zeros(comp_img_arr.shape[:2])
        comp_msk_arr[start_y:start_y+height, start_x:start_x+width] = fg_msk_arr
        comp_msk = Image.fromarray(comp_msk_arr.astype(np.uint8)).convert('L')

        comp_img_list.append(comp_img)
        comp_msk_list.append(comp_msk)
        bbox_list.append([start_x, start_y, width, height])
    return comp_img_list, comp_msk_list, bbox_list


class Evaluator(object):
    def __init__(self, save_img=False):
        self.count = 0
        self.gts = []
        self.gen_res = []

        self.save_img = save_img

    def start(self, save_dir, epoch, eval_type='test'):
        self.count = 0
        self.gts.clear()
        self.gen_res.clear()

        if self.save_img:
            eval_dir = os.path.join(str(save_dir), eval_type, str(epoch))
            assert (not os.path.exists(eval_dir))
            self.img_sav_dir = os.path.join(eval_dir, 'images')
            self.msk_sav_dir = os.path.join(eval_dir, 'masks')
            self.csv_sav_file = os.path.join(eval_dir, '{}.csv'.format(eval_type))
            os.makedirs(eval_dir)
            os.mkdir(self.img_sav_dir)
            os.mkdir(self.msk_sav_dir)

    @torch.no_grad()
    def update(self, bg_imgs, fg_imgs, fg_msks, outputs, targets):
        def csv_str(fg_id, bg_id, gen_comp_bbox, catnm, gen_file_name):
            return '{},{},"{}",{},-1,images/{}.jpg,masks/{}.png'.format(fg_id, bg_id, gen_comp_bbox, catnm, gen_file_name, gen_file_name)
        if self.save_img:
            for i in range(len(targets)):
                index = targets[i]['index']
                #index = str(self.count + i)
                fg_id = targets[i]['fg_id']
                bg_id = targets[i]['bg_id']
                #bboxes = targets[i]['bboxes'].cpu().numpy().astype(np.float32)
                bg_img_arr = targets[i]['bg_img_arr']
                fg_img_arr = targets[i]['fg_img_arr']
                fg_msk_arr = targets[i]['fg_msk_arr']
                catnm = targets[i]['catnm']

                bg_img = Image.fromarray(bg_img_arr.astype(np.uint8)).convert('RGB')
                fg_img = Image.fromarray(fg_img_arr.astype(np.uint8)).convert('RGB')
                fg_msk = Image.fromarray(fg_msk_arr.astype(np.uint8)).convert('L')
                bboxes_pred = outputs[i]['bboxes'].cpu().numpy().astype(np.float32)

                gen_comp_imgs, gen_comp_msks, gen_comp_bboxs = gen_composite_images(
                    bg_img=bg_img,
                    fg_img=fg_img,
                    fg_msk=fg_msk,
                    trans_list=bboxes_pred
                )

                # ---------- test-------------#
                # for t in range(len(gen_comp_imgs)):
                #     blend_img = gen_comp_imgs[t]
                #     blend_msk = gen_comp_msks[t]
                #     comp_img = targets[i]['comp_imgs'][t]
                #     comp_msk = targets[i]['comp_msks'][t]
                #     plt.figure()
                #     plt.subplot(2, 2, 1)
                #     plt.imshow(comp_img)
                #     plt.subplot(2, 2, 2)
                #     plt.imshow(blend_img)
                #     plt.subplot(2, 2, 3)
                #     plt.imshow(comp_msk)
                #     plt.subplot(2, 2, 4)
                #     plt.imshow(blend_msk)
                #     plt.show()

                for repeat_id in range(len(gen_comp_imgs)):
                    gen_file_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(index, repeat_id, fg_id, bg_id,
                                                                     gen_comp_bboxs[repeat_id][0],
                                                                     gen_comp_bboxs[repeat_id][1],
                                                                     gen_comp_bboxs[repeat_id][2],
                                                                     gen_comp_bboxs[repeat_id][3])
                    gen_comp_imgs[repeat_id].save(os.path.join(self.img_sav_dir, '{}.jpg'.format(gen_file_name)))
                    gen_comp_msks[repeat_id].save(os.path.join(self.msk_sav_dir, '{}.png'.format(gen_file_name)))
                    self.gen_res.append(csv_str(fg_id, bg_id, gen_comp_bboxs[repeat_id], catnm, gen_file_name))


    def summarize(self):
        def csv_title():
            return 'annID,scID,bbox,catnm,label,img_path,msk_path'
        if self.save_img:
            with open(self.csv_sav_file, "w") as f:
                f.write(csv_title() + '\n')
                for line in self.gen_res:
                    f.write(line + '\n')

        res = {'count': self.count}
        return res

def build_evaluator(save_img=False):
    dataset = Evaluator(save_img)
    return dataset