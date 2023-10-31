import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from PIL import Image
import json
import matplotlib.pyplot as plt


def get_trans_label(bg_img, fg_img, bbox):
    bbox = np.array(bbox).reshape(-1, 4)
    num = bbox.shape[0]
    assert ((bg_img.size[0] > bbox[:, 2]).any() and (bg_img.size[1] > bbox[:, 3]).any())
    bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
    trans_label = np.zeros([num, 3], dtype=np.float32)  # [relative_scale, relative_x, relative_y] in (0,1)^3
    if bg_w / bg_h > fg_w / fg_h:
        trans_label[:, 0] = bbox[:, 3] / bg_h
    else:
        trans_label[:, 0] = bbox[:, 2] / bg_w
    trans_label[:, 1] = bbox[:, 0] / (bg_w - bbox[:, 2])    # relative_x
    trans_label[:, 2] = bbox[:, 1] / (bg_h - bbox[:, 3])    # relative_y
    assert ((trans_label.min() >= 0).any() and (trans_label.max() <= 1).any())
    return trans_label


class ImageDataset(Dataset):
    def __init__(self, dataset_path, img_size=256, mode_type='train'):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.mode_type = mode_type
        self.bg_dir = os.path.join(dataset_path, "background")
        self.fg_dir = os.path.join(dataset_path, "foreground")
        self.fg_msk_dir = os.path.join(dataset_path, "foreground")

        if mode_type == "train_pos_only":
            json_file = os.path.join(dataset_path, "train_set_pos_only_group.json")
        elif mode_type == "test_pos_only":
            json_file = os.path.join(dataset_path, "test_set_pos_only_group.json")
        elif mode_type == "test_pos_single":
            json_file = os.path.join(dataset_path, "test_set_pos_single.json")
        else:
            raise NotImplementedError

        with open(json_file, 'r') as file:
            load_dict = json.load(file)
        self.annotation = load_dict['results']['annotation']

        print("mode_type:", mode_type, "size:", len(self.annotation))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        # get data
        id = self.annotation[index]['id']
        if self.mode_type == "test_pos_single":
            id = str(index)
        fg_id = self.annotation[index]['fg_id']
        bg_id = self.annotation[index]['bg_id']
        catnm = self.annotation[index]['catnm']
        position_list = self.annotation[index]['position']
        label_list = self.annotation[index]['label']
        img_path_list = self.annotation[index]['img_name']
        msk_path_list = self.annotation[index]['mask_name']

        bg_path = os.path.join(self.bg_dir, catnm, "{}.jpg".format(bg_id))
        fg_path = os.path.join(self.fg_dir, catnm, "{}.jpg".format(fg_id))
        fg_msk_path = os.path.join(self.fg_msk_dir, catnm, "mask_{}.jpg".format(fg_id))

        bg_img = Image.open(bg_path).convert('RGB')
        fg_img = Image.open(fg_path).convert('RGB')
        fg_msk = Image.open(fg_msk_path).convert('L')

        bboxes = list()
        labels = list()
        comp_imgs = list()
        comp_msks = list()
        for i in range(len(position_list)):
            p = position_list[i]
            bboxes.append(list(map(int, p[1:-1].split(','))))
            labels.append(int(label_list[i]))

            if 'test' in self.mode_type:
                img_path = os.path.join(self.dataset_path, img_path_list[i])
                msk_path = os.path.join(self.dataset_path, msk_path_list[i])
                comp_img = Image.open(img_path).convert('RGB')
                comp_msk = Image.open(msk_path).convert('L')
                comp_imgs.append(comp_img)
                comp_msks.append(comp_msk)

        bg_feat = self.img_trans_bg(bg_img)
        fg_feat = self.img_trans_fg(fg_img, 'color', bg_img, fg_img)
        fg_msk_feat = self.img_trans_fg(fg_msk, 'gray', bg_img, fg_img)

        fg_bbox = self.get_fg_bbox(bg_img, fg_img)   # [x1, y1, width, height]
        bboxes = get_trans_label(bg_img, fg_img, bboxes)

        labels = torch.tensor(labels, dtype=torch.int32)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        fg_bbox = torch.tensor(fg_bbox, dtype=torch.float32)

        if 'test' in self.mode_type:
            bg_img_arr = np.array(bg_img, dtype=np.uint8)
            fg_img_arr = np.array(fg_img, dtype=np.uint8)
            fg_msk_arr = np.array(fg_msk, dtype=np.uint8)
            sample = {'index': id, 'fg_id': fg_id, 'bg_id': bg_id, 'labels': labels, 'bboxes': bboxes, 'fg_bbox': fg_bbox,
                      'bg_img_arr': bg_img_arr, 'fg_img_arr': fg_img_arr, 'fg_msk_arr': fg_msk_arr, 'catnm': catnm,
                      'comp_imgs': comp_imgs, 'comp_msks': comp_msks}
        else:
            sample = {'index': id, 'fg_id': fg_id, 'bg_id': bg_id, 'labels': labels, 'bboxes': bboxes, 'fg_bbox': fg_bbox}

        return bg_feat, fg_feat, fg_msk_feat, sample

    def img_trans_bg(self, x):
        y = transforms.Resize((self.img_size, self.img_size), interpolation=tf.InterpolationMode.BILINEAR)(x)
        y = transforms.ToTensor()(y)
        return y

    def img_trans_fg(self, x, x_mode, bg_img, fg_img):
        assert (x_mode in ['gray', 'color'])
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        if bg_w / bg_h > fg_w / fg_h:
            y = transforms.Resize((self.img_size, (self.img_size * fg_w * bg_h) // (fg_h * bg_w)), interpolation=tf.InterpolationMode.BILINEAR)(x)
            delta_w = self.img_size - y.size[0]
            delta_w0, delta_w1 = delta_w // 2, delta_w - (delta_w // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((self.img_size, delta_w0), dtype=np.uint8)
                d1 = np.zeros((self.img_size, delta_w1), dtype=np.uint8)
            else:
                d0 = np.zeros((self.img_size, delta_w0, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((self.img_size, delta_w1, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=1)
        else:
            y = transforms.Resize(((self.img_size * fg_h * bg_w) // (fg_w * bg_h), self.img_size), interpolation=tf.InterpolationMode.BILINEAR)(x)
            delta_h = self.img_size - y.size[1]
            delta_h0, delta_h1 = delta_h // 2, delta_h - (delta_h // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((delta_h0, self.img_size), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.img_size), dtype=np.uint8)
            else:
                d0 = np.zeros((delta_h0, self.img_size, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.img_size, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=0)
        y = Image.fromarray(y_arr)
        assert (y.size == (self.img_size, self.img_size))
        y = transforms.ToTensor()(y)
        return y

    def get_fg_bbox(self, bg_img, fg_img):
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        if bg_w / bg_h > fg_w / fg_h:
            y_w = (self.img_size * fg_w * bg_h) // (fg_h * bg_w)
            delta_w0 = (self.img_size - y_w) // 2
            fg_bbox = np.array([delta_w0, 0, y_w, self.img_size]) / self.img_size
        else:
            y_h = (self.img_size * fg_h * bg_w) // (fg_w * bg_h)
            delta_h0 = (self.img_size - y_h) // 2
            fg_bbox = np.array([0, delta_h0, self.img_size, y_h]) / self.img_size
        return fg_bbox   # [x1, y1, width, height]



def build_dataset(mode_type, args):
    dataset = ImageDataset(dataset_path=args.dataset_path, img_size=args.img_size, mode_type=mode_type)
    return dataset


def gen_blend(bg_img, fg_img, fg_msk, fg_bbox, trans):
    device = bg_img.device
    batch_size = len(trans)
    theta = torch.cat((
        1 / (trans[:, 0] + 1e-6), torch.zeros(batch_size).to(device), (1 - 2 * trans[:, 1]) * (1 / (trans[:, 0] + 1e-6) - fg_bbox[:, 2]),
        torch.zeros(batch_size).to(device), 1 / (trans[:, 0] + 1e-6), (1 - 2 * trans[:, 2]) * (1 / (trans[:, 0] + 1e-6) - fg_bbox[:, 3])
    ), dim=0).view(2, 3, batch_size).permute(2, 0, 1).contiguous()
    grid = F.affine_grid(theta, fg_img.size(), align_corners=True)
    fg_img_out = F.grid_sample(fg_img, grid, align_corners=True)
    fg_msk_out = F.grid_sample(fg_msk, grid, align_corners=True)
    comp_out = fg_msk_out * fg_img_out + (1 - fg_msk_out) * bg_img
    return comp_out, fg_msk_out


if __name__ == '__main__':
    dataset = ImageDataset(dataset_path='D:/Dataset/ObjectPlacement/OPA_src', img_size=256, mode_type='test_pos_only')

    bbox_num = []

    for idx in range(len(dataset)):
        bg_img, fg_img, fg_msk, sample = dataset.__getitem__(idx)

        id = sample['index']
        labels = sample['labels']
        bboxes = sample['bboxes']
        fg_bbox = sample['fg_bbox']
        comp_imgs = sample['comp_imgs']
        comp_msks = sample['comp_msks']
        print(bg_img.shape, fg_img.shape, fg_msk.shape, labels.shape, bboxes.shape, fg_bbox.shape, len(comp_imgs), len(comp_msks))

        num = labels.shape[0]

        blend_imgs, blend_msks = gen_blend(bg_img.repeat(num, 1, 1, 1), fg_img.repeat(num, 1, 1, 1), fg_msk.repeat(num, 1, 1, 1), fg_bbox.unsqueeze(0).repeat(num, 1), bboxes)
        print(blend_imgs.shape, blend_msks.shape)


        for j in range(len(comp_imgs)):
            comp_img = comp_imgs[j].resize((256, 256))
            comp_msk = comp_msks[j].resize((256, 256))

            blend_img = blend_imgs[j].permute(1, 2, 0).cpu().detach().numpy().copy() * 255
            blend_msk = blend_msks[j].permute(1, 2, 0).cpu().detach().numpy().copy() * 255

            blend_img = Image.fromarray(blend_img.astype(np.uint8)).convert('RGB')
            blend_msk = Image.fromarray(blend_msk[:, :, 0].astype(np.uint8)).convert('L')

            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(comp_img)
            plt.subplot(2, 2, 2)
            plt.imshow(blend_img)
            plt.subplot(2, 2, 3)
            plt.imshow(comp_msk)
            plt.subplot(2, 2, 4)
            plt.imshow(blend_msk)
            plt.show()

        # bg_img = bg_img.permute(1, 2, 0).cpu().detach().numpy().copy()
        # fg_img = fg_img.permute(1, 2, 0).cpu().detach().numpy().copy()
        # fg_mask = fg_mask.permute(1, 2, 0).cpu().detach().numpy().copy()
        # bboxs = bboxs.cpu().detach().numpy().copy()
    #     # boxes_trans = boxes_trans.cpu().detach().numpy().copy()
    #     # # comp_img = comp_img.permute(1, 2, 0).cpu().detach().numpy().copy()
    #     # # comp_mask = comp_mask.permute(1, 2, 0).cpu().detach().numpy().copy()
    #     #
    #     #
    #     bg_img = (bg_img * 255).astype(np.uint8)[..., [2, 1, 0]]
    #     fg_img = (fg_img * 255).astype(np.uint8)[..., [2, 1, 0]]
    #     fg_mask = (fg_mask * 255).astype(np.uint8)
    #     # # comp_img = (comp_img * 255).astype(np.uint8)[..., [2, 1, 0]]
    #     # # comp_mask = (comp_mask * 255).astype(np.uint8)
    #     #
    #     bg_h, bg_w, _ = bg_img.shape
    #     fg_h, fg_w, _ = fg_img.shape
    #     for i in range(labels.shape[0]):
    #         print(labels[i])
    #         cx = int(bboxs[i, 0] * bg_img.shape[1])
    #         cy = int(bboxs[i, 1] * bg_img.shape[0])
    #         r = bboxs[i, 2]
    #         fg_w_new = int(fg_w * r)
    #         fg_h_new = int(fg_h * r)
    #         fg_img_scale = cv2.resize(fg_img, (fg_w_new, fg_h_new), interpolation=cv2.INTER_LINEAR)
    #         fg_mask_scale = cv2.resize(fg_mask, (fg_w_new, fg_h_new), interpolation=cv2.INTER_LINEAR)
    #         x1 = cx - fg_w_new // 2
    #         y1 = cy - fg_h_new // 2
    #         x2 = x1 + fg_w_new
    #         y2 = y1 + fg_h_new
    #
    #         x1 = max(0, x1)
    #         y1 = max(0, y1)
    #         x2 = min(x2, bg_w)
    #         y2 = min(y2, bg_h)
    #
    #         mask_new = np.zeros([bg_h, bg_w, 1], np.uint8)
    #         mask_new[y1:y2, x1:x2, 0] = fg_mask_scale[0:y2-y1, 0:x2-x1]
    #
    #         fg_img_new = np.zeros([bg_h, bg_w, 3], np.uint8)
    #         fg_img_new[y1:y2, x1:x2, :] = fg_img_scale[0:y2-y1, 0:x2 - x1, :]
    #
    #         mask_new_01 = mask_new / 255
    #         img_new = fg_img_new * mask_new_01 + bg_img * (1 - mask_new_01)
    #         img_new = img_new.astype(np.uint8)
    #         bg_img = bg_img.astype(np.uint8)
    #
    #         blend_img = blend_imgs[i].permute(1, 2, 0).cpu().detach().numpy().copy()
    #         blend_mask = blend_masks[i].permute(1, 2, 0).cpu().detach().numpy().copy()
    #         blend_img = (blend_img * 255).astype(np.uint8)[..., [2, 1, 0]]
    #         blend_mask = (blend_mask * 255).astype(np.uint8)
    #
    #
    #         cv2.circle(img_new, (cx, cy), color=(0, 0, 255), thickness=10, radius=1)
    #         cv2.imshow("img_new", img_new)
    #         cv2.imshow("fg_img", fg_img)
    #         cv2.imshow("bg_img", bg_img)
    #         cv2.imshow("mask_new", mask_new)
    #         cv2.imshow("blend_img", blend_img)
    #         cv2.imshow("blend_mask", blend_mask)
    #         cv2.waitKey(0)