import numpy as np
import torch
import torch.nn.functional as F

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


def command2action(command_ids, trans_bbox, terminals, unit=0.02):
    batch_size = len(command_ids)
    for i in range(batch_size):
        if terminals[i] == 1:
            continue
        if command_ids[i, 0] == 0:
            trans_bbox[i, 0] += unit
        elif command_ids[i, 0] == 1:
            trans_bbox[i, 0] -= unit
        elif command_ids[i, 0] == 2:
            trans_bbox[i, 1] += unit
        elif command_ids[i, 0] == 3:
            trans_bbox[i, 1] -= unit
        elif command_ids[i, 0] == 4:
            trans_bbox[i, 2] += unit
        elif command_ids[i, 0] == 5:
            trans_bbox[i, 2] -= unit
        elif command_ids[i, 0] == 6:
            terminals[i] = 1
        else:
            raise NameError('undefined command type !!!')

        trans_bbox = torch.clamp(trans_bbox, min=0, max=1)
        # if trans_bbox[i, 0] <= 0.05 or trans_bbox[i, 0] >= 0.95:  # scale
        # if trans_bbox[i, 0] <= 0.03:
        #     terminals[i] = 1

    return trans_bbox, terminals


class Environment(object):
    def __init__(self, args, scorer, device, batch_size=1):
        self.scorer = scorer
        self.bbox_unit = args.bbox_unit
        self.base_reward = args.base_reward
        self.step_neg_reward_scale = args.step_neg_reward_scale
        self.neg_reward = args.neg_reward
        self.device = device
        self.batch_size = batch_size

    def reset(self, bg_img, fg_img, fg_msk, fg_bbox, init_bbox):
        self.bg_img = bg_img
        self.fg_img = fg_img
        self.fg_msk = fg_msk
        self.fg_bbox = fg_bbox
        self.trans_bbox = init_bbox
        blend_img, blend_msk = gen_blend(self.bg_img, self.fg_img, self.fg_msk, self.fg_bbox, self.trans_bbox)

        out = self.scorer(blend_img, blend_msk)
        self.origin_score = torch.softmax(out['label'], dim=-1).cpu().numpy()[..., 1][0]  # [1, 1]
        feat = out['feat'].detach()  # [1, 512]
        self.cur_score = self.origin_score

        self.dones = np.repeat([0], self.batch_size, axis=0)
        self.steps = 0
        return feat

    def step(self, action):
        self.steps += 1
        self.trans_bbox, self.dones = command2action(action, self.trans_bbox, self.dones, self.bbox_unit)
        blend_img, blend_msk = gen_blend(self.bg_img, self.fg_img, self.fg_msk, self.fg_bbox, self.trans_bbox)

        out = self.scorer(blend_img, blend_msk)
        score = torch.softmax(out['label'], dim=-1).cpu().numpy()[..., 1][0]  # [1, 1]
        feat = out['feat'].detach()  # [1, 512]

        reward = np.sign(score - self.cur_score) * self.base_reward
        self.cur_score = score
        reward -= self.step_neg_reward_scale * self.steps
        reward = max(min(reward, self.base_reward), -self.base_reward)

        reward -= self.neg_reward if self.trans_bbox[0, 0] <= 0.05 or self.trans_bbox[0, 0] >= 0.95 else 0
        if np.any(self.dones == 1):
            reward = 0

        return feat, reward, self.trans_bbox, np.all(self.dones == 1)

    def cur_status(self):
        return self.trans_bbox, self.cur_score - self.origin_score