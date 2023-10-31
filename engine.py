# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
import torch
import numpy as np
import util.misc as utils

def train_one_epoch(model, criterion, environment, data_loader, optimizer,
                    num_steps: int, device: torch.device, batch_size: int, epoch: int, max_norm: float,
                    display_freq: int = 100):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for bg_imgs, fg_imgs, fg_msks, targets in metric_logger.log_every(data_loader, display_freq, header):
        bg_imgs = bg_imgs.to(device)
        fg_imgs = fg_imgs.to(device)
        fg_msks = fg_msks.to(device)
        for t in targets:
            for k, v in t.items():
                if k in ['labels', 'bboxes', 'fg_bbox']:
                    t[k] = v.to(device)
        fg_bboxes = torch.cat([t['fg_bbox'].unsqueeze(0) for t in targets], dim=0)
        bs = fg_imgs.shape[0]

        # loss
        loss_dict = {'loss_policy': 0, 'loss_value': 0, 'loss_entropy': 0}
        # start batch
        for b in range(bs):
            bg_img = bg_imgs[b:b + 1]
            fg_img = fg_imgs[b:b + 1]
            fg_msk = fg_msks[b:b + 1]
            fg_bbox = fg_bboxes[b:b + 1]

            trans_bbox = torch.rand(1, 3).to(device)
            hx = torch.zeros(1, model.hidden_dim).to(device)
            cx = torch.zeros(1, model.hidden_dim).to(device)

            state = environment.reset(bg_img, fg_img, fg_msk, fg_bbox, trans_bbox)
            done = environment.dones
            values = []
            log_probs = []
            rewards = []
            entropies = []
            for step in range(num_steps):
                value, logit, (hx, cx) = model((state, (hx, cx), trans_bbox))

                prob = torch.softmax(logit, dim=-1)

                log_prob = torch.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)

                action = prob.multinomial(num_samples=1).detach()
                log_prob = log_prob.gather(1, action)

                state, reward, trans_bbox, done = environment.step(action)

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

            R = torch.zeros(1, 1).to(device)
            if not done:
                value, _, _ = model((state, (hx, cx), trans_bbox))
                R = value.detach()
            values.append(R)

            loss = criterion(R, rewards, values, log_probs, entropies)
            for k in loss.keys():
                loss_dict[k] = loss_dict[k] + loss[k]

        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] / bs

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, environment, data_loader, evaluator, device, output_dir, eval_type,
             max_steps: int, num_select: int, epoch: int, display_freq: int = 100):
    model.eval()
    criterion.eval()

    evaluator.start(output_dir, epoch, eval_type)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for bg_imgs, fg_imgs, fg_msks, targets in metric_logger.log_every(data_loader, display_freq, header):
        bg_imgs = bg_imgs.to(device)
        fg_imgs = fg_imgs.to(device)
        fg_msks = fg_msks.to(device)
        for t in targets:
            for k, v in t.items():
                if k in ['labels', 'bboxes', 'fg_bbox']:
                    t[k] = v.to(device)
        fg_bboxes = torch.cat([t['fg_bbox'].unsqueeze(0) for t in targets], dim=0)

        results = []
        for b in range(len(targets)):
            bg_img = bg_imgs[b:b + 1]
            fg_img = fg_imgs[b:b + 1]
            fg_msk = fg_msks[b:b + 1]
            fg_bbox = fg_bboxes[b:b + 1]

            num_topk = num_select if num_select > 0 else targets[b]['bboxes'].shape[0]
            each_results = []
            for i in range(num_topk):
                trans_bbox = torch.rand(1, 3).to(device)
                observation = environment.reset(bg_img, fg_img, fg_msk, fg_bbox, trans_bbox)

                hx = torch.zeros(1, model.hidden_dim).to(device)
                cx = torch.zeros(1, model.hidden_dim).to(device)

                done, not_finish = False, False
                tot_reward = 0
                tot_steps = 0
                while not done:
                    with torch.no_grad():
                        value_ts, logit_ts, (hx, cx) = model((observation, (hx, cx), trans_bbox))
                        prob = torch.softmax(logit_ts, dim=-1)
                        action_ts = torch.argmax(prob, dim=-1, keepdim=True).detach()

                    observation, reward, trans_bbox, done = environment.step(action_ts)

                    tot_reward += reward
                    tot_steps += 1
                    if tot_steps > max_steps - 1:
                        not_finish = True
                        break
                (trans_bbox, score_diff) = environment.cur_status()
                each_results.append(trans_bbox.squeeze(0))

                metric_logger.update(not_finish=int(not_finish))

            each_results = torch.stack(each_results, dim=0)
            results.append({'bboxes': each_results})

        evaluator.update(bg_imgs, fg_imgs, fg_msks, results, targets)

    test_res = evaluator.summarize()
    metric_logger.update(**test_res)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return stats
