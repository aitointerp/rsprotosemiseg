import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from generalframeworks.utils import simplex

class Attention_Threshold_Loss(nn.Module):
    def __init__(self, strong_threshold):
        super(Attention_Threshold_Loss, self).__init__()
        self.strong_threshold = strong_threshold

    def forward(self, pred: torch.Tensor, pseudo_label: torch.Tensor, logits: torch.Tensor):
        batch_size = pred.shape[0]
        valid_mask = (pseudo_label >= 0).float() # only count valid pixels (class)
        weighting = logits.view(batch_size, -1).ge(self.strong_threshold).sum(-1) / (valid_mask.view(batch_size, -1).sum(-1)) # May be nan if the whole target is masked in cutout
        #self.tmp_valid_num = logits.ge(self.strong_threshold).view(logits.shape[0], -1).float().sum(-1).mean(0)
        # weight represent the proportion of valid pixels in this batch
        loss = F.cross_entropy(pred, pseudo_label, reduction='none', ignore_index=-1) # pixel-wise
        weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
        # weight torch.size([4]) -> weight[:, None, None] torch.size([4, 1, 1]) for broadcast to multiply the weight to the corresponding class
        # torch.masked_select to select loss > 0 only leaved 
        
        return weighted_loss


class Mix_Loss_single(nn.Module):
    # For single GPU users
    def __init__(self, num_queries, num_negatives, temp=0.5, mean=False, strong_threshold=0.97):
        super(Mix_Loss_single, self).__init__()
        self.temp = temp
        self.mean = mean
        self.num_queries = num_queries
        self.num_negatives = num_negatives
        self.strong_threshold = strong_threshold

    def forward(self, weights, mu, sigma, label, mask, prob):
        batch_size, num_feat, mu_w, mu_h = mu.shape
        num_segments = label.shape[1]  # 21
        valid_pixel_all = label * mask

        mu = mu.permute(0, 2, 3, 1)
        sigma = sigma.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        #print(weights.shape)

        mu_all_list = []
        sigma_all_list = []
        weight_all_list = []
        mu_hard_list = []
        sigma_hard_list = []
        weight_hard_list = []
        num_list = []
        proto_mu_list = []
        proto_sigma_list = []
        proto_weight_list =[]

        for i in range(num_segments):
            valid_pixel = valid_pixel_all[:, i]
            if valid_pixel.sum() == 0:
                continue
            #print(sigma[valid_pixel.bool()].shape)
            prob_seg = prob[:, i, :, :]
            rep_mask_hard = (prob_seg < self.strong_threshold) * valid_pixel.bool()

            with torch.no_grad():
                # 使用混合高斯分布的权重对均值进行加权平均
                #proto_mu_ = torch.sum((mu[valid_pixel.bool()] * weights[valid_pixel.bool()]), dim=0, keepdim=True)
                #proto_sigma_ = 1 / torch.sum((1 / sigma[valid_pixel.bool()] * weights[valid_pixel.bool()]), dim=0, keepdim=True)
                proto_sigma_ = 1 / torch.sum((1 / sigma[valid_pixel.bool()]*weights[valid_pixel.bool()]), dim=0, keepdim=True)
                proto_mu_ = torch.sum((proto_sigma_ / sigma[valid_pixel.bool()]*weights[valid_pixel.bool()]) \
                                      * mu[valid_pixel.bool()], dim=0, keepdim=True)
                proto_weight_ = 1 / torch.sum((1 / weights[valid_pixel.bool()]), dim=0, keepdim=True)

                proto_mu_list.append(proto_mu_)
                proto_sigma_list.append(proto_sigma_)
                proto_weight_list.append(proto_weight_)

            mu_all_list.append(mu[valid_pixel.bool()])
            sigma_all_list.append(sigma[valid_pixel.bool()])
            weight_all_list.append(weights[valid_pixel.bool()])
            mu_hard_list.append(mu[rep_mask_hard])
            sigma_hard_list.append(sigma[rep_mask_hard])
            weight_hard_list.append(weights[rep_mask_hard])
            num_list.append(int(valid_pixel.sum().item()))

        if len(num_list) <= 1:
            return torch.tensor(0.0)
        else:
            prcl_loss = torch.tensor(0.0)
            proto_mu = torch.cat(proto_mu_list)  # [c]
            proto_sigma = torch.cat(proto_sigma_list)
            proto_weight = torch.cat(proto_weight_list)
            valid_num = len(num_list)
            seg_len = torch.arange(valid_num)

            for i in range(valid_num):
                if len(mu_hard_list[i]) > 0:
                    sample_idx = torch.randint(len(mu_hard_list[i]), size=(self.num_queries,))
                    anchor_mu = mu_hard_list[i][sample_idx]
                    anchor_sigma = sigma_hard_list[i][sample_idx]
                    anchor_weight = weight_hard_list[i][sample_idx]
                else:
                    continue

                with torch.no_grad():
                    id_mask = torch.cat(([seg_len[i:], seg_len[: i]]))
                    # 传递混合高斯分布的权重给 mutual_likelihood_score 函数
                    proto_sim = mutual_likelihood_score(proto_mu[id_mask[0].unsqueeze(0)],
                                                        proto_mu[id_mask[1:]],
                                                        proto_sigma[id_mask[0].unsqueeze(0)],
                                                        proto_sigma[id_mask[1:]],
                                                        proto_weight[id_mask[0].unsqueeze(0)],
                                                        proto_weight[id_mask[1:]])
                    proto_prob = torch.softmax(proto_sim / self.temp, dim=0)
                    negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                    samp_class = negative_dist.sample(sample_shape=[self.num_queries, self.num_negatives])
                    samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)
                    negative_num_list = num_list[i + 1:] + num_list[: i]
                    negative_index = negative_index_sampler(samp_num, negative_num_list)
                    negative_mu_all = torch.cat(mu_all_list[i + 1:] + mu_all_list[: i])
                    negative_sigma_all = torch.cat(sigma_all_list[i + 1:] + sigma_all_list[: i])
                    negative_weight_all = torch.cat(weight_all_list[i + 1:] + weight_all_list[: i])
                    negative_mu = negative_mu_all[negative_index].reshape(self.num_queries, self.num_negatives,
                                                                          num_feat)
                    negative_sigma = negative_sigma_all[negative_index].reshape(self.num_queries, self.num_negatives,
                                                                                num_feat)
                    negative_weight = negative_weight_all[negative_index].reshape(self.num_queries, self.num_negatives,
                                                                                num_feat)
                    positive_mu = proto_mu[i].unsqueeze(0).unsqueeze(0).repeat(self.num_queries, 1, 1)
                    positive_sigma = proto_sigma[i].unsqueeze(0).unsqueeze(0).repeat(self.num_queries, 1, 1)
                    positive_weight = proto_weight[i].unsqueeze(0).unsqueeze(0).repeat(self.num_queries, 1, 1)
                    all_mu = torch.cat((positive_mu, negative_mu), dim=1)
                    all_sigma = torch.cat((positive_sigma, negative_sigma), dim=1)
                    all_weight = torch.cat((positive_weight, negative_weight), dim=1)

                logits = mutual_likelihood_score(anchor_mu.unsqueeze(1), all_mu, anchor_sigma.unsqueeze(1), all_sigma, anchor_weight.unsqueeze(1),all_weight)
                prcl_loss = prcl_loss + F.cross_entropy(logits / self.temp, torch.zeros(self.num_queries).long().cuda())

            return prcl_loss / valid_num


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[: j]),
                                                high=sum(seg_num_list[: j + 1]),
                                                size=int(samp_num[i, j])).tolist()

    return negative_index


#### MLS ####
def mutual_likelihood_score(mu_0, mu_1, sigma_0, sigma_1, weights_0, weights_1):
    mu_0 = F.normalize(mu_0, dim=-1)
    mu_1 = F.normalize(mu_1, dim=-1)

    up = (mu_0*weights_0 - mu_1*weights_1) ** 2
    down = sigma_0*weights_0 + sigma_1*weights_1
    mls = -0.5 * (up / down + torch.log(down)).mean(-1)

    return mls
