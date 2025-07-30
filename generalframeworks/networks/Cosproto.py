import torch
import torch.nn as nn
import numpy as np
import pickle

class CosProto_Module(nn.Module):
    def __init__(self, in_planes, num_classes, num_micro_proto, init_proto_path, proto_unpdate_momentum=0.99):
        super(CosProto_Module, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_classes
        self.num_micro_proto = num_micro_proto
        self.init_proto_path = init_proto_path
        self.temp = 0.1

        # Load initial prototypes
        with open(init_proto_path, 'rb') as handle:
            init_protos = pickle.load(handle)
        num_class = len(init_protos)
        all_protos = list()
        for cls_id in range(num_class):
            all_protos.append(torch.tensor(np.stack(init_protos[cls_id], 0)))
        proto_tensor = torch.stack(all_protos, 0)
        self.proto_list = proto_tensor

        # Set momentum
        self.proto_unpdate_momentum = proto_unpdate_momentum

    def forward(self, x, select_mask):
        bs, in_channel, h_origin, w_origin = x.size()

        # Pixel-to-sample
        x_p2s = x.view(-1, in_channel).float()

        # L2-normalize
        x_p2s = torch.nn.functional.normalize(x_p2s, dim=1)

        # L2-normalize prototypes
        proto_list = self.proto_list.clone().float()
        proto_list = torch.nn.functional.normalize(proto_list, dim=1)

        # Cosine similarity
        masks = torch.mm(x_p2s, proto_list.t())

        # Argmax
        res_idx, res = torch.max(masks, dim=1)

        # Unsatisfied locations
        unsatisfied_loc = (select_mask == 0).float()

        # Reshape
        res = res.view(bs, h_origin, w_origin, self.num_classes)
        res = res * self.temp
        res[unsatisfied_loc] = 0

        return res, res_idx

    def update_proto(self, rep, cls_id, proto_id):
        self.proto_list[cls_id * self.num_micro_proto + proto_id] = (1 - self.proto_unpdate_momentum) * rep + self.proto_unpdate_momentum * self.proto_list[cls_id * self.num_micro_proto + proto_id]

