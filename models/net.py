# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from models.net_utils import FC, MLP, LayerNorm
from models.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self,):
        super(AttFlat, self).__init__()


        self.mlp = MLP(in_size=512,mid_size=512,out_size=1,dropout_r=0.1,use_relu=True)

        self.linear_merge = nn.Linear(
            512 * 1,
            1024
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(1):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

  
        self.backbone = MCA_ED()

        self.attflat_img = AttFlat()
        self.attflat_lang = AttFlat()




    def forward(self, lang_feat, img_feat, lang_feat_mask, img_feat_mask):

        # Make mask
        # lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        # img_feat_mask = self.make_mask(img_feat)

        # # Pre-process Language Feature
        # lang_feat = self.embedding(ques_ix)
        # lang_feat, _ = self.lstm(lang_feat)

        # # Pre-process Image Feature
        # img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )


        # proj_feat = lang_feat + img_feat
        # proj_feat = self.proj_norm(proj_feat)
        # proj_feat = torch.sigmoid(self.proj(proj_feat))

        return lang_feat, img_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
