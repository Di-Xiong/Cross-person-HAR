# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from alg.algs.ERM import ERM

class Our(ERM):
    def __init__(self, args):
        super(Our, self).__init__(args)
        self.num_classes = args.num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.featurizer.in_features
        self.ssc_l = args.ssc_l
        self.ssc_f = args.ssc_f
        #256 or 1024
        hidden_size = input_feat_size if input_feat_size==2048 else 256
        
        self.cdpl = nn.Sequential(
                            nn.Linear(input_feat_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, input_feat_size),
                            nn.BatchNorm1d(input_feat_size)
        )
        
    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        lam = np.random.beta(0.5, 0.5)
        
        batch_size = all_y.size()[0]
        
        # cluster and order features into same-class group
        with torch.no_grad():   
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex==val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y
        
        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)
        
        logit = self.classifier(feat)

        # shuffle logit and proj
        logit_2 = torch.zeros_like(logit)
        proj_2 = torch.zeros_like(proj)
        logit_3 = torch.zeros_like(logit)
        proj_3 = torch.zeros_like(proj)
        ex = 0
        
        for end in intervals:
            shuffle_indices = torch.randperm(end-ex)+ex
            shuffle_indices2 = torch.randperm(end-ex)+ex
            for idx in range(end-ex):
                logit_2[idx+ex] = logit[shuffle_indices[idx]]
                proj_2[idx+ex] = proj[shuffle_indices[idx]]
                logit_3[idx+ex] = logit[shuffle_indices2[idx]]
                proj_3[idx+ex] = proj[shuffle_indices2[idx]]
            ex = end
        
        # Mixup logit and proj
        logit_3 = lam*logit_2 + (1-lam)*logit_3
        proj_3 = lam*proj_2 + (1-lam)*proj_3

        # Regularization Block
        L_id_logit = self.MSEloss(logit, logit_2)
        L_iid_logit = self.MSEloss(logit, logit_3)
        L_id_feat =  self.MSEloss(feat, proj_2)
        L_iid_feat =  self.MSEloss(feat, proj_3)
        
        L_logit = self.ssc_l*(L_id_logit + L_iid_logit)
        L_feat = self.ssc_f*(L_id_feat + L_iid_feat)

        cl_loss = F.cross_entropy(logit, all_y)
        L_ssc = L_logit+L_iid_logit + L_feat
        loss = cl_loss + L_ssc
     
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        return {'class': loss.item()}
    

    def predict(self, x):
        features = self.featurizer(x)
        pred = self.classifier(features)
        return pred