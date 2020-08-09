'''
    file:   pointnet2_ssg_feat.py
'''

import sys
import os
sys.path.append(os.getcwd())
sys.path.append('../')

from stn3d import STN3dR, STN3dRQuad

import torch
import torch.nn as nn
from pointnet2.utils import pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import PointnetSAModule
from collections import namedtuple

def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'acc'])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.to('cuda', non_blocking=True)
            labels = labels.to('cuda', non_blocking=True)

            preds = model(inputs)
            labels = labels.view(-1)
            loss = criterion(preds, labels)

            _, classes = torch.max(preds, -1)
            acc = (classes == labels).float().sum() / labels.numel()

            return ModelReturn(
                preds, loss, {
                    "acc": acc.item(),
                    'loss': loss.item()
                }
            )

    return model_fn


class Pointnet2SSG(nn.Module):
    r"""
        PointNet2 with single-scale grouping Classification network.
                    fail to work without cuda

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True, num_points = 2500, with_stn = 'STN3dR'):
        super().__init__()
        
        self.feat_dim = 1024
        self.num_points = num_points


        self.with_stn = with_stn
        if self.with_stn == 'STN3dR':
            self.stn = STN3dR(num_points = self.num_points)
        elif self.with_stn == 'STN3dRQuad':
            self.stn = STN3dRQuad(num_points = self.num_points)


        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(mlp=[256, 256, 512, self.feat_dim], use_xyz=use_xyz)
        )

        # self.FC_layer = nn.Sequential(
        #     pt_utils.FC(1024, 512, bn=True),
        #     nn.Dropout(p=0.5),
        #     pt_utils.FC(512, 256, bn=True),
        #     nn.Dropout(p=0.5),
        #     pt_utils.FC(256, num_classes, activation=None)
        # )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network

            input:
                pointcloud: torch.cuda.FloatTensor
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on each point in the point-cloud MUST
                    be formated as (x, y, z, features...)

            output: 
                features: B x 1024
        """
        xyz, features = self._break_up_pc(pointcloud) # xyz.shape => B, N ,3; features => B, 3, N

        if self.with_stn != 'None':
            xyz = xyz.transpose(1,2)
            trans = self.stn(xyz)  
            xyz = self.stn.transform(xyz, trans)
            xyz = xyz.transpose(1,2)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        
        return features.view(-1, self.feat_dim)
        #return self.FC_layer(features.squeeze(-1))

if __name__ == '__main__':
    from types import SimpleNamespace
    
    batch_size = 2
    num_points = 2500
    # #sim_data = torch.rand(1,3,num_points) # jjcao, failed for batch 1
    inputs = torch.rand(batch_size, num_points, 3)
    inputs = torch.cat([inputs, inputs], 2).cuda()
    
    #####################
    ## test pointnet with STN3dR
    #args = SimpleNamespace(with_stn='STN3dR')
    args = SimpleNamespace(with_stn='None')
    pointfeat = Pointnet2SSG(input_channels=3, use_xyz=True, 
                            num_points = num_points, with_stn = args.with_stn).cuda()

    print(pointfeat)

    out = pointfeat(inputs)
    print('global feat', out.size())

    print('test finished')