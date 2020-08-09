import sys
import os
sys.path.append(os.getcwd())
sys.path.append('../')

import torch
import torch.nn as nn
from pointnet2.utils import pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import PointnetSAModule, PointnetFPModule
from collections import namedtuple


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'acc'])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.to('cuda', non_blocking=True)
            labels = labels.to('cuda', non_blocking=True)

            preds = model(inputs)
            loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))

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
        PointNet2 with single-scale grouping
        displacement network that uses feature propogation layers

        Parameters
        ----------
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True):
        super().__init__()

        num_classes = 3 # dimmensions of displacement vector

        self.SA_modules = nn.ModuleList()
        #from smpl template 6890 to 1024
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[input_channels, 32, 32, 64],
                use_xyz=use_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=use_xyz
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointnetFPModule(mlp=[128 + input_channels, 128, 128, 128])
        )
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.FC_layer = nn.Sequential(
            pt_utils.Conv1d(128, 128, bn=True), nn.Dropout(),
            #pt_utils.Conv1d(128, 64, bn=True),
            pt_utils.Conv1d(128, num_classes, activation=None)
        )

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

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud) # xyz.shape => B, N ,3; features => B, 3, N

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.FC_layer(l_features[0]).transpose(1, 2).contiguous()

def test_qg():
    from pointnet2.utils.pointnet2_utils import QueryAndGroup

    # sampling on verts, but query and group features (xyz of pts) from pts
    B, N, npoint, nsample = 2, 50, 100, 2

    verts = torch.randn(B, npoint, 3).cuda()
    pts = torch.randn(B, N, 3).cuda()
    qg = QueryAndGroup(radius = 0.2, nsample = nsample, use_xyz = False).cuda()
    nfeat = qg(xyz = pts, new_xyz = verts, features = pts.transpose(1,2).contiguous()) #(B, C=3, npoint, nsample) 
    # to (B, npoint, nsample*3)
    target = nfeat.transpose(1,2).transpose(2,3).reshape((B, npoint, nsample*3))

    inputs = torch.cat([verts, target], 2)
    pnet = Pointnet2SSG(input_channels=target.shape[2], use_xyz=True).cuda()
    print(pnet)
    out = pnet(inputs)
def test_qg_real():
    '''
        not finished
    '''
    import sys
    import os
    sys.path.append(os.getcwd())
    sys.path.append('../')
    from datasets import SurrealDataset   
    root = '/data/spe_dataset_surreal_val'##spe_dataset_surreal_1pose_1shape
    args.batch_size = 2 
    data_set = SurrealDataset(root=root)
    data = data_set[0]
    
    # from torch.utils.data import DataLoader
    # loader = DataLoader(
    #         dataset = data_set,
    #         batch_size = args.batch_size,
    #         shuffle = True,
    #         drop_last = True,
    #         pin_memory = True,
    #         num_workers = args.batch_size
    #     )
    #loader = iter(loader)
    #data = loader.next()
    pts, shapepara, posepara = data['pts'], data['shape'], data['pose']
    if torch.cuda.is_available():
        pts, shapepara, posepara = pts.cuda(), shapepara.cuda(), posepara.cuda()

    gtverts, j3d, r = data_set.smpl(shapepara, posepara, get_skin = True)

    
    # #sim_data = torch.rand(1,3,num_points) # jjcao, failed for batch 1
    #inputs = torch.rand(batch_size, num_points, 3)
    gtverts = torch.cat([gtverts, gtverts], 2)
    
    pnet = Pointnet2SSG(input_channels=3, use_xyz=True)
    if torch.cuda.is_available(): 
        gverts = gverts.cuda()
        pnet = pnet.cuda()

    print(pnet)

    out = pnet(gverts)

if __name__ == '__main__':
    test_qg()
    #test_qg_real()
    print('test finished')