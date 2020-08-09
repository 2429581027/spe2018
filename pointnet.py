'''
    file:   pointnet.py
'''

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data

from stn3d import STN3dR, STN3dRQuad, STN3dTR

class PointNetfeatMini(nn.Module):
    def __init__(self, num_points = 2500, in_dim = 3, global_feat_dim = 1024, global_feat = True, with_stn = 'STN3dR'):
        super(PointNetfeatMini, self).__init__()
        self.global_feat = global_feat
        self.global_feat_dim = global_feat_dim
        if self.global_feat:
            self.feat_dim = self.global_feat_dim
        else:
            self.feat_dim = self.global_feat_dim+64   

        self.num_points = num_points
        self.with_stn = with_stn
        if self.with_stn == 'STN3dR':
            self.stn = STN3dR(num_points = self.num_points)
        elif self.with_stn == 'STN3dTR':
            self.stn = STN3dTR(num_points = self.num_points)
        elif self.with_stn == 'STN3dRQuad':
            self.stn = STN3dRQuad(num_points = self.num_points)


        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.global_feat_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.global_feat_dim)
        #self.mp1 = torch.nn.MaxPool1d(self.num_points)

    def forward(self, x) -> torch.Tensor:  
        if self.with_stn != 'None':
            trans = self.stn(x)          
            y = self.stn.transform(x, trans)
            x = F.relu(self.bn1(self.conv1(y)))
        else:
            x = F.relu(self.bn1(self.conv1(x)))

        pointfeat = x 
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        #x = self.mp1(x)
        x,_ = torch.max(x, 2)
        x = x.contiguous().view(-1, self.global_feat_dim)
        if self.global_feat:
            #return x, y
            return x
        else:
            x = x.contiguous().view(-1, self.global_feat_dim, 1)
            x = x.repeat(1, 1, self.num_points)
            #return torch.cat([x, pointfeat], 1), y
            return torch.cat([x, pointfeat], 1)


class PointNetfeat(nn.Module):
    '''
        with two more convolution layer than PointNetfeatMini
    '''
    def __init__(self, num_points = 2500, in_dim = 3, global_feat_dim = 1024, 
                        global_feat = True, with_stn = 'STN3dR', with_stn_feat = True):
        super(PointNetfeat, self).__init__()
        self.global_feat = global_feat
        self.global_feat_dim = global_feat_dim
        if self.global_feat:
            self.feat_dim = self.global_feat_dim
        else:
            self.feat_dim = self.global_feat_dim+64   

        self.num_points = num_points
        self.with_stn = with_stn
        if self.with_stn == 'STN3dR':
            self.stn = STN3dR(num_points = self.num_points)
        elif self.with_stn == 'STN3dTR':
            self.stn = STN3dTR(num_points = self.num_points)            
        elif self.with_stn == 'STN3dRQuad':
            self.stn = STN3dRQuad(num_points = self.num_points)


        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        
        self.with_stn_feat = with_stn_feat
        if self.with_stn_feat:
            self.stn_feat = STN3dR(num_points = self.num_points, dim = 64)

        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, self.global_feat_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.global_feat_dim)
        #self.mp1 = torch.nn.MaxPool1d(self.num_points)

    def forward(self, x) -> torch.Tensor:  
        '''
            x: [B, 3, N]
        '''
        N = x.size()[2]
        if self.with_stn != 'None':
            trans = self.stn(x)          
            y = self.stn.transform(x, trans)
            x = F.relu(self.bn1(self.conv1(y)))
        else:
            x = F.relu(self.bn1(self.conv1(x)))
        
        x = F.relu(self.bn2(self.conv2(x)))

        if self.with_stn_feat:
            trans = self.stn_feat(x)
            x = self.stn_feat.transform(x, trans)
        pointfeat = x 

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        #x = self.mp1(x)
        x,_ = torch.max(x, 2)
        x = x.contiguous().view(-1, self.global_feat_dim)
        if self.global_feat:
            #return x, y
            return x
        else:
            x = x.contiguous().view(-1, self.global_feat_dim, 1)
            x = x.repeat(1, 1, N)
            #return torch.cat([x, pointfeat], 1), y
            return torch.cat([x, pointfeat], 1)

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        '''
            x (B, bottleneck_size, N)
        '''
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = 2*self.th(self.conv4(x))
        return x

if __name__ == '__main__':
    from types import SimpleNamespace
    
    num_points = 2500
    # #sim_data = torch.rand(1,3,num_points) # jjcao, failed for batch 1
    sim_data = torch.rand(32,3,num_points)
    

    #####################
    ## test pointnet with STN3dR
    args = SimpleNamespace(with_stn='STN3dR')
    pointfeat = PointNetfeat(num_points = num_points, global_feat = True, with_stn = args.with_stn)    
    out = pointfeat(sim_data)
    print('global feat', out.size())

    #####################
    ## test pointnet with STN3dRQuad
    args.with_stn = 'STN3dRQuad'
    pointfeat = PointNetfeat(global_feat=False, with_stn = args.with_stn)
    out = pointfeat(sim_data)
    print('point feat', out.size())

    print('test finished')