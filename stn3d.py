import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from util.quaternion import qrot

class STN3dT(nn.Module):
    ''' 
        3d translation               
    '''
    def __init__(self, num_points = 6890, dim = 3):
        super(STN3dT, self).__init__()
        self.dim = dim
        self.num_points = num_points
        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1) 
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        #self.mp1 = torch.nn.MaxPool1d(self.num_points)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.dim) # 3
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)


    def forward(self, x):
        '''
            input: B x 3 x N
            output: B x 3
        '''
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.mp1(x)
        x,_ = torch.max(x, 2)
        x = x.contiguous().view(-1, 256)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
 
        return x


    def transform(self, x, trans):
        '''
            x: (B, N, 3)
            trans: (B, 3)
        '''
        x = x - trans.unsqueeze(1)#.expand(B, verts.shape[1], 3)
        return x

    # def transform(self, x, trans):
    #     '''
    #         x: (B, 3, N)
    #         trans: (B, 3)
    #     '''
    #     x = x.transpose(2,1)
    #     x = x - trans.unsqueeze(1)#.expand(B, verts.shape[1], 3)
    #     x = x.transpose(2,1)
    #     return x

class STN3dTR(nn.Module):
    ''' 
        translate + rotation matrix
    '''
    def __init__(self, num_points = 2500, dim = 3):
        super(STN3dTR, self).__init__()
        self.dim = dim
        self.num_points = num_points
        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1) 
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(self.num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + self.dim * self.dim) # 3 x 3
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        '''
            input: B x 3 x N
            output: x (B, 3+9)
            # output: T (B, 3); R (B, 3, 3)
        '''
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.mp1(x)
        x,_ = torch.max(x, 2)
        x = x.contiguous().view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim).view(1, self.dim * self.dim).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x[:, 3:] = x[:, 3:] + iden # x: B x 9  
        
        return x
        #return T, R

    def transform(self, x, trans):
        '''
            x: (B, 3, N)
            trans: (B, 3, 3)
        '''        
        T = trans[:, :3].contiguous() 
        R = trans[:, 3:].contiguous() 
        R = R.view(-1, self.dim, self.dim) # x: B x 3 x 3 

        x = x.transpose(2,1)        
        x = torch.bmm(x, R)
        x = x - T.unsqueeze(1)
        x = x.transpose(2,1)
        return x

class STN3dR(nn.Module):
    ''' 
        use rotation matrix        
        In PointNet & 3d-coded, there are no BatchNorm1d in STN. 
        In my experiments for regress shape and pose parameters of SMPL, STN with BatchNorm1d is far better than STN without BatchNorm1d. 
    '''
    def __init__(self, num_points = 2500, dim = 3):
        super(STN3dR, self).__init__()
        self.dim = dim
        self.num_points = num_points
        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1) 
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(self.num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim * self.dim) # 3 x 3
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        '''
            input: B x 3 x N
            output: B x 3 x 3
        '''
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.mp1(x)
        x,_ = torch.max(x, 2)
        x = x.contiguous().view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim).view(1, self.dim * self.dim).repeat(batchsize,1)
        #iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden # x: B x 9  # todo why? jjcao
        x = x.contiguous().view(-1, self.dim, self.dim) # x: B x 3 x 3 
        return x

    def transform(self, x, trans):
        '''
            x: (B, 3, N)
            trans: (B, 3, 3)
        '''        
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        return x

class STN3dRQuad(nn.Module):
    ''' 
        use quaternion.

        References:
            1. pcpnet.py

        todo: should be replaced by Li algebra: refer to deep6dpose?

            x = F.normalize(F.tanh(self.fc2(x))), according to https://github.com/JHUVisionLab/multi-modal-regression/blob/master/quaternion.py
    '''
    def __init__(self, num_points = 2500, dim = 3):
        super(STN3dRQuad, self).__init__()
        self.dim = dim
        self.num_points = num_points
        #torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1) 
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(self.num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.mp1(x)
        x,_ = torch.max(x, 2)
        x = x.contiguous().view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        return x

    def transform(self, x, trans):
        # trans: B x 4 => B x self.num_points x 4
        trans = trans.contiguous().view(-1, 1, 4)
        trans = trans.repeat(1, self.num_points, 1)
        x = x.transpose(2,1)
        x = qrot(trans,x)
        x = x.transpose(2,1)
        
        return x

if __name__ == '__main__':
    num_points = 2500
    # #sim_data = Variable(torch.rand(1,3,num_points)) # jjcao, failed for batch 1
    sim_data = torch.rand(2,3,num_points)

    #####################
    # test STN3dR
    trans = STN3dTR()
    out = trans(sim_data)
    print('stn3d', out.size())        
    verts = trans.transform(sim_data, out)  

    #####################
    # test STN3dRQuad
    #trans = STN3dRQuad()
    trans = STN3dR()
    out = trans(sim_data)
    print('stn3dQuad', out.size())
    verts = trans.transform(sim_data, out)  

    print('test finished')