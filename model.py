'''
    file:   model.py

    date:   2018_09_19
    author: Junjie Cao
'''

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from config import args
from SMPL import SMPL
import trimesh
from LinearModel import LinearModel
from pointnet import PointNetfeatMini, PointNetfeat, PointGenCon
from stn3d import STN3dR, STN3dRQuad, STN3dT, STN3dTR
# ssg
if args.encoder == 'pointnet2':
    from pointnet2.models import Pointnet2FeatSSG
if args.decoder == 'pointnet2':
    from pointnet2.models import Pointnet2DisSSG
    from pointnet2.utils.pointnet2_utils import QueryAndGroup
from util.util import load_mean_smpl_parameters, center_pts_np_array, center_pts_tensor

class SmplRegressor(LinearModel):
    '''
        for beta + theta: self.begin = 0, self.end = 82
        for beta:         self.begin = 72, self.end = 82
        for theta:        self.begin = 0, self.end = 72
    '''
    def __init__(self, begin, end, fc_layers, use_dropout, drop_prob, use_ac_func, iterations):
        super().__init__(fc_layers, use_dropout, drop_prob, use_ac_func)
        self.begin = begin
        self.end = end
        self.iterations = iterations

        mean_smpl = np.tile(load_mean_smpl_parameters(args.smpl_mean_theta_path), args.batch_size).reshape((args.batch_size, -1))
        self.register_buffer('mean_smpl', torch.from_numpy(mean_smpl).float())

    def forward(self, inputs):
        '''
            param:
                inputs: is the output of encoder, which has 1024 features
            
            return:
                a list contains [ [para1, para, ..., para1], [para2, para2, ..., para2], ... , ], shape is iterations X B X 82
        '''     
        paras = []
        para = self.mean_smpl[:inputs.shape[0], self.begin:self.end]
        for _ in range(self.iterations):
            total_inputs = torch.cat([inputs, para], 1)
            para = para + self.fc_blocks(total_inputs)
            paras.append(para)
        return paras

class SPENetBase(nn.Module):
    def __init__(self):
        super().__init__()
        self._read_configs()  

        print('start creating sub modules...')
        # self.smpl_model == neutral_smpl_with_cocoplus_reg.txt is only for female.
        self.smpl = SMPL(self.smpl_model, obj_saveable = True)        
        self._create_sub_modules()

        if args.decoder != 'None':
            #self.mesh = trimesh.load("./data/template/template_smpl_mean_center.ply", process=False)
            self.mesh = trimesh.load("./data/template/template.ply", process=False)
            point_set, translation = center_pts_np_array( self.mesh.vertices)
            self.register_buffer('vertex', torch.from_numpy(point_set).float())
            self.num_vertex = self.vertex.size(0)

    def _read_configs(self):
        self.with_stn = args.with_stn
        self.with_stn_feat = args.with_stn_feat
        self.point_count = args.point_count           
        self.smpl_model = args.smpl_model
        #self.smpl_mean_theta_path = args.smpl_mean_theta_path
        self.trans_smpl_generated = args.trans_smpl_generated
        self.beta_count = 10 # shape
        self.theta_count = 72 # pose
        self.joint_count = 24
        self.beta_code_dim = 512 # 512?
        self.theta_code_dim = 1024

    def _create_encoder(self, in_dim = 3, code_dim = 1024):
        if args.encoder == 'pointnet2':
            core_encoder = Pointnet2FeatSSG(num_points = self.point_count, with_stn = self.with_stn)
        elif args.encoder == 'pointnetmini':
            core_encoder = PointNetfeatMini(num_points = self.point_count, in_dim = in_dim, 
                                global_feat_dim = code_dim, global_feat=True, with_stn = self.with_stn)                              
        else:
            core_encoder = PointNetfeat(num_points = self.point_count, in_dim = in_dim, 
                                global_feat_dim = code_dim, global_feat=True, with_stn = self.with_stn,
                                with_stn_feat = self.with_stn_feat)  

        # this has been tested, no use
        # encoder = nn.Sequential(
        #             core_encoder,
        #             nn.Linear(code_dim, code_dim), # 512 to 512 or 1024 to 1024
        #             nn.BatchNorm1d(code_dim),
        #             nn.ReLU()
        #             )  

        return core_encoder

    def _create_regressor(self, para_count, code_dim = 1024, fc2=1024):
        '''
            regressor can predict beta & theta, for SMPL, from coder extracted by encoder in a iteratirve way
        '''
        fc_layers = [code_dim + para_count, 1024, fc2, para_count] # There are 3 fc layers (1st 1024 is for encoder's output vector)
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False] #unactive the last layer
        begin = 0
        end = para_count
        if para_count == 10:
            begin = 72
            end = 82
        return SmplRegressor(begin, end, fc_layers, use_dropout, drop_prob, use_ac_func, iterations=3)

    def _create_sub_modules(self):
        self._create_encoder_modules()
        if self.trans_smpl_generated == 'stn':
            #self.translate_stn = STN3dT(num_points=6890, dim = 3)
            self.translate_stn = STN3dTR(num_points=6890, dim = 3)
        else:
            self.translate_stn = None
        
        self._create_decoder_modules()
        print('finished create the SPENet modules...')

    def _calc_detail_info(self, beta, theta):
        #theta[0:3] = 0.0
        verts, j3d, Rs = self.smpl(beta = beta, theta = theta, get_skin = True)
        return (verts, j3d)

    def encode(self, inputs, inBeta = None, inTheta = None): 
        '''
        outputs
            (beta, theta, verts, j3d, codeBeta, codePose)  
            verts: (B, 6890, 3)
        '''
        B, N = inputs.shape[0], inputs.shape[1]

        #####################
        if args.encoder == 'pointnet2':
            inputs = torch.cat([inputs, inputs], 2)
        else: # PointNetfeatMini or PointNetfeat
            inputs = inputs.transpose(2, 1)

        beta, theta, verts, j3d, codeBeta, codePose = self._encoding(inputs, inBeta, inTheta)

        #####################
        if self.trans_smpl_generated == 'center':
            translation = center_pts_tensor(verts)
            verts = verts - translation.unsqueeze(1)            
        elif self.trans_smpl_generated == 'stn': 
            verts = verts.transpose(2, 1)
            trans = self.translate_stn(verts)  
            verts = self.translate_stn.transform(verts, trans)
            verts = verts.transpose(2, 1) 

        return (beta, theta, verts, j3d, codeBeta, codePose)  

    def decode(self, code, idx = None):
        '''
            idx: (B, N)
        '''
        if args.decoder == 'PointGenCon':
            if idx is None:
                B = code.shape[0]
                N = self.vertex.shape[0] # self.vertex (6890, 3)
                rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(B, 3,-1)
            else:
                B, N = idx.shape[0], idx.shape[1]
                idx = idx.view(-1)
                idx = idx.cpu().numpy().astype(np.int)

                rand_grid = self.vertex[idx,:] # self.vertex (6890, 3)
                #rand_grid = self.smpl.v_template[idx,:]
                rand_grid = rand_grid.view(B, -1, 3).transpose(1,2).contiguous()
            
            rand_grid = Variable(rand_grid) # (B, 3, N==self.point_count) 
            
            # (B, 82) => (B, 82, N)
            y = code.unsqueeze(2).expand(B, code.size(1), N ).contiguous()            
            verts_decoded = self.decoderNet(torch.cat([rand_grid, y], 1).contiguous() ) # (B, 3+82, N) => (B, 3, N)
            verts_decoded = verts_decoded.transpose(2,1)

        else:
            with torch.no_grad():  # this works for pytorh 0.4.0               
                verts_decoded = torch.zeros(1)
                if torch.cuda.is_available(): verts_decoded = verts_decoded.cuda()       

        return verts_decoded

    def forward(self, inputs, idx = None, inBeta = None, inTheta = None): 
        '''
            inputs: (B, N, 3)
            idx: (B, N)

            output:
                theta: (B, 82)
                verts: (B, 6890, 3)
                j3d: (B, N, 3)
                verts_decoded: (B, N, 3)

            N is 2400 usually.
        '''
        beta, theta, verts, j3d, codeBeta, codePose = self.encode(inputs, inBeta, inTheta)
        
        if self.decoderNet.bottleneck_size == 3 + self.beta_count + self.theta_count:
            verts_decoded = self.decode(torch.cat([beta, theta], 1), idx) 
        elif self.decoderNet.bottleneck_size == 3 + self.theta_code_dim: # codeTheta is code actually, i.e. include info for both shape & pose 
            verts_decoded = self.decode(codePose, idx) 
        else: # self.decoderNet.bottleneck_size == 3 + self.beta_code_dim + self.theta_code_dim
            verts_decoded = self.decode(torch.cat([codeBeta, codePose], 1).contiguous(), idx) 

        return (beta, theta, verts, j3d, verts_decoded)   

    def decode_full(self, code):
        '''            
            self.vertex_HR has 241002 vertices
            div = 20
            batch = int(self.num_vertex_HR/div) # batch = 12050

            worked
        '''        
        self.mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        #self.mesh_HR.show()
        point_set_HR, translation = center_pts_np_array( self.mesh_HR.vertices)

        self.vertex_HR = torch.from_numpy(point_set_HR).float()
        if torch.cuda.is_available(): self.vertex_HR = self.vertex_HR.cuda()
        self.num_vertex_HR = self.vertex_HR.size(0)

        if args.decoder == 'PointGenCon':
            B = code.shape[0]
            outs = []
            div = 40
            batch = int(self.num_vertex_HR/div)
            
            # (B, 82) => (B, 82, batch)      
            y = code.unsqueeze(2).expand(B, code.size(1), batch ).contiguous()

            for i in range(div-1):
                rand_grid = self.vertex_HR[batch*i:batch*(i+1)]
                rand_grid = rand_grid.view(B, batch, 3).transpose(1,2).contiguous()
                rand_grid = Variable(rand_grid)

                outs.append( self.decoderNet(torch.cat( (rand_grid, y), 1).contiguous()) )
                torch.cuda.synchronize()

            i = div - 1
            rand_grid = self.vertex_HR[batch*i:].view(B, -1, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)            
            y = code.unsqueeze(2).expand(B, code.size(1), rand_grid.size(2) ).contiguous()
            outs.append( self.decoderNet(torch.cat( (rand_grid, y), 1).contiguous()) )
            torch.cuda.synchronize()

            verts_decoded = torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
          
        else:
            with torch.no_grad():  # this works for pytorh 0.4.0               
                verts_decoded = torch.zeros(verts.shape)
                if torch.cuda.is_available(): verts_decoded = verts_decoded.cuda()       

        return verts_decoded

    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        r"""Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.
        """
        return self.train(False)
  
class SPENet(SPENetBase):
    def __init__(self):
        super().__init__()

    def _create_encoder_modules(self):
        self.encoder = self._create_encoder()
        self.regressor = self._create_regressor(self.beta_count + self.theta_count)

    def _create_decoder_modules(self):
        #total_para_count = self.beta_count + self.theta_count
        #self.decoderNet = PointGenCon(bottleneck_size = 3 + total_para_count)
        self.decoderNet = PointGenCon(bottleneck_size = 3 + self.theta_code_dim)

    def _encoding(self, inputs, inBeta = None, inTheta = None): 
        code = self.encoder(inputs) # (B, 1024)

        paras = self.regressor(code)
        para = paras[-1]        
        beta = para[:, 72:].contiguous() 
        theta = para[:, :72].contiguous() 
        (verts, j3d) = super()._calc_detail_info(beta, theta) # verts (B, 6890, 3)

        return (beta, theta, verts, j3d, None, code)

class SPENetBeta(SPENetBase):
    def __init__(self):
        super().__init__()
    
    def _create_encoder_modules(self):
        self.encoderBeta = self._create_encoder(in_dim = 3, code_dim = self.beta_code_dim)
        self.regressorBeta = self._create_regressor(para_count = self.beta_count, code_dim = self.beta_code_dim, fc2=512)

    def _create_decoder_modules(self):
        total_para_count = self.beta_count + self.theta_count
        self.decoderNet = PointGenCon(bottleneck_size = 3 + total_para_count)

    def _encoding(self, inputs, inBeta = None, inTheta = None):         
        codeBeta = self.encoderBeta(inputs)# (B, 1024)

        # add inTheta to inputs, no improvement!!
        # B = inputs.shape[0]
        # if args.encoder == 'pointnet2':
        #     N = inputs.shape[1]
        #     y = inTheta.unsqueeze(2).expand(B, N, inTheta.size(1) ).contiguous()
        #     codeBeta = self.encoderBeta( torch.cat([inputs, inTheta], 2) ) 
        # else:# PointNetfeatMini or PointNetfeat
        #     N = inputs.shape[2]
        #     y = inTheta.unsqueeze(2).expand(B, inTheta.size(1), N ).contiguous()            
        #     codeBeta = self.encoderBeta( torch.cat([inputs, y], 1) ) 

        beta = self.regressorBeta(codeBeta)
        beta = beta[-1]
        theta = inTheta
        (verts, j3d) = super()._calc_detail_info(beta, theta) # verts (B, 6890, 3)

        return (beta, theta, verts, j3d, codeBeta, None)

class SPENetPose(SPENetBase):
    def __init__(self):
        super().__init__()
        
    def _create_encoder_modules(self):
        self.encoderPose = self._create_encoder(code_dim = self.theta_code_dim)
        self.regressorPose = self._create_regressor(para_count = self.theta_count, code_dim = self.theta_code_dim)

    def _create_decoder_modules(self):
        total_para_count = self.beta_count + self.theta_count
        self.decoderNet = PointGenCon(bottleneck_size = 3 + total_para_count)

    def _encoding(self, inputs, inBeta = None, inTheta = None): 
        codePose = self.encoderPose(inputs) # (B, 1024)

        theta = self.regressorPose(codePose)
        theta = theta[-1]
        beta = inBeta
        (verts, j3d) = super()._calc_detail_info(beta, theta) # verts (B, 6890, 3)

        return (beta, theta, verts, j3d, None, codePose)
    

class SPENetSiam(SPENetBase):
    '''
        2 encoders + SMPL + translation + decoder

        Comments:
            Do not share STN3dR is better for regress beta and pose. Why?
                1. more parameters? 
                2. inputs are STN 2 times actually, 1st for beta encoder, 2nd for pose encoder.
    '''
    def __init__(self):
        super().__init__()
    def _create_encoder_modules(self):
        self.encoderBeta = self._create_encoder(code_dim = self.beta_code_dim)
        self.regressorBeta = self._create_regressor(para_count = self.beta_count, code_dim = self.beta_code_dim, fc2=self.beta_code_dim)

        self.encoderPose = self._create_encoder(code_dim = self.theta_code_dim)
        self.regressorPose = self._create_regressor(para_count = self.theta_count, code_dim = self.theta_code_dim)

    def _create_decoder_modules(self):
        #total_para_count = self.beta_count + self.theta_count
        #self.decoderNet = PointGenCon(bottleneck_size = 3 + total_para_count)
        self.decoderNet = PointGenCon(bottleneck_size = 3 + self.beta_code_dim + self.theta_code_dim)

    def _encoding(self, inputs, inBeta = None, inTheta = None): 
        codeBeta = self.encoderBeta(inputs) # (B, 1024)
        codePose = self.encoderPose(inputs) 

        betas = self.regressorBeta(codeBeta)
        beta = betas[-1]
        thetas = self.regressorPose(codePose)
        theta = thetas[-1]
        (verts, j3d) = super()._calc_detail_info(beta, theta) # verts (B, 6890, 3)

        return (beta, theta, verts, j3d, codeBeta, codePose)

class SPENetSiam2(SPENetBase):
    '''
        same encoding with SPENetSiam, but 2 times decoding: betaCode + template first, then thetaCode + dM1
    '''
    def __init__(self):
        super().__init__()

    def _create_decoder_modules(self):
        self.decoderNetBeta = PointGenCon(bottleneck_size = 3 + self.beta_code_dim)
        self.decoderNetTheta = PointGenCon(bottleneck_size = 3 + self.theta_code_dim)

    def decode(self, code, idx = None):
        '''
            idx: (B, N)
        '''
        if args.decoder == 'PointGenCon':
            #torch.cat([codeBeta, codePose], 1)
            codeBeta = code[:, :self.beta_code_dim].contiguous() 
            codeTheta = code[:, self.beta_code_dim:].contiguous() 

            if idx is None:
                B = codeBeta.shape[0]
                N = self.vertex.shape[0] # self.vertex (6890, 3)
                rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(B, 3,-1)
            else:
                B, N = idx.shape[0], idx.shape[1]
                idx = idx.view(-1)
                idx = idx.cpu().numpy().astype(np.int)

                rand_grid = self.vertex[idx,:] # self.vertex (6890, 3)
                #rand_grid = self.smpl.v_template[idx,:]
                rand_grid = rand_grid.view(B, -1, 3).transpose(1,2).contiguous()
            
            rand_grid = Variable(rand_grid) # (B, 3, N==self.point_count) 
            
            # folding by beta
            # (B, 512) => (B, 512, N)
            y = codeBeta.unsqueeze(2).expand(B, codeBeta.size(1), N ).contiguous()       
            verts_decoded = self.decoderNetBeta(torch.cat([rand_grid, y], 1).contiguous() ) # (B, 3+82, N) => (B, 3, N)

            # 2nd folding by theta
            # (B, 1024) => (B, 1024, N)
            y = codeTheta.unsqueeze(2).expand(B, codeTheta.size(1), N ).contiguous()
            # verts_decoded = self.smpl.v_template.unsqueeze(0).expand(verts.size()).transpose(2,1).contiguous() # (6890, 3) => (B, 3, 6890)
            verts_decoded = self.decoderNetTheta(torch.cat([verts_decoded, y], 1)) # (B, 3+82, 6890) => (B, 3, 6890)
            verts_decoded = verts_decoded.transpose(2,1)

        else:
            with torch.no_grad():  # this works for pytorh 0.4.0               
                verts_decoded = torch.zeros(1)
                if torch.cuda.is_available(): verts_decoded = verts_decoded.cuda()       

        return verts_decoded

class SPENetSiamSharedStn(SPENetBase):
    '''
        not ready after refactor
        SPENetSiamSharedStn is worse than seperate stn, i.e. SPENetSiam
    '''
    def __init__(self):
        super().__init__()

    def _create_sub_modules(self):       
        '''
            SMPL can create a mesh from beta & theta
        '''
        # self.smpl_model == neutral_smpl_with_cocoplus_reg.txt is only for female.
        self.smpl = SMPL(self.smpl_model, obj_saveable = True)

        if self.with_stn == 'STN3dR':
            self.spatial_stn = STN3dR(num_points = self.point_count)
        elif self.with_stn == 'STN3dRQuad':
            self.spatial_stn = STN3dRQuad(num_points = self.point_count)

        if args.encoder == 'pointnet2':
            self.encoderBeta = Pointnet2FeatSSG(num_points = self.point_count, with_stn = 'None')
            self.encoderPose = Pointnet2FeatSSG(num_points = self.point_count, with_stn = 'None')
        else:
            self.encoderBeta = PointNetfeat(num_points = self.point_count, global_feat=True, with_stn = 'None')
            self.encoderPose = PointNetfeat(num_points = self.point_count, global_feat=True, with_stn = 'None')

        '''
            regressor can predict betas(include beta and theta which needed by SMPL) from coder extracted from encoder in a iteratirve way
        '''        
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False] #unactive the last layer
        fc_layers = [self.encoderBeta.feat_dim + 10, 1024, 1024, 10] # There are 3 fc layers (1st 1024 is for encoder's output vector)
        self.regressorBeta = BetaRegressor(fc_layers, use_dropout, drop_prob, use_ac_func, iterations=3)
        fc_layers = [self.encoderPose.feat_dim + 72, 1024, 1024, 72]
        self.regressorPose = PoseRegressor(fc_layers, use_dropout, drop_prob, use_ac_func, iterations=3)

        '''
            displament net, estimate vertex displacement between mesh,
                                             generated by smpl(regressed beta & pose parameters), 
                                             & gt mesh.
        '''
        if args.decoder != 'None':
            nsample = 32
            self.disNet = Pointnet2DisSSG(input_channels=3*nsample, use_xyz=True)
            # todo no knn is supported by current pointnet2 implementation
            self.qg = QueryAndGroup(radius = 0.2, nsample = nsample, use_xyz = False)

        print('finished create the SPENet modules...')
      
    def forward(self, x, betapara = None, posepara = None): 
        '''
            x: B x N x 3

            posepara: is for debug only.
        '''
        B, N = x.shape[0], x.shape[1]

        if self.with_stn != 'None':
            x = x.transpose(2, 1)
            trans = self.spatial_stn(x)          
            x = self.spatial_stn.transform(x, trans)
            if args.encoder == 'pointnet2':
                x = x.transpose(2, 1)
                x = torch.cat([x, x], 2)            
        else:
            if args.encoder == 'pointnet2':
                x = torch.cat([x, x], 2)
            else:
                x = x.transpose(2, 1)          

        featureBeta = self.encoderBeta(x)# feature.shape => B, 1024
        featurePose = self.encoderPose(x)

        betas = self.regressorBeta(featureBeta)
        beta = betas[-1]
        poses = self.regressorPose(featurePose)
        pose = poses[-1]
        theta = torch.cat([pose, beta], 1)
        (theta, verts, j3d) = super()._calc_detail_info(theta)

        if args.encoder == 'pointnet2':
            x = x[:, :, :2]
        else:
            x = x.transpose(2, 1)

        #################################
        if args.decoder == 'None':        
            with torch.no_grad():  # this works for pytorh 0.4.0               
                disp = torch.zeros(verts.shape)
                if torch.cuda.is_available(): disp = disp.cuda()             
        else:  
            npoint = verts.shape[1]
            #(B, C=3, npoint, nsample) 
            nfeat = self.qg(xyz = x, new_xyz = verts.contiguous(), features = x.transpose(1,2).contiguous())
            # to (B, npoint, nsample*3)
            nfeat = nfeat.transpose(1,2).transpose(2,3).reshape((B, npoint, self.qg.nsample*3))

            pc_fea = torch.cat([verts, nfeat], 2) # pc_fea: B x 6890 x (3+C)
            disp = self.disNet(pc_fea)

        return (theta, verts, j3d, disp)        


def test(sim_data, sim_beta, sim_pose, idx):
    net = SPENet()
    #net = SPENetBeta()
    #net = SPENetPose()
    #net = SPENetSiam()
    #print(net)

    #net = nn.DataParallel(net) # for multiple gpu    
    if torch.cuda.is_available():
        sim_data = sim_data.cuda()
        sim_beta = sim_beta.cuda()
        sim_pose = sim_pose.cuda()
        #idx = idx.cuda()
        net = net.cuda()


    out = net(sim_data, idx, sim_beta, sim_pose)
    (predict_beta, predict_theta, predict_verts, predict_j3d, verts_decoded) = out

    # for i in range(predict_j3d.shape[0]): 
    #     j3d = predict_j3d[i].detach().cpu().numpy()
    #     np.save('j3d.npy', j3d)

    # j3d = np.load('j3d.npy')

    print(args.encoder)
    print('output length', len(out))

if __name__ == '__main__':
    args.batch_size = 2
    sim_data = torch.rand(args.batch_size, args.point_count ,3)
    sim_beta = torch.rand(args.batch_size, 10)
    sim_pose = torch.rand(args.batch_size, 72)
    idx = np.random.choice(6890, 2400, replace=False)
    idx = torch.from_numpy(idx)
    idx = idx.unsqueeze(0).expand(args.batch_size, 2400).contiguous()

    ##########################################
    args.encoder = 'pointnet'
    args.decoder = 'PointGenCon'
    test(sim_data, sim_beta, sim_pose, idx)
    ##########################################
    # args.encoder = 'pointnet2'
    # from pointnet2.models import Pointnet2FeatSSG    
    # test(sim_data, sim_beta, sim_pose)

    # args.decoder = 'pointnet2'
    # from pointnet2.models import Pointnet2DisSSG