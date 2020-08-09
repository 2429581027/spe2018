'''
    file:   train.py

    train for shape & pose parameters regression

    date:   2018_09_19
    author: Junjie Cao
    based on pointnet.pytorch and pytorch_hmr
'''

import os
import random
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
import visdom
import numpy as np
from torch.utils.data import DataLoader

from config import args
from model import SPENet, SPENetSiam, SPENetBeta, SPENetPose
from datasets2 import SmplDataset

from util.util import copy_state_dict, batch_rodrigues, weights_init
from util.robustifiers import GMOf
#import pdb; pdb.set_trace()

def log(info, flag='a'):
    '''
        info: a string
        flag: w or a
    '''
    print(info)
    with open(os.path.join(args.outf, 'log.txt'), flag) as fp:
        fp.write(info + '\n')

class SPETrainer(object):
    '''
        Learning pose & shape parameters from point clouds
    '''
    def __init__(self, bTrain = True, bVal = False):
        self.bVal = bVal
        self.bTrain = bTrain
        self.lr = args.lr        
        self._init_weight_(bUniform = True)

        if args.ls == 'GMOF':
            self.loss_fun = lambda a, b: GMOf(a-b,1)
        elif args.ls == 'L1':
            self.loss_fun = lambda a, b: abs(a-b)            
        else:
            self.loss_fun = lambda a, b: (a-b)**2

        self._build_model()
        if self.bTrain: self._create_train_data_loader()
        if self.bVal: self._create_val_data_loader()

        # self.human = {}
        # self.human['female'] = load_model('./datageneration/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        # self.human['male'] = load_model('./datageneration/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    def _init_weight_(self, bUniform=True):
        if os.path.exists(args.pervertex_weight): # and not bUniform
            loaded = np.load(args.pervertex_weight)
            v = loaded['pervertex_weight'] # v belong to [0.1, 1)
            v = np.exp(v)
            #v = v/v.max()
            v = torch.from_numpy(v).float() 
            self.pervertex_weight = v.repeat(args.batch_size,1)# 6890 to (B, 6890)  
        else:
            vert_count = 6890
            self.pervertex_weight = torch.ones(args.batch_size, vert_count)
            log( 'pervertex_weight {} not exist.'.format(args.pervertex_weight))

        if bUniform:
            shape_weight = np.ones(10)
        else:
            shape_weight = np.array([10.0, 10.0, 4.0, 4.0, 4.0,
                                      1.0,  1.0, 1.0, 1.0, 1.0])
        shape_weight = torch.from_numpy(shape_weight).float() 
        self.shape_weight = shape_weight.repeat(args.batch_size,1)# 10 to (B, 10)  

        # smpl_pose_names,24 = ['Root'0, 'Left_Hip'1, 'Right_Hip'2, 'Waist'3, 'Left_Knee'4, 'Right_Knee'5, 
        # 'Upper_Waist'6(3), 'Left_Ankle'7, 'Right_Ankle'8, 'Chest'9(6), 'Left_Toe'10, 'Right_Toe'11, 
        # 'Base_Neck'12(9), 'Left_Shoulder'13(9), 'Right_Shoulder'14(9), 'Upper_Neck'15(12), 'Left_Arm'16, 'Right_Arm'17, 
        # 'Left_Elbow'18, 'Right_Elbow'19, 'Left_Wrist'20, 'Right_Wrist'21, 'Left_Finger'22, 'Right_Finger'23]
        if bUniform:
            pose_weight = np.ones(24)
        else:
            pose_weight = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 
                                1.0, 4.0, 4.0, 1.0, 8.0, 8.0,
                                1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
                                8.0, 8.0, 8.0, 8.0, 8.0, 8.0])
        pose_weight = torch.from_numpy(pose_weight).float() 
        self.pose_weight = pose_weight.repeat(args.batch_size,1)# 24 to (B, 24)  

        # joint, 19 = ['RAnkle'0, 'RKnee'1, 'RHip'2, 'LHip'3, 'LKnee'4, 'LAnkle'5,               #6
        #              'RWrist'6, 'RElbow'7, 'RShoulder'8, 'LShoulder'9, 'LElbow'10, 'LWrist'11, #6
        #              'Neck'12, Head(Top)'13, 'Noise'14 'LEye'15, 'REye'16, 'REar'17, 'LEar'18] #7
        if bUniform:
            joint_weight = np.ones(19)            
        else: 
            joint_weight = np.array([8.0, 4.0, 2.0, 2.0, 4.0, 8.0, 
                                8.0, 4.0, 2.0, 2.0, 4.0, 8.0, 
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        joint_weight = torch.from_numpy(joint_weight).float() 
        self.joint_weight = joint_weight.repeat(args.batch_size,1)# 19 to (B, 19)  

        if torch.cuda.is_available():
            self.pervertex_weight = self.pervertex_weight.cuda()   
            self.shape_weight = self.shape_weight.cuda()
            self.pose_weight = self.pose_weight.cuda()
            self.joint_weight = self.joint_weight.cuda()

    def _build_model(self):
        log('start building modle.')

        '''
            load model
        '''
        if args.network == 'SPENetPose':
            spenet = SPENetPose()
        elif args.network == 'SPENetBeta':
            spenet = SPENetBeta()
        elif args.network == 'SPENetSiam':
            spenet = SPENetSiam()
        else: #'SPENet'
            spenet = SPENet()

        #spenet.apply(weights_init)  # initialization of the weight

        model_path = args.model
        log('load pretrained model')
        if os.path.exists(model_path):
            copy_state_dict(
                spenet.state_dict(), 
                torch.load(model_path),
                prefix = 'module.'
            )
        else:
            info = 'model {} not exist!'.format(model_path)
            log(info)

        self.spenet = nn.DataParallel(spenet) # for multiple gpu
        #self.spenet = spenet # this can only use one gpu
        if torch.cuda.is_available():
            self.spenet = self.spenet.cuda()           

        # ASGD is far better than SGD for large (maybe wrong) weight decay, such as 0.9
        # normal weight decay should be 0.0001, not compared yet.
        # self.opt = torch.optim.ASGD(self.spenet.parameters(), lr=args.lr, 
        #                             #momentum=args.momentum, 
        #                             weight_decay = args.wd) 

        # adam is far better than ASGD and SGD for small weight decay, such as 0.0001.
        self.opt = torch.optim.Adam(
            self.spenet.parameters(),
            lr = self.lr,
            weight_decay = args.wd # default weight_decay in pytorch is 0, 3d-coded use 0.
        )

        # self.sche = torch.optim.lr_scheduler.StepLR(
        #     self.opt,
        #     step_size = args.step_lr,
        #     gamma = 0.9 # 0.98
        # )

        log(str(self.spenet))

    def _create_train_data_loader(self):
        data_set = SmplDataset(root = args.data_root, npts = args.point_count, train = True)
        self.len_dataset = len(data_set)
        
        log('Train dataset: {}'.format( data_set.path) )
        log('size of data_set: {}'.format( self.len_dataset) )
        self.loader = DataLoader(
            dataset = data_set,
            batch_size = args.batch_size,
            shuffle = True,
            drop_last = True,
            pin_memory = True,
            num_workers = args.workers
        )

    def _create_val_data_loader(self):
        val_data_set = SmplDataset(root = args.data_root, npts = args.point_count, train = False)
        log('Val dataset: {}'.format( val_data_set.path) )
        log('size of data_set: {}'.format( len(val_data_set)) )
        self.valloader = DataLoader(
            dataset = val_data_set,
            batch_size = args.batch_size,
            shuffle = True,
            drop_last = True,
            pin_memory = True,
            num_workers = args.workers
        )

    def _create_vis_env(self):
        # Launch visdom for visualization
        self.vis = visdom.Visdom(port=8097, env=args.vis)
        self.vis.close(None, env=args.vis)
        
        # initialize learning curve on visdom, and color for each primitive in visdom display
        self.shape_curve = self.vis.line(X=np.array([0]),
                                          Y=np.array([0]),
                                          name='shape',
                                          )
        self.pose_curve = self.vis.line(X=np.array([0]),
                                         Y=np.array([0]),
                                         name='pose',
                                         )
        self.d3d_curve = self.vis.line(X=np.array([0]),
                                    Y=np.array([0]),
                                    name='d3d',
                                    )
        self.d3d_decoded_curve = self.vis.line(X=np.array([0]),
                                    Y=np.array([0]),
                                    name='d3d_decoded',
                                    win = self.d3d_curve,
                                    )
        self.j3d_curve = self.vis.line(X=np.array([0]),
                                         Y=np.array([0]),
                                         name='j3d',
                                         )                                                                             
        self.lr_curve = self.vis.line(X=np.array([args.lr]),
                                  Y=np.array([0]),
                                  name='train_lr',
                                  )
        self.scatterGT = self.vis.scatter(
                X=np.zeros((1,3)),
                #X=np.array(np.squeeze(pts.contiguous()[0].data.cpu())),
                opts=dict(
                    markersize=0.5,
                    markercolor=np.array([20]),
                    ztickmin=-2,
                    ztickmax=2,
                    xtickmin=-2,
                    xtickmax=2,
                    ytickmin=-2,
                    ytickmax=2,
                    title='input',
                ),
                name="input",
                )  
        self.scatterSMPL = self.vis.scatter(
                X=np.zeros((1,3)),
                #X=np.array(np.squeeze(pts.contiguous()[0].data.cpu())),
                opts=dict(
                    markersize=0.5,
                    markercolor=np.array([20]),
                    ztickmin=-2,
                    ztickmax=2,
                    xtickmin=-2,
                    xtickmax=2,
                    ytickmin=-2,
                    ytickmax=2,
                    title='SMPL',
                ),
                name="SMPL",
                )                  
        self.scatterGen = self.vis.scatter(X=np.zeros((1,3)),
                    #X=np.array(np.squeeze(predict_verts.contiguous()[0].data.cpu())),
                    opts=dict(
                        markersize=0.5,
                        markercolor=np.array([20]),
                        ztickmin=-2,
                        ztickmax=2,
                        xtickmin=-2,
                        xtickmax=2,
                        ytickmin=-2,
                        ytickmax=2,
                        title='gen mesh',
                    ),
                    name="gen_mesh",
                    )                                                
        return

    def train(self):
        def save_model(msg):
            exclude_key = 'module.smpl' # smpl need not be saved.
            exclude_key1 = 'module.vertex'
            def exclude_smpl(model_dict):
                result = OrderedDict()
                for (k, v) in model_dict.items():
                    if exclude_key in k or exclude_key1 in k:
                        continue
                    result[k] = v
                return result

            title = msg['title']
            generator_save_path = os.path.join(args.outf, title + '.pkl')
            torch.save(exclude_smpl(self.spenet.state_dict()), generator_save_path)
            with open(os.path.join(args.outf, title + '.txt'), 'w') as fp:
                fp.write(str(msg))
                fp.write(str(args))

        # Launch visdom for visualization
        if args.vis != 'None':
            self._create_vis_env()
        ########################
        torch.backends.cudnn.benchmark = True
        iter_per_epoch = int(self.len_dataset/args.batch_size)

        self.val_folder = os.path.join(args.outf, 'val')
        if not os.path.exists(self.val_folder):
            os.makedirs(self.val_folder)

        for epoch in range(args.start_epoch, args.start_epoch + args.no_epoch):         
            if epoch == args.start_epoch + 20:
                self.lr = self.lr/10.0  # learning rate scheduled decay 0.0001
                self.opt = torch.optim.Adam(self.spenet.parameters(), lr=self.lr, weight_decay = args.wd) 
                args.decoded_ratio = args.decoded_ratio * 10
                #self._init_weight_(bUniform = False)
            if epoch == args.start_epoch + 90: # 90-120
                self.lr = self.lr/10.0  # learning rate scheduled decay 0.00001
                self.opt = torch.optim.Adam(self.spenet.parameters(), lr=self.lr, weight_decay = args.wd) 
                #self._init_weight_(bUniform = False)
            
            self.spenet.train()  
            for iter_index, data in enumerate(self.loader, 0): 
                pts, shapepara, posepara = data['pts'], data['shape'], data['pose']
                if torch.cuda.is_available():
                    pts, shapepara, posepara = pts.cuda(), shapepara.cuda(), posepara.cuda()
                
                outputs = self.spenet(pts, idx = data['idx'], inBeta = shapepara, inTheta = posepara)
                
                loss_dict = self._calc_loss(outputs, pts, shapepara, posepara, data)
                loss_shape, loss_pose = loss_dict["shape"], loss_dict["pose"] 
                loss_3d, loss_3d_decoded = loss_dict["3d"], loss_dict["3d_decoded"] 
                loss_j3d = loss_dict["j3d"]

                loss = loss_shape * args.shape_ratio + loss_pose * args.pose_ratio + \
                            loss_3d * args.threeD_ratio + loss_3d_decoded * args.decoded_ratio + \
                            loss_j3d * args.j3d_ratio 

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()                

                iter_msg = OrderedDict(
                    [
                        #('time',datetime.datetime.now().strftime('%m-%d-%H:%M')),
                        ('iter', '{}:{}/{}'.format(epoch, iter_index, iter_per_epoch) ),
                        ('loss', "{0:.3f}".format(loss.item()) ),                 
                        ('shape_loss',"{0:.3f}".format(loss_shape.item()* args.shape_ratio)),
                        ('pose_loss', "{0:.4f}".format(loss_pose.item()*args.pose_ratio)),
                        ('3d_loss', "{0:.5f}".format(loss_3d.item()*args.threeD_ratio)),
                        ('3d_decoded_loss', "{0:.5f}".format(loss_3d_decoded.item()*args.decoded_ratio)),
                        ('j3d_loss', "{0:.5f}".format(loss_j3d.item()* args.j3d_ratio)), 
                        ('lr', "{0:.5f}".format(self.opt.param_groups[0]['lr'])),  
                    ]
                )
                print(iter_msg)

                # UPDATE CURVES
                if args.vis != 'None' and iter_index % 50 == 0:
                    self.update_visualization_(epoch*int(self.len_dataset/args.batch_size) + iter_index, 
                                                loss_shape* args.shape_ratio, 
                                                loss_pose * args.pose_ratio,                                           
                                                loss_3d * args.threeD_ratio, 
                                                loss_3d_decoded * args.decoded_ratio, 
                                                loss_j3d * args.j3d_ratio, 
                                                pts, outputs)                
            #self.sche.step()
            # end for training an epoch
            #             
            # validation & same model
            if epoch % args.step_save == 0:
                with torch.no_grad():
                    val_i = 0
                    ave_shape, ave_pose = 0.0, 0.0
                    ave_3d, ave_3d_decoded, ave_j3d = 0.0, 0.0, 0.0
                    self.spenet.eval()
                    for data in self.valloader:
                        pts, shapepara, posepara = data['pts'], data['shape'], data['pose']                
                        name = data['name']
                        if torch.cuda.is_available():
                            pts, shapepara, posepara = pts.cuda(), shapepara.cuda(), posepara.cuda()
                                                
                        outputs = self.spenet(pts, idx = None, inBeta = shapepara, inTheta = posepara)

                        loss_dict = self._calc_loss(outputs, pts, shapepara, posepara, data)
                        loss_shape, loss_pose = loss_dict["shape"], loss_dict["pose"] 
                        loss_3d, loss_3d_decoded = loss_dict["3d"], loss_dict["3d_decoded"]                 

                        loss = loss_shape + loss_pose + loss_3d + loss_3d_decoded + loss_j3d 
                        
                        ave_shape += loss_shape
                        ave_pose += loss_pose
                        ave_3d += loss_3d
                        ave_3d_decoded += loss_3d_decoded                    
                        ave_j3d += loss_j3d

                        val_msg = OrderedDict(
                            [
                                ('loss', "{0:.4f}".format(loss.item())),                 
                                ('shape_loss',"{0:.4f}".format(loss_shape.item())),
                                ('pose_loss', "{0:.4f}".format(loss_pose.item())),
                                ('3d_loss', "{0:.5f}".format(loss_3d.item())),
                                ('3d_decoded_loss', "{0:.5f}".format(loss_3d_decoded.item())),
                                ('j3d_loss', "{0:.5f}".format(loss_j3d.item())), 
                                ('lr', "{0:8f}".format(self.opt.param_groups[0]['lr'])),  
                            ]
                        )
                        print(val_msg)

                        # if val_i == 0:
                        #     self.batch_save_obj_png(pts, verts, outputs, name, val_msg)
                        val_i += 1
                       
                    ave_shape = ave_shape / val_i
                    ave_pose = ave_pose / val_i
                    ave_3d = ave_3d / val_i
                    ave_3d_decoded = ave_3d_decoded / val_i
                    ave_j3d = ave_j3d / val_i
                    ave_loss = ave_shape + ave_pose + ave_3d + ave_3d_decoded + ave_j3d

                    val_msg = OrderedDict(
                        [
                            #('time', datetime.datetime.now().strftime('%m-%d-%H:%M')),
                            ('epoch', epoch),
                            ('ave_loss', "{0:4f}".format(ave_loss.item())),                 
                            ('ave_shape_loss',"{0:3f}".format(ave_shape.item())),
                            ('ave_pose_loss', "{0:4f}".format(ave_pose.item())),
                            ('ave_3d_loss', "{0:5f}".format(ave_3d.item())),
                            ('ave_3d_decoded_loss', "{0:5f}".format(ave_3d_decoded.item())),
                            ('ave_j3d_loss', "{0:.5f}".format(ave_j3d.item())),
                            ('lr', "{0:8f}".format(self.opt.param_groups[0]['lr'])),  
                        ]
                    )

                    val_msg['title'] = '{}_{}_{}_{}_{}_s{}_p{}_3d{}_decoded{}_j{}_r{}'.format(
                                        args.network, args.encoder, args.decoder,
                                        val_msg['epoch'], val_msg['ave_loss'], 
                                        val_msg['ave_shape_loss'], val_msg['ave_pose_loss'], 
                                        val_msg['ave_3d_loss'], val_msg['ave_3d_decoded_loss'],
                                        val_msg['ave_j3d_loss'], val_msg['lr'])
                    print(val_msg)
                    save_model(val_msg)

                    # # keep losses on the same level
                    # ave_shape = ave_shape * args.shape_ratio
                    # ave_pose = ave_pose * args.pose_ratio
                    # ave_3d = ave_3d * args.threeD_ratio
                    # ave_3d_decoded = ave_3d_decoded * args.decoded_ratio
                    # ave_j3d = ave_j3d * args.j3d_ratio
                    # ave_chamfer = ave_chamfer * args.chamfer_ratio

                    # if ave_pose/(ave_shape + 1e-8) > 10:
                    #     args.shape_ratio = args.shape_ratio * 10
                    #     print('updated shape_ratio: {}'.format(args.shape_ratio))
                    # if ave_pose/(ave_3d + 1e-8) > 10:
                    #     args.threeD_ratio = args.threeD_ratio * 10
                    #     args.decoded_ratio = args.threeD_ratio 
                    #     print('updated threeD_ratio: {}'.format(args.threeD_ratio))
                    # if ave_pose/(ave_j3d + 1e-8) > 10:
                    #     args.j3d_ratio = args.j3d_ratio * 10
                    #     print('updated j3d_ratio: {}'.format(args.j3d_ratio))                                              

        log('train finished.')

    def batch_save_obj_png(self, pts, verts, outputs, name, valmsg): 
        '''
        pts: input points
        verts: vertices GT mesh, i.e. generated mesh from shape and pose parameters of pts
        '''        

        (predict_shape, predict_pose, predict_verts, predict_j3d, decoded_verts) = outputs        
        for i in range(predict_verts.shape[0]):              
            fname = '{}_s{}_p{}_3d{}_j3d{}_decoded{}'.format(name[i], 
                                            valmsg['shape_loss'], valmsg['pose_loss'],
                                            valmsg['3d_loss'], valmsg['j3d_loss'], valmsg['3d_decoded_loss'])

            self.spenet.module.smpl.save_obj(predict_verts[i].detach().cpu().numpy(), 
                                      os.path.join(self.val_folder, fname + '.obj'))
            self.spenet.module.smpl.save_obj(verts[i].detach().cpu().numpy(), 
                                      os.path.join(self.val_folder, name[i] + '.obj'))
            np.save(os.path.join(self.val_folder, name[i] + 'j3d.npy'), predict_j3d[i].detach().cpu().numpy())                                                                            

            if args.decoder != 'None':
                self.spenet.module.smpl.save_png(pts[i].detach().cpu().numpy(), 
                                            predict_verts[i].detach().cpu().numpy(),
                                            decoded_verts[i].detach().cpu().numpy(), 
                                            os.path.join(self.val_folder, fname + '.png'))                                        
            else:
                self.spenet.module.smpl.save_png(pts[i].detach().cpu().numpy(), 
                            verts[i].detach().cpu().numpy(), #predict_verts[i].detach().cpu().numpy(),
                            predict_verts[i].detach().cpu().numpy(), 
                            os.path.join(self.val_folder, fname + '.png'))

    def batch_shape_l2_loss(self, real_shape, fake_shape):
        '''
            purpose:
                calc mse * 0.5

            Inputs:
                real_shape  :   (B, 10)
                fake_shape  :   (B, 10)
        '''
        k = real_shape.shape[0] * 10.0
        shape_dif = self.loss_fun(real_shape, fake_shape) * self.shape_weight
        return  shape_dif.sum() / k

    def batch_pose_l2_loss_matrix(self, real_pose, fake_pose):
        '''
            compute pose parameter loss for each joint, 24 joints in total, 
                    except for the global joint, i.e. 1st joint, via rotation matrix representation.

            Input:
                real_pose   : N x 72
                fake_pose   : N x 72
            
            L2 of elements of Rotation matrix is used instead of angle difference
        '''
        B = real_pose.shape[0]
        k = B * 207.0
        rq, real_rs = batch_rodrigues(real_pose.contiguous().view(-1, 3))
        real_rs = real_rs.contiguous().view(-1, 24, 9)[:,1:,:]

        fq, fake_rs = batch_rodrigues(fake_pose.contiguous().view(-1, 3))
        fake_rs = fake_rs.contiguous().contiguous().view(-1, 24, 9)[:,1:,:]

        pose_weight = self.pose_weight.unsqueeze(2).expand(B, 24, 9)
        tmp = pose_weight[:,1:,:].contiguous().view(-1, 207)
        dif_rs = self.loss_fun(real_rs, fake_rs).view(-1, 207) * tmp
        return dif_rs.sum() / k

    def batch_pose_l2_loss_quad(self, real_pose, fake_pose) -> torch.Tensor:
        '''
            compute pose parameter loss for each joint, 24 joints in total, 
                    except for the global joint, i.e. 1st joint, via quaternion representation.
                    Note: Once the global joint is considered, the loss will keep high. Why? todo
            
            before: L2 of elements of Rotation matrix is used instead of angle difference
            now: L2 of elements of quaternion instead of matrix

            By construction (batch_rodrigues(theta) in util.py), quaternions are unit-norm.

            Metrics for 3D Rotations: Comparison and Analysis
                diff(q,s) = abs( q.dot(s) ) => dist(q,s) = 1 - diff(q,s)

            todo: 
            improve it? 3D Pose Regression using Convolutional Neural Networks


            Input:
                real_pose   : B x 72, 72 = 24 * 3
                fake_pose   : B x 72

        '''
        # 96 = 24*4, 24 for global rotation + 23 joints; 4 for quaternion
        # 207 = 23 * 9, 23 for 23 joints (without global rotation); 9 for rotation matrix; 
        npara = 92
        k = real_pose.shape[0] * npara 

        real_rot, _ = batch_rodrigues( real_pose.contiguous().view(-1, 3) )
        #real_rot = real_rot.contiguous().view(-1, 24, 4)[:,1:,:] # not compare global rotation
        real_rot = real_rot.contiguous().view(-1, 24, 4)[:,1:,:]
        fake_rot, _ = batch_rodrigues( fake_pose.contiguous().view(-1, 3) )
        fake_rot = fake_rot.contiguous().view(-1, 24, 4)[:,1:,:]
        
        #1 - abs(torch.dot(real_rot, fake_rot))
        c = torch.bmm(real_rot, fake_rot.transpose(1,2)) 
        a_dot_b = c[:,:,0]
        dif_rs = abs(a_dot_b)
        return dif_rs.sum() / k

    def batch_3d_l2_loss(self, verts, predict_verts, idx = None):
        '''
            verts: (B, 6890) if idx == None, else (B, 2400)
            idx: (B, 2400)
        '''
        B = verts.shape[0]
        if idx is None:
            N = verts.shape[1]
            #dif = self.loss_fun(verts, predict_verts)
            weight = self.pervertex_weight # (B, 6890)
        else:
            # todo select tensor element using an indexing tensor
            N = idx.shape[1]
            #dif = self.loss_fun( verts[idx.unsqueeze(2).expand(B, idx.shape[1], 3) ], predict_verts)
            #dif = self.loss_fun( verts[idx, :], predict_verts)
            weight = self.pervertex_weight[0]
            idx = idx.view(-1)
            weight = weight[idx].view(B, N)
        
        dif = self.loss_fun(verts, predict_verts)
        dif = dif.sum(2) * weight  # B x 6890 x 3 => B x 6890 x 1
        k = B * N * 3    
        return  dif.sum() / k

    def batch_j3d_l2_loss(self, j3d, predict_j3d):
        B = j3d.shape[0]
        N = j3d.shape[1]
        k = B * N * 3       

        joint_weight = self.joint_weight.unsqueeze(2).expand(B, N, 3)
        tmp = joint_weight.contiguous().view(-1, N*3)
        dif = self.loss_fun(j3d, predict_j3d).view(-1, N*3) * tmp
        return  dif.sum() / k

    def _calc_loss(self, outputs, pts, real_shape, real_pose, data): 
        '''
            data['idx']: (B, 2400)
            pts: (B, 2400, 3)
            predict_verts: (B, 6890, 3)
            decoded_verts: (B, 2400, 3)
        '''

        (predict_shape, predict_pose, predict_verts, predict_j3d, decoded_verts) = outputs
    
        loss_shape = self.batch_shape_l2_loss(real_shape, predict_shape)             
        loss_pose = self.batch_pose_l2_loss_matrix(real_pose, predict_pose)                 

        verts, j3d = data['verts'], data['j3d']
        if torch.cuda.is_available():
            verts, j3d = verts.cuda(), j3d.cuda()

        loss_3d = self.batch_3d_l2_loss(verts, predict_verts)
        if args.decoder == 'None':
            loss_3d_decoded = torch.tensor(0.0)
            if torch.cuda.is_available():
                loss_3d_decoded = loss_3d_decoded.cuda()
        else:
            if self.spenet.training:
                loss_3d_decoded = self.batch_3d_l2_loss(pts, decoded_verts, data['idx'])
            else:
                loss_3d_decoded = self.batch_3d_l2_loss(verts, decoded_verts)
        
        loss_j3d = self.batch_j3d_l2_loss(j3d, predict_j3d)
             
       
        loss =	{"shape": loss_shape,
                    "pose": loss_pose,
                    "3d": loss_3d,
                    "3d_decoded": loss_3d_decoded,
                    "j3d": loss_j3d,
                    }
        return loss

    def update_visualization_(self, iter_index, loss_shape, loss_pose, 
                                loss_3d, loss_3d_decoded, loss_j3d,
                                pts, outputs):
        str_update = 'append'
        if iter_index == args.start_epoch * self.len_dataset + 1:
            str_update = 'replace'

        if loss_shape.item() != 0:
            self.vis.line(
                X=np.array([iter_index]),
                Y=np.array([loss_shape.item()]),
                win= self.shape_curve,
                opts=dict(title='shape_loss'),
                update=str_update,
            )

        if loss_pose.item() != 0:           
            self.vis.line(
                X=np.array([iter_index]),
                Y=np.array([loss_pose.item()]),
                win=self.pose_curve,
                opts=dict(title='pose_loss'),
                update=str_update,
            )
        if loss_3d.item() != 0:
            self.vis.line(
                X=np.array([iter_index]),
                Y=np.array([loss_3d.item()]),
                name='d3d',
                win=self.d3d_curve,
                #opts=dict(title='d3d_loss'),
                update=str_update,
            )
        #if loss_3d_decoded.item() != 0:
        self.vis.line(
            X=np.array([iter_index]),
            Y=np.array([loss_3d_decoded.item()]),
            name='d3d_decoded',
            win=self.d3d_curve,
            #opts=dict(title='d3d_decoded_loss'),
            update=str_update,
        )

        if loss_j3d.item() != 0:
            self.vis.line(
                X=np.array([iter_index]),
                Y=np.array([loss_j3d.item()]),
                win=self.j3d_curve,
                opts=dict(title='j3d_loss'),
                update=str_update,
            )                                        

        if self.opt.param_groups[0]['lr'] != 0:
            self.vis.line(
                X=np.array([iter_index]),
                Y=np.array([self.opt.param_groups[0]['lr']]),
                win=self.lr_curve,
                opts=dict(title='lr'),
                update=str_update,
            )

        # vis the train shapes
        # if iter_index % 50 == 0:
        (predict_shape, predict_pose, predict_verts, predict_j3d, decoded_verts) = outputs 
                     
        self.vis.scatter(X=np.array(np.squeeze(pts[0].data.cpu())),
                    win = self.scatterGT,
                    opts=dict(
                        markersize=0.5,
                        title='input',
                    ),
                    )

        self.vis.scatter(X=np.array(predict_verts[0].data.cpu()),
                    win = self.scatterSMPL,
                    opts=dict(
                        markersize=0.5,
                        title='SMPL',
                    ),
                    )  
        if args.decoder != 'None':
            self.vis.scatter(X=np.array(decoded_verts[0].data.cpu()),
                        win = self.scatterGen,
                        opts=dict(
                            markersize=0.5,
                            title='decoded',
                        ),
                        )
       

if __name__ == '__main__':
    args.outf = '{}_{}_{}_{}_{}'.format(args.outf, args.network, args.encoder, args.decoder, datetime.datetime.now().strftime('%m-%d-%H:%M'))

    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    log(str(args), 'w')

    args.manualSeed = random.randint(1, 10000) # fix seed
    log('{}: {}'.format('Random Seed: ', args.manualSeed))
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    trainer = SPETrainer(bTrain=True, bVal=True)
    trainer.train()
