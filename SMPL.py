
'''
    file:   SMPL.py
    adapted by Junjie Cao @ 2018.09

    date:   2018_05_03
    author: zhangxiong(1025679612@qq.com)
    mark:   the algorithm is cited from original SMPL
'''
import sys
import json
import numpy as np
import trimesh
import torch
import torch.nn as nn
from config import args
from util.util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose, load_mean_smpl_parameters, center_pts_np_array


class SMPL(nn.Module):
    def __init__(self, model_path, b_center = True, joint_type = 'cocoplus', obj_saveable = False):
        super(SMPL, self).__init__()

        if joint_type not in ['cocoplus', 'lsp']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(joint_type)
            sys.exit(msg)

        self.model_path = model_path
        self.joint_type = joint_type
        with open(model_path, 'r') as reader:
            model = json.load(reader)
        
        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None

        # v_template: Mean template vertices
        np_v_template = np.array(model['v_template'], dtype = np.float)
        if b_center:
            np_v_template, translation = center_pts_np_array( np_v_template)

        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
         # Size of mesh [Number of vertices 6980, 3]
        self.size = [np_v_template.shape[0], 3]

        # Shape blend shape basis: 6980 x 3 x 10
        # reshaped to 206700(6980*30) x 10, transposed to 10x(6980*30)
        np_shapedirs = np.array(model['shapedirs'], dtype = np.float)
        self.num_betas = np_shapedirs.shape[-1] #10
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        # Regressor for joint locations given shape - 6890 x 24
        np_J_regressor = np.array(model['J_regressor'], dtype = np.float)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        np_posedirs = np.array(model['posedirs'], dtype = np.float)
        num_pose_basis = np_posedirs.shape[-1]
        # 207 x 20670
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        # indices of parents for each joints
        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        # LBS weights
        np_weights = np.array(model['weights'], dtype = np.float)
        vertex_count = np_weights.shape[0] 
        vertex_component = np_weights.shape[1]
        np_weights = np.tile(np_weights, (args.batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))

        # This returns 19 keypoints: 6890 x 19
        np_joint_regressor = np.array(model['cocoplus_regressor'], dtype = np.float)
        if joint_type == 'lsp':
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor[:, :14]).float())
        else:
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())


        self.register_buffer('e3', torch.eye(3).float())
        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if not self.faces:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

            for f in self.faces: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )
    
    def save_png(self, inpts, verts, pred_verts, png_name):
        '''
            pred_verts: 6890*3
            verts: 6890*3
        '''       
        if not self.faces:
            msg = 'obj not saveable!'
            sys.exit(msg)


        gtmesh = trimesh.Trimesh(verts, self.faces)
        for i in range(len(gtmesh.faces)):
            gtmesh.visual.face_colors[i] = np.array([0,255,0,50], dtype = np.uint8)
        scene = gtmesh.scene()      

        mesh = trimesh.Trimesh(pred_verts, self.faces)
        for i in range(len(mesh.faces)):
            mesh.visual.face_colors[i] = np.array([0,0,255,150], dtype = np.uint8)
        scene.add_geometry(mesh)


        samples = trimesh.PointCloud(inpts)
        scene.add_geometry(samples)

        #scene.show()
        try:
            png = scene.save_image(resolution=[800, 800],visible=True)
            with open(png_name, 'wb') as f:
                f.write(png)
                f.close()

        except BaseException as E:
            print("unable to save image", str(E))        

    def forward(self, beta, theta, get_skin = False):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: B x 10
          theta: B x 72 (with 3-D axis-angle rep)
        Updates:
        self.J_transformed: B x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: B x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: B x 6980 x 3
          - Rs: B x 24 x 3 x 3. rotation matrix for each joint
        """
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]

        # 1. Add shape blend shapes
        # (B x 10) x (10 x 6890*3) = B x 6890 x 3
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor) # (1x6890) * (6890 x 24)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)

        # 3. Add pose blend shapes
        # B x 24 x 3 x 3
        Rs_quad, Rs = batch_rodrigues(theta.view(-1, 3))
        Rs = Rs.view(-1, 24, 3, 3) # theta(1*72)

        # theta.continguous().view(-1, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207) # todo, why? jjcao
        # (B x 207) x (207, 20670) -> B x 6890 x 3
        # todo it is impossibble to change this into quaternion, since posedirs are rotation matrix based. 
        # reodo this is hard. 
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        # 5. Do skinning:
        # W is B x 6890 x 24
        weight = self.weight[:num_batch]
        W = weight.view(num_batch, -1, 24)
        # (B x 6890 x 24) x (B x 24 x 16)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
        
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]# (B x 6890 x 4 x 1) =>  (B x 6890 x 3)

        # Get cocoplus or lsp joints:
        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

def generate(beta, pose, smpl, outfile, outimg):
    vpose = torch.tensor(np.array([pose])).float()
    if torch.cuda.is_available():
        vpose = vpose.cuda()
    shapepara = torch.tensor(np.array([beta])).float()
    if torch.cuda.is_available():
        shapepara = shapepara.cuda() 

    verts, j, r = smpl(shapepara, vpose, get_skin = True)
    
    #tmp = smpl.v_template.cpu().numpy()
    tmp = verts[0].cpu().numpy()
    smpl.save_obj(tmp, outfile) 
    smpl.save_png(tmp, tmp, tmp, outimg) 

    return verts, j

def test_rotation(smpl, outpath):
    ##############
    #  generate input points: standard zero pose (T pose) facing z+, heading y+
    # beta = np.zeros(10)
    # pose = np.zeros(72)
    mean = load_mean_smpl_parameters(args.smpl_mean_theta_path)
    beta = mean[72:]
    pose = mean[:72] # pose[0:3]==0.0 for this mean

    vpose = torch.tensor(np.array([pose])).float()
    if torch.cuda.is_available():
        vpose = vpose.cuda()
    shapepara = torch.tensor(np.array([beta])).float()
    if torch.cuda.is_available():
        shapepara = shapepara.cuda() 

    inpts, j, r = smpl(shapepara, vpose, get_skin = True)    
    smpl.save_obj(inpts[0].cpu().numpy(), os.path.join(outpath, 'rot_in.obj'))

    ################
    #  rotate input points via smpl
    vpose[0,0]= 0.5 * np.pi # around x axis    
    #vpose[0,1] = 0.5 * np.pi #pose[1]= 0.5 * np.pi
    #vpose[0,2]= 0.5 * np.pi
    
    verts, j1, r = smpl(shapepara, vpose, get_skin = True)
    smpl.save_obj(verts[0].cpu().numpy(), os.path.join(outpath, 'rot_in_via_smpl.obj'))

    ################
    #  rotate input points via rotation matrix    
    Rs = batch_rodrigues(vpose[0,:3].view(-1,3)).view(1,3,3)# acutally Rs == r[0,0,:,:]
    
    from torch.autograd import Variable
    def make_A(R, t):
        N = R.shape[0]
        # Rs is N x 3 x 3, ts is N x 3 x 1
        R_homo = nn.functional.pad(R, [0, 0, 0, 1, 0, 0])
        if t.is_cuda:
            t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).cuda()], dim = 1)
        else:
            t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1))], dim = 1)
        return torch.cat([R_homo, t_homo], 2)    
        
    tmpA = smpl.J_transformed[0,0,:]
    #tmpA[1]=0 #y
    #tmpA[0]=0    
    A = make_A(Rs, tmpA.view(1,3,1))    
    A = A.transpose(1,2)

    inpts_homo = torch.cat([inpts, torch.ones(Rs.shape[0], inpts.shape[1], 1, device = smpl.cur_device)], dim = 2)        
    rpts_homo = torch.matmul(inpts_homo, A)
    rpts = rpts_homo[:, :, :3]# (N x 6890 x 4) =>  (N x 6890 x 3)
    
    smpl.save_obj(rpts[0].cpu().numpy(), os.path.join(outpath, 'rot_in_via_matrix.obj'))
    
    ################
    #  rotate input points via rotation matrix B    
    tmpB = smpl.J_transformed[0,0,:]
    #tmpB[1]=0
    #tmpB[0]= 10* tmpB[0]
    tmpB[2]=0
    B = make_A(Rs, tmpB.view(1,3,1)) 
    B = B.transpose(1,2)
    rpts_homo = torch.matmul(inpts_homo, B)
    rpts = rpts_homo[:, :, :3]# (N x 6890 x 4) =>  (N x 6890 x 3)    
    smpl.save_obj(rpts[0].cpu().numpy(), os.path.join(outpath, 'rot_in_via_matrix_b.obj'))
    

    smpl.save_png(inpts[0].cpu().numpy(), rpts[0].cpu().numpy(), verts[0].cpu().numpy(), 
                    os.path.join(outpath, 'rot_in.png')) 
    return

def test_smpl_joints(smpl, outpath):
    # standard zero pose (T pose) facing z+, heading y+
    beta = np.zeros(10)
    pose = np.zeros(72)
    verts, joints = generate(beta, pose, smpl, 
                            os.path.join(outpath, 'SMPL_T_pose.obj'),
                            os.path.join(outpath, 'SMPL_T_pose.png'))

    # parents, 24 = [-1 0 0 0 1 2 
    # 3  4  5  6  7  8 
    # 9  9  9  12 13 14 
    # 16 17 18 19 20 21]
    # smpl_pose_names,24 = ['Root'0, 'Left_Hip'1, 'Right_Hip'2, 'Waist'3, 'Left_Knee'4, 'Right_Knee'5, 
    # 'Upper_Waist'6(3), 'Left_Ankle'7, 'Right_Ankle'8, 'Chest'9(6), 'Left_Toe'10, 'Right_Toe'11, 
    # 'Base_Neck'12(9), 'Left_Shoulder'13(9), 'Right_Shoulder'14(9), 'Upper_Neck'15(12), 'Left_Arm'16, 'Right_Arm'17, 
    # 'Left_Elbow'18, 'Right_Elbow'19, 'Left_Wrist'20, 'Right_Wrist'21, 'Left_Finger'22, 'Right_Finger'23]
    # pose_weight, 24 = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 
    #                    1.0, 4.0, 4.0, 1.0, 4.0, 4.0,
    #                    1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
    #                    8.0, 8.0, 8.0, 8.0, 4.0, 4.0]

    # joint, 19 = ['RAnkle'0, 'RKnee'1, 'RHip'2, 'LHip'3, 'LKnee'4, 'LAnkle'5,               #6
    #              'RWrist'6, 'RElbow'7, 'RShoulder'8, 'LShoulder'9, 'LElbow'10, 'LWrist'11, #6
    #              'Neck'12, Head(Top)'13, 'Noise'14 'LEye'15, 'REye'16, 'REar'17, 'LEar'18] #7
    # joint_weight, 19 = [8.0, 4.0, 2.0, 2.0, 4.0, 8.0, 
    #                     8.0, 4.0, 2.0, 2.0, 4.0, 8.0, 
    #                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    mesh = trimesh.Trimesh(verts[0].cpu().numpy(), smpl.faces)
    for i in range(len(mesh.faces)):
        mesh.visual.face_colors[i] = np.array([0,255,0,50], dtype = np.uint8)
    scene = mesh.scene()      

    colors=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 
            [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    colors = np.array(colors, dtype = np.float)*255.0     

    joints = joints[0].cpu().numpy()
    samples = trimesh.PointCloud(joints)#, colors)
    #samples.colors = [255.0, 0, 0, 255]
    samples.colors = colors
    scene.add_geometry(samples)
    scene.show() #samples.show()

    return

def test1_center_or_not(outpath):
    smpl = SMPL(args.smpl_model, b_center = False, obj_saveable = True)
    smpl_centered = SMPL(args.smpl_model, b_center = True, obj_saveable = True)
    if torch.cuda.is_available():
        smpl = smpl.cuda()
        smpl_centered = smpl_centered.cuda()
    
    # # standard zero pose (T pose) facing z+, heading y+
    # beta = np.zeros(10)
    # pose = np.zeros(72)
    # generate(beta, pose, smpl, 
    #                         os.path.join(outpath, 'SMPL_T_pose.obj'),
    #                         os.path.join(outpath, 'SMPL_T_pose.png'))
    # generate(beta, pose, smpl_centered, 
    #                         os.path.join(outpath, 'SMPL_T_pose_centerB.obj'),
    #                         os.path.join(outpath, 'SMPL_T_pose_centerB.png'))

    # # mean pose
    # mean = load_mean_smpl_parameters(args.smpl_mean_theta_path)
    # beta = mean[72:]
    # pose = mean[:72]
    # generate(beta, pose, smpl, 
    #                         os.path.join(outpath, 'SMPL_mean_pose.obj'),
    #                         os.path.join(outpath, 'SMPL_mean_pose.png'))
    # generate(beta, pose, smpl_centered, 
    #                         os.path.join(outpath, 'SMPL_mean_pose_centerB.obj'),
    #                         os.path.join(outpath, 'SMPL_mean_pose_centerB.png'))

    # a pose facing x+, heading z+, and mainly locate at y-
    pose= np.array([
        1.22162998e+00,   1.17162502e+00,   1.16706634e+00,
        -1.20581151e-03,   8.60930011e-02,   4.45963144e-02,
        -1.52801601e-02,  -1.16911056e-02,  -6.02894090e-03,
        1.62427306e-01,   4.26302850e-02,  -1.55304456e-02,
        2.58729942e-02,  -2.15941742e-01,  -6.59851432e-02,
        7.79098943e-02,   1.96353287e-01,   6.44420758e-02,
        -5.43042570e-02,  -3.45508829e-02,   1.13200583e-02,
        -5.60734887e-04,   3.21716577e-01,  -2.18840033e-01,
        -7.61821344e-02,  -3.64610642e-01,   2.97633410e-01,
        9.65453908e-02,  -5.54007106e-03,   2.83410680e-02,
        -9.57194716e-02,   9.02515948e-02,   3.31488043e-01,
        -1.18847653e-01,   2.96623230e-01,  -4.76809204e-01,
        -1.53382001e-02,   1.72342166e-01,  -1.44332021e-01,
        -8.10869411e-02,   4.68325168e-02,   1.42248288e-01,
        -4.60898802e-02,  -4.05981280e-02,   5.28727695e-02,
        3.20133418e-02,  -5.23784310e-02,   2.41559884e-03,
        -3.08033824e-01,   2.31431410e-01,   1.62540793e-01,
        6.28208935e-01,  -1.94355965e-01,   7.23800480e-01,
        -6.49612308e-01,  -4.07179184e-02,  -1.46422181e-02,
        4.51475441e-01,   1.59122205e+00,   2.70355493e-01,
        2.04248756e-01,  -6.33800551e-02,  -5.50178960e-02,
        -1.00920045e+00,   2.39532292e-01,   3.62904727e-01,
        -3.38783532e-01,   9.40650925e-02,  -8.44506770e-02,
        3.55101633e-03,  -2.68924050e-02,   4.93676625e-02],dtype = np.float)
    pose[0:3]=0.0 # cancel global transformation    
    beta = np.array([10.25349993,  0.25009069,  0.21440795,  0.78280628,  0.08625954,
            0.28128183,  0.06626327, -0.26495767,  0.09009246,  0.06537955 ])
    
    generate(beta, pose, smpl, 
                            os.path.join(outpath, 'SMPL_some_pose.obj'),
                            os.path.join(outpath, 'SMPL_some_pose.png'))
    generate(beta, pose, smpl_centered, 
                            os.path.join(outpath, 'SMPL_some_pose_center.obj'),
                            os.path.join(outpath, 'SMPL_some_pose_center.png'))

def batch_pose_l2_loss_matrix():
    fake_pose = np.zeros(72)
    mean = load_mean_smpl_parameters(args.smpl_mean_theta_path)
    real_pose = mean[:72]

    fake_pose = torch.tensor(fake_pose).float()
    real_pose = torch.tensor(real_pose).float()
    fake_pose = fake_pose.unsqueeze(0).expand(2, -1).contiguous()
    real_pose = real_pose.unsqueeze(0).expand(2, -1).contiguous()

    from util.util import batch_rodrigues        
    B = real_pose.shape[0]
    k = B * 207.0
    rq, real_rs = batch_rodrigues(real_pose.contiguous().view(-1, 3))
    real_rs = real_rs.contiguous().view(-1, 24, 9)[:,1:,:]

    fq, fake_rs = batch_rodrigues(fake_pose.contiguous().view(-1, 3))
    fake_rs = fake_rs.contiguous().contiguous().view(-1, 24, 9)[:,1:,:]

    pose_weight = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 
                            1.0, 4.0, 4.0, 1.0, 4.0, 4.0,
                            1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
                            8.0, 8.0, 8.0, 8.0, 4.0, 4.0])
    pose_weight = torch.from_numpy(pose_weight).float() 
    pose_weight = pose_weight.unsqueeze(0).expand(2, -1).contiguous()

    pose_weight = pose_weight.unsqueeze(2).expand(B, 24, 9)
    tmp = pose_weight[:,1:,:].contiguous().view(-1, 207)

    dif_rs = abs(real_rs - fake_rs).view(-1, 207) * tmp
    return dif_rs.sum() / k
        
if __name__ == '__main__':
    import os
    args.batch_size = 4
    DATA_ROOT = '../../data'    
    posepath = os.path.join(DATA_ROOT, 'smpl_data_test')
    outpath = os.path.join(DATA_ROOT, 'spe_data_test/mesh') 
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    #batch_pose_l2_loss_matrix()
    test1_center_or_not(outpath)

    # smpl = SMPL(args.smpl_model, obj_saveable = True) #'./model/neutral_smpl_with_cocoplus_reg.txt'
    # if torch.cuda.is_available():
    #     smpl = smpl.cuda() 

    #test_rotation(smpl, outpath)
    #test_smpl_joints(smpl, outpath)    