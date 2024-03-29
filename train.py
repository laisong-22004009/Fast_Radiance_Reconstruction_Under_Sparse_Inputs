import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, vgce
from lib.load_data import load_data

import os
os.environ['CUDA_VISIBLE_DEVICES']= "1"


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export python  scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    parser.add_argument("--export_pointcloud", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')

    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, gt_masks=None, savedir=None, render_factor=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    disps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    masked_psnrs = []

    imageio_ssims = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = vgce.get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'disp', 'mask', 'depth']
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(16, 0), rays_d.split(16, 0), viewdirs.split(16, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks])
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        disp = render_result['disp'].cpu().numpy()

        depth = render_result['depth'].cpu().numpy()
        depth = depth / depth.max()

        # mask_render = render_result['mask'].cpu().numpy()
        # mask_render = np.sum(mask_render, axis=2)
        # mask_render = mask_render / mask_render.max()
        # print(mask_render)

        rgbs.append(rgb)
        disps.append(disp)
        if i==0:
            print('Testing', rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))


        if gt_imgs is not None and render_factor==0 and gt_masks is not None:
            mask = gt_masks[i]
            mask_bin = (mask == 1.)
            mse = ((gt_imgs[i] - rgb)[mask_bin]**2).mean()
            masked_p = -10. * np.log10(mse)
            masked_psnrs.append(masked_p)


        if savedir is not None:
            rgb8 = utils.to8b(rgbs[-1])
            disp8 = utils.to8b(disps[-1])
            filename_rgb = os.path.join(savedir, '{:03d}.png'.format(i))
            filename_dsip = os.path.join(savedir, '{:03d}_disp.png'.format(i))
            imageio.imwrite(filename_rgb, rgb8)
            imageio.imwrite(filename_dsip, disp8)

            # mask8 = utils.to8b(mask_render)
            # filename_mask = os.path.join(savedir, '{:03d}_mask.png'.format(i))
            # imageio.imwrite(filename_mask, mask8)

            depth8 = utils.to8b(depth)
            # zero_depth_mask = (depth8 > 200)
            # depth8 = depth8 * (1-zero_depth_mask)
            # import matplotlib.pyplot as plt
            # plt.imshow(depth8, cmap='gray')
            # plt.show()



            filename_depth = os.path.join(savedir, '{:03d}_depth.png'.format(i))
            imageio.imwrite(filename_depth, depth8)

    rgbs = np.array(rgbs)
    disps = np.array(disps)
    if len(psnrs):
        '''
        print('Testing psnr', [f'{p:.3f}' for p in psnrs])
        if eval_ssim: print('Testing ssim', [f'{p:.3f}' for p in ssims])
        if eval_lpips_vgg: print('Testing lpips (vgg)', [f'{p:.3f}' for p in lpips_vgg])
        if eval_lpips_alex: print('Testing lpips (alex)', [f'{p:.3f}' for p in lpips_alex])
        '''
        print('Testing psnr', np.mean(psnrs), '(avg)')
        print('Testing masked psnr', np.mean(masked_psnrs), '(avg)')
        print('Testing ssim (imageio)', np.mean(imageio_ssims), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

        result_list = []
        result_list.append(np.mean(psnrs))
        result_list.append(np.mean(ssims))
        result_list.append(np.mean(lpips_vgg))
        result_list.append(np.mean(lpips_alex))
        result = np.array(result_list)
        txt_path = os.path.join(savedir, 'result_all.txt')
        np.savetxt(txt_path, result, fmt="%.3f")

    return rgbs, disps


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
        'hwf', 'HW', 'Ks', 'near', 'far',
        'i_train', 'i_val', 'i_test', 'irregular_shape',
        'poses', 'render_poses', 'images', 'masks'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = vgce.get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w,
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale) and reload_ckpt_path is None:
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
    model = vgce.VoxelGrid(
        xyz_min=xyz_min, xyz_max=xyz_max,
        num_voxels=num_voxels,
        mask_cache_path=coarse_ckpt_path,
        **model_kwargs)
    if cfg_model.maskout_near_cam_vox:
        model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    model = model.to(device)

    # init optimizer
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    # load checkpoint if there is
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = vgce.get_training_rays_in_maskcache_sampling(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train],
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = vgce.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = vgce.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = vgce.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)

            # low-density initialization
            with torch.no_grad():
                model.density[cnt <= 2] = -100

        per_voxel_init()

    vh_initial = False
    sparse = False
    if stage == 'fine' and vh_initial == True:
        from VH_fit import vh_fit
        import open3d as o3d
        voxel_shape = model.density.shape
        vh_masks = data_dict["masks"][data_dict['i_train']]
        vh_poses = data_dict["poses"][data_dict['i_train']]
        vh_Ks = data_dict["Ks"][data_dict['i_train']]
        xyz_min = model.xyz_min
        xyz_max = model.xyz_max
        VH_bool = vh_fit(voxel_shape, xyz_max, xyz_min, vh_masks, vh_poses, vh_Ks)

        # 计算所有坐标点
        with torch.no_grad():
            VH_bool = torch.from_numpy(VH_bool)
            model.density[VH_bool[None, None] == 0] = -100

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            model.scale_volume_grid(model.num_voxels * 2)
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.density.data.sub_(1)

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, global_step=global_step, **render_kwargs)

        if sparse == True:
            activate = F.softplus(model.density + model.act_shift)
            grad_x = torch.pow((activate[:, :, :, :, :-1] - activate[:, :, :, :, 1:]), 2)
            grad_y = torch.pow((activate[:, :, :, :-1, :] - activate[:, :, :, 1:, :]), 2)
            grad_z = torch.pow((activate[:, :, :-1, :, :] - activate[:, :, 1:, :, :]), 2)

            sparse_loss = grad_x[:, :, :-1, :-1, :]+ grad_y[:, :, :-1, :, :-1] + grad_z[:, :, :, :-1, :-1]
            sparse_loss = sparse_loss.mean()


        # print(sparse_loss)
        import open3d as o3d
        show = False
        if global_step % 10000 == 0 and show == True:
            voxel_shape = model.density.shape
            xyz_min = model.xyz_min.cpu().numpy()
            xyz_max = model.xyz_max.cpu().numpy()
            xyz = np.stack(np.meshgrid(
                np.linspace(xyz_min[0], xyz_max[0], voxel_shape[2]),
                np.linspace(xyz_min[1], xyz_max[1], voxel_shape[3]),
                np.linspace(xyz_min[2], xyz_max[2], voxel_shape[4]),
                indexing="ij"), -1)
            object_density = model.density.squeeze(0).squeeze(0)
            object_activate = object_density.cpu().detach().numpy()
            result_bool = (object_activate > 0.1)
            if result_bool.sum() > 0:
                points_result = xyz[result_bool.nonzero()]
                pcd_result = o3d.geometry.PointCloud()
                pcd_result.points = o3d.utility.Vector3dVector(points_result)
                pcd_result.paint_uniform_color([0, 1.0, 0])
                o3d.visualization.draw_geometries([pcd_result])


        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        if sparse==True:
            loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target) + 0.05 * sparse_loss
        else:
            loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach()).item()
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_cum'][...,-1].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target.unsqueeze(-2)).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
            loss += cfg_train.weight_rgbper * rgbper_loss
        if cfg_train.weight_tv_density>0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0:
            loss += cfg_train.weight_tv_density * model.density_total_variation()
        if cfg_train.weight_tv_k0>0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0:
            loss += cfg_train.weight_tv_k0 * model.k0_total_variation()
        loss.backward()
        optimizer.step()
        psnr_lst.append(psnr)

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            if sparse == True:
                tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                           f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                           f'Eps: {eps_time_str} / '  f'Sparse_loss: {sparse_loss}')
            else:
                tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                           f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f}')
            psnr_lst = []

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'MaskCache_kwargs': model.get_MaskCache_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'MaskCache_kwargs': model.get_MaskCache_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):

    args.eval_ssim = True
    args.eval_lpips_alex = True
    args.eval_lpips_vgg = True

    # init
    print('train: start')
    vh_voxel_shape = [100, 100, 100]
    vh_xyz_min = np.array([-1, -1, -1])
    vh_xyz_max = np.array([1, 1, 1])
    from lib.vh_fit import vh_fit_color
    from copy import deepcopy
    Ks,  i_train, poses, images, masks = [
        data_dict[k] for k in [
            'Ks', 'i_train', 'poses', 'images', 'masks'
        ]
    ]

    Ks_vh = deepcopy(Ks)
    i_train_vh = deepcopy(i_train)
    poses_vh = deepcopy(poses)
    images_vh = deepcopy(images)
    print(type(images_vh))
    masks_vh = deepcopy(masks)

    masks_vh = masks_vh[..., np.newaxis]
    images_vh_np = images_vh.cpu().numpy()
    poses_vh_np = poses_vh.cpu().numpy()
    print(Ks_vh[0] == Ks_vh[1])
    vh_bool, pcd_100, xyz_min, xyz_max = vh_fit_color(vh_voxel_shape, vh_xyz_max, vh_xyz_min, masks_vh[i_train_vh], poses_vh_np[i_train_vh], Ks_vh[0],
                                                        images_vh_np[i_train_vh])
    print('train: xyz_min_vh', xyz_min)
    print('train: xyz_max_vh', xyz_max)
    xyz_min =  torch.tensor(xyz_min)
    xyz_max =  torch.tensor(xyz_max)

    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min, xyz_max=xyz_max,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=None,
            vh_bool=vh_bool)


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = vgce.get_rays_of_a_view(
                H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            #cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*near*0.1)]))
        np.savez_compressed(args.export_bbox_and_cams_only,
                            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
                            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(vgce.VoxelGrid, ckpt_path).to(device)
            alpha = model.activate_density(model.density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model = utils.load_model(vgce.VoxelGrid, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
            },
        }

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_train']],
            HW=data_dict['HW'][data_dict['i_train']],
            Ks=data_dict['Ks'][data_dict['i_train']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
            savedir=testsavedir,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_test']],
            HW=data_dict['HW'][data_dict['i_test']],
            Ks=data_dict['Ks'][data_dict['i_test']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
            gt_masks=[data_dict['masks'][i]for i in data_dict['i_test']],
            savedir=testsavedir,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
            render_poses=data_dict['render_poses'],
            HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
            Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
            render_factor=args.render_video_factor,
            savedir=testsavedir,
            **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    print('Done')

    if args.export_pointcloud:
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model = utils.load_model(vgce.VoxelGrid, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
            },
        }

        from VH_fit import vh_fit
        import open3d as o3d
        voxel_shape = model.density.shape
        vh_masks = data_dict["masks"][data_dict['i_train']]
        vh_poses = data_dict["poses"][data_dict['i_train']]
        vh_Ks = data_dict["Ks"][data_dict['i_train']]
        xyz_min = model.xyz_min
        xyz_max = model.xyz_max
        VH_bool = vh_fit(voxel_shape, xyz_max, xyz_min, vh_masks, vh_poses, vh_Ks)


        # 计算所有坐标点
        xyz_min = xyz_min.cpu().numpy()
        xyz_max = xyz_max.cpu().numpy()
        xyz = np.stack(np.meshgrid(
            np.linspace(xyz_min[0], xyz_max[0], voxel_shape[2]),
            np.linspace(xyz_min[1], xyz_max[1], voxel_shape[3]),
            np.linspace(xyz_min[2], xyz_max[2], voxel_shape[4]),
            indexing="ij"), -1)
        #VH_bool = torch.from_numpy(VH_bool)

        points_vh = xyz[VH_bool.nonzero()]
        pcd_vh = o3d.geometry.PointCloud()
        pcd_vh.points = o3d.utility.Vector3dVector(points_vh)
        pcd_vh.paint_uniform_color([0, 0, 1.0])

        with torch.no_grad():
            VH_bool = torch.from_numpy(VH_bool)
            model.density[VH_bool[None, None] == 0] = -100
            # density = model.density.cpu().numpy()
            # density = np.squeeze(density, axis=0)
            # density = np.squeeze(density, axis=0)
            # print(density.shape)
            # # color = model.k0
            #
            # result_bool = (density > 0.)
            # points_result = xyz[result_bool.nonzero()]
            # pcd_result = o3d.geometry.PointCloud()
            # pcd_result.points = o3d.utility.Vector3dVector(points_result)
            # pcd_result.paint_uniform_color([0, 1.0, 0])
            #
            # o3d.visualization.draw_geometries([pcd_result, pcd_vh])

        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_vh_remove_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps = render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_test']],
            HW=data_dict['HW'][data_dict['i_test']],
            Ks=data_dict['Ks'][data_dict['i_test']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
            gt_masks=[data_dict['masks'][i]for i in data_dict['i_test']],
            savedir=testsavedir,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            **render_viewpoints_kwargs)





