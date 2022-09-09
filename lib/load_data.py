import os.path

import imageio
import numpy as np

from .load_blender import load_blender_data



def load_data(args):

    K, depths = None, None

    # init masks
    masks = None

    if args.dataset_type == 'llff':
        images, depths, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir, args.factor,
            recenter=True, bd_factor=.75,
            spherify=args.spherify,
            load_depths=args.load_depths)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        if args.view_select == 1:
            i_train = np.array([67, 25, 90, 13])
        if args.view_select == 2:
            i_train = np.array([0, 25, 90, 13])
        if args.view_select == 3:
            i_train = np.array([0, 25, 80, 20])
        if args.view_select == 4:
            i_train = np.array([0, 25, 85, 28])
        if args.view_select == 5:
            i_train = np.array([6, 10, 31, 98])
        if args.input_num == 4:
            if args.view_select == 'chair':##
                i_train = np.array([0, 25, 90, 13])
            if args.view_select == 'ship':##
                i_train = np.array([10, 12, 65, 93])
            if args.view_select == 'drums': ##
                i_train = np.array([67, 25, 90, 13])
            if args.view_select == 'ficus': ##
                i_train = np.array([0, 25, 90, 13])
            if args.view_select == 'hotdog':
                i_train = np.array([0, 25, 80, 20])
            if args.view_select == 'lego':
                i_train = np.array([67, 25, 90, 13])
            if args.view_select == 'mic':
                i_train = np.array([0, 1, 2, 3])
            if args.view_select == 'materials':
                i_train = np.array([47, 95, 28, 15])


        if  args.input_num == 8:
            i_train = np.array([200, 213, 226, 239, 252, 265, 278, 291])

        # np.random.seed(10)
        # i_train = i_train[np.random.choice([i for i in range(0, 100)], 4, replace=False)]
        # print("训练输入数据", i_train)

        near, far = 2., 6.

        masks = images[..., -1]


        os.makedirs(os.path.join(args.basedir, args.expname, "trainset"), exist_ok=True)
        for i in range(args.input_num):
            img_idx = i_train[i]
            train_image = images[img_idx]
            out_path = os.path.join(args.basedir, args.expname, "trainset", "input_{:d}.png".format(img_idx))
            imageio.imwrite(out_path, train_image)

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

        # import cv2
        # for i in range(i_train.shape[0]):
        #     img = (images[i_train[i]] * 255)
        #     cv2.imwrite("./{:03d}.png".format(i), img[...,::-1].astype(np.uint8))

    elif args.dataset_type == 'blendedmvs':
        images, poses, render_poses, hwf, K, i_split = load_blendedmvs_data(args.datadir)
        print('Loaded blendedmvs', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        assert images.shape[-1] == 3

    elif args.dataset_type == 'tankstemple':
        images, poses, render_poses, hwf, K, i_split = load_tankstemple_data(args.datadir)
        print('Loaded tankstemple', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'nsvf':
        images, poses, render_poses, hwf, i_split = load_nsvf_data(args.datadir)
        print('Loaded nsvf', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'deepvoxels':
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.scene, basedir=args.datadir, testskip=args.testskip)
        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R - 1
        far = hemi_R + 1
        assert args.white_bkgd
        assert images.shape[-1] == 3

    elif args.dataset_type == 'co3d':
        # each image can be in different shapes and intrinsics
        images, masks, poses, render_poses, hwf, K, i_split = load_co3d_data(args)
        print('Loaded co3d', args.datadir, args.annot_path, args.sequence_name)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)

        for i in range(len(images)):
            if args.white_bkgd:
                images[i] = images[i] * masks[i][...,None] + (1.-masks[i][...,None])
            else:
                images[i] = images[i] * masks[i][...,None]


    #######################添加DTU数据集的加载#############################
    elif args.dataset_type == 'DTU':
        images, poses, render_poses, hwf, masks, K = load_DTU(args)
        if args.white_bkgd:
            images = images[...] * masks[...] + (1. - masks)
        else:
            images = images[...] * masks[...]
        masks =masks.squeeze(-1)

        i_test = np.delete(np.arange(0, 49), [25, 22, 28, 40, 44, 48, 0, 8, 13, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39])

        if args.input_num == 3:
            i_train = np.array([22, 25, 28])
            i_val = np.array([16, 22, 47])
            print("3 inputs")
        elif args.input_num == 6:
            i_train = np.array([22, 25, 28, 40, 44, 48])
            i_val = np.array([16, 22, 47])
            print("6 inputs")
        elif args.input_num == 9:
            i_train = np.array([22, 25, 28, 40, 44, 48, 0, 8, 13])
            i_val = np.array([16, 22, 47])
            print("9 inputs")
        else:
            # all
            i_train = np.arange(0, 49)
            i_val = np.array([16, 22, 47])
            print("all inputs")

        near = 0.1
        far = 5.0

        os.makedirs(os.path.join(args.basedir, args.expname, "trainset"), exist_ok=True)
        for i in range(args.input_num):
            img_idx = i_train[i]
            train_image = images[img_idx]
            out_path = os.path.join(args.basedir, args.expname, "trainset", "input_{:d}.png".format(img_idx))
            imageio.imwrite(out_path, train_image)

    elif args.dataset_type == 'DTU_test':
        images, poses, render_poses, hwf, masks, K = load_DTU_test(args)
        if args.white_bkgd:
            images = images[...] * masks[...] + (1. - masks)
        else:
            images = images[...] * masks[...]
        masks =masks.squeeze(-1)

        if args.input_num == 3:
            i_train = np.array([22, 25, 28])
            i_val = np.array([16, 22, 47])
            i_test = np.delete(np.arange(0, 49), [22, 25, 28])
            print("3 inputs")
        elif args.input_num == 6:
            i_train = np.array([22, 25, 28, 40, 44, 48])
            i_val = np.array([16, 22, 47])
            i_test = np.delete(np.arange(0, 49), [22, 25, 28, 40, 44, 48])
            print("6 inputs")
        elif args.input_num == 9:
            i_train = np.array([22, 25, 28, 40, 44, 48, 0, 8, 13])
            i_val = np.array([16, 22, 47])

            print("9 inputs")
        else:
            # all
            i_train = np.arange(0, 49)
            i_val = np.array([16, 22, 47])
            i_test = np.array([16, 22, 47])
            print("all inputs")

        near = 0.1
        far = 5.0

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    # 把masks也加到data-dict
    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
        masks=masks
    )

    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far

