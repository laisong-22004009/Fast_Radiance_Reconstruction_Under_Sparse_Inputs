import os
import torch
import glob
import imageio
import numpy as np
import cv2


def load_DTU(args):
    root_dir = args.datadir
    print("Loading DTU from: ", root_dir)
    rgb_paths = [
        x
        for x in glob.glob(os.path.join(root_dir, "image", "*"))
        if (x.endswith(".jpg") or x.endswith(".png"))
    ]
    rgb_paths = sorted(rgb_paths)
    mask_paths = sorted(glob.glob(os.path.join(root_dir, "masks", "*.png")))

    sel_indices = np.arange(len(rgb_paths))

    cam_path = os.path.join(root_dir, "cameras.npz")
    all_cam = np.load(cam_path)

    all_imgs = []
    all_poses = []
    all_masks = []

    # Prepare to average intrinsics over images
    fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0

    _coord_trans_world = torch.tensor(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )
    _coord_trans_cam = torch.tensor(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )

    for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
        i = sel_indices[idx]
        img = imageio.imread(rgb_path)
        # import matplotlib.pyplot as plt
        # plt.imshow(img.astype(np.uint8))
        # plt.show()
        all_imgs.append(img)

        x_scale = y_scale = 1.0
        xy_delta = 0.0

        mask = imageio.imread(mask_path)
        mask = mask[..., :1] / 255.
        all_masks.append(mask)

        # Decompose projection matrix
        # DVR uses slightly different format for DTU set
        P = all_cam["world_mat_" + str(i)]
        P = P[:3]
        K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
        K = K / K[2, 2]

        # 计算pose
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        scale_mtx = all_cam.get("scale_mat_" + str(i))

        norm_trans = scale_mtx[:3, 3:]
        norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

        pose[:3, 3:] -= norm_trans
        pose[:3, 3:] /= norm_scale

        pose = (_coord_trans_world@ torch.tensor(pose, dtype=torch.float32) @ _coord_trans_cam).cpu().numpy().astype(np.float32)

        all_poses.append(pose)

        fx += torch.tensor(K[0, 0]) * x_scale
        fy += torch.tensor(K[1, 1]) * y_scale
        cx += (torch.tensor(K[0, 2]) + xy_delta) * x_scale
        cy += (torch.tensor(K[1, 2]) + xy_delta) * y_scale

    imgs = (np.array(all_imgs) / 255.).astype(np.float32)
    poses = np.array(all_poses).astype(np.float32)
    masks = np.array(all_masks)

    H, W = imgs[0].shape[:2]
    fx /= len(rgb_paths)
    fy /= len(rgb_paths)
    cx /= len(rgb_paths)
    cy /= len(rgb_paths)
    focal = float(((fx + fy) / 2))

    render_poses = poses

    return imgs, poses, render_poses, [H, W, focal], masks, K

if __name__ == "__main__":
    images, poses, render_poses, hwf, masks, Ks = load_DTU()
    images = images[...] * masks[...] + (1. - masks)
    image = images[0, ...]
    print(image)
    import matplotlib.pyplot as plt
    plt.imshow((image * 255).astype(np.uint8))
    plt.show()