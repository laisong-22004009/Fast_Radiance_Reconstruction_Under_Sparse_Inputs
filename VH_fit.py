import open3d as o3d
import numpy as np

def vh_fit(voxel_shape, xyz_max, xyz_min, input_masks, input_poses, Ks):
    K = Ks[0]
    K[0, 0] = -1 * K[0, 0]
    input_poses = input_poses.cpu().numpy()
    xyz_min = xyz_min.cpu().numpy()
    xyz_max = xyz_max.cpu().numpy()

    xyz = np.stack(np.meshgrid(
        np.linspace(xyz_min[0], xyz_max[0], voxel_shape[2]),
        np.linspace(xyz_min[1], xyz_max[1], voxel_shape[3]),
        np.linspace(xyz_min[2], xyz_max[2], voxel_shape[4]),
        indexing="ij"), -1)
    ones = np.ones(xyz.shape[:-1])[:,:,:,np.newaxis]
    xyz_add = np.concatenate((xyz, ones), axis=3)[...,np.newaxis]
    Nx, Ny, Nz = xyz.shape[:3]
    voxels_count = np.zeros((Nx, Ny, Nz), dtype=np.int32)

    import cv2
    for i in range(input_poses.shape[0]):
        print("第 {} 张图片".format(i))
        # read mask w2c KT
        mask = input_masks[i,...]   #加入图像膨胀

        mask[0, 0] = 0
        cv2.imshow("mask", (mask * 240).astype(np.uint8))
        cv2.waitKey(100)
        kernel = np.ones((10, 10), np.uint8)
        mask= cv2.dilate(mask, kernel, 1)
        print("pengzhang")
        cv2.imshow("mask", (mask * 240).astype(np.uint8))
        cv2.waitKey(100)

        c2w = input_poses[i, :, :]
        w2c = np.linalg.inv(c2w)[0:3, :]
        KT = np.dot(K, w2c)

        KT_add = KT[np.newaxis, np.newaxis, np.newaxis, :, :] # 1 * 1 * 11 * 3  * 4
        points2D = np.matmul(KT_add, xyz_add) # 215 * 126 * 152 * 3 * 1
        Z = np.concatenate((points2D[:,:,:,2,:], points2D[:,:,:,2,:], points2D[:,:,:,2,:]), axis=3)[...,np.newaxis] # 215 * 126 * 152 * 3 * 1
        points2D = points2D / Z  # 215 * 126 * 152 * 3 * 1
        points2D = points2D.squeeze(axis=-1) # 215 * 126 * 152 * 3

        # remove out image
        points2D_uv = points2D[..., :2] # 215 * 126 * 152 * 2
        points2D_uv_in = (points2D_uv < (300-1)) & (points2D_uv > 0)
        points2D_uv_in = points2D_uv_in[..., 0] & points2D_uv_in[..., 1]
        points2D_uv_in = np.stack((points2D_uv_in, points2D_uv_in), axis=-1)
        points2D_uv_remain = points2D_uv * points2D_uv_in # 215 * 126 * 152 * 2

        # int index
        points2D_uv_remain_int = np.floor(points2D_uv_remain).astype(np.int32)

        # index mask
        fb_val_point3D = mask[points2D_uv_remain_int[...,1], points2D_uv_remain_int[...,0]] # 215 * 126 * 152
        remain_point3D_index = (fb_val_point3D > 0)


        # visual num count
        voxels_count = voxels_count + remain_point3D_index

    cv2.destroyAllWindows()


    VH_bool = (voxels_count == voxels_count.max())

    print(VH_bool.shape)
    points_vh = xyz[VH_bool.nonzero()]
    pcd_vh = o3d.geometry.PointCloud()
    pcd_vh.points = o3d.utility.Vector3dVector(points_vh)


    voxel_grid_vh = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_vh, voxel_size=0.011)
    o3d.visualization.draw_geometries([voxel_grid_vh])


    return VH_bool