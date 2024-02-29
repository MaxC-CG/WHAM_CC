import os
import os.path as osp
import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar
from scipy.spatial.transform import Rotation as R
from lib.vis.renderer import Renderer, get_global_cameras

def run_vis_on_demo(cfg, video, results, output_pth, smpl, vis_global=True):
    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)
    
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # create renderer with cliff focal length estimation
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)
    
    if vis_global:
        # setup global coordinate subject
        # current implementation only visualize the subject appeared longest
        n_frames = {k: len(results[k]['frame_ids']) for k in results.keys()}
        sid = max(n_frames, key=n_frames.get)
        global_output = smpl.get_output(
            body_pose=tt(results[sid]['pose_world'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world'][:, :3]),
            betas=tt(results[sid]['betas']),
            transl=tt(results[sid]['trans_world']))
        verts_glob = global_output.vertices.cpu()
        verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
        cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
        scale = max(sx.item(), sz.item()) * 1.5
        
        # set default ground
        renderer.set_ground(scale, cx.item(), cz.item())
        
        # build global camera
        global_R, global_T, global_lights = get_global_cameras(verts_glob, cfg.DEVICE)

        # Define camera rotations for top, front, and side views
        top_view_rotation = torch.tensor(R.from_euler('x', -90, degrees=True).as_matrix())
        front_view_rotation = torch.eye(3)
        side_view_rotation = torch.tensor(R.from_euler('y', 90, degrees=True).as_matrix())

        # Define custom translation matrices for top, front, and side views
        translation_top = torch.tensor([0, 0, 4])  # Move camera up along z-axis
        translation_front = torch.tensor([0, 4, 0])  # Move camera back along y-axis
        translation_side = torch.tensor([-4, 0, 0])  # Move camera right along x-axis
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    writer = imageio.get_writer(
        osp.join(output_pth, 'output.mp4'), 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1,
        codec='libx264', bitrate='16M',
        ffmpeg_params=['-vf', f'scale={width * 3}:{height * 2}']
    )
    bar = Bar('Rendering results ...', fill='#', max=length)

    frame_i = 0
    _global_R, _global_T = None, None
    # run rendering
    while (cap.isOpened()):
        flag, org_img = cap.read()
        if not flag: break
        img_original = org_img[..., ::-1].copy()
        img_keypoints = img_original.copy()

        # Render 3D mesh onto the original image
        img_overlaid = img_original.copy()
        renderer.create_camera(default_R, default_T)
        for _id, val in results.items():
            # Render onto the image
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            img_overlaid = renderer.render_mesh(torch.from_numpy(val['verts'][frame_i2]).to(cfg.DEVICE), img_overlaid)

            # Draw keypoints on the image
            if 'keypoints' in val and 'contact' in val:
                keypoints = val['keypoints'][frame_i2].reshape(-1, 3)
                contact_probs = val['contact'][frame_i2]
                max_contact_idx = np.argmax(contact_probs)  # Index of the foot keypoint with the highest contact probability

                for idx, joint in enumerate(keypoints):
                    x, y, confidence = joint
                    radius = int(max(1, confidence * 20))  # Scale the radius based on confidence
                    color = (0, 255, 0) if idx in [22, 23, 24, 25] and idx == max_contact_idx + 22 else (255, 0, 0)  # Use green for the keypoint with the highest contact probability, red otherwise
                    cv2.circle(img_keypoints, (int(x), int(y)), radius, color, -1)  # Draw the circle

        # Render global views
        if vis_global:
            # Render the global coordinate
            if frame_i in results[sid]['frame_ids']:
                frame_i3 = np.where(results[sid]['frame_ids'] == frame_i)[0]
                verts = verts_glob[[frame_i3]].to(cfg.DEVICE)
                faces = renderer.faces.clone().squeeze(0)
                colors = torch.ones((1, 4)).float().to(cfg.DEVICE); colors[..., :3] *= 0.9

                if _global_R is None:
                    _global_R = global_R[frame_i3].clone(); _global_T = global_T[frame_i3].clone()
                cameras = renderer.create_camera(global_R[frame_i3], global_T[frame_i3])
                img_glob = renderer.render_with_ground(verts, faces, colors, cameras, global_lights)
                
                # Render top, front, and side views using render_with_ground
                camera_top = renderer.create_camera(top_view_rotation, translation_top)
                img_top = renderer.render_with_ground(verts, faces, colors, camera_top, global_lights)

                camera_front = renderer.create_camera(front_view_rotation, translation_front)
                img_front = renderer.render_with_ground(verts, faces, colors, camera_front, global_lights)

                camera_side = renderer.create_camera(side_view_rotation, translation_side)
                img_side = renderer.render_with_ground(verts, faces, colors, camera_side, global_lights)
            else:
                img_top = np.ones_like(img_original) * 255
                img_front = np.ones_like(img_original) * 255
                img_side = np.ones_like(img_original) * 255
                img_glob = np.ones_like(img_original) * 255

            # Concatenate all views in a 3x2 layout
            top_row = np.concatenate([img_top, img_front, img_side], axis=1)
            bottom_row = np.concatenate([img_original, img_keypoints, img_overlaid], axis=1)
            final_img = np.concatenate([top_row, bottom_row], axis=0)
        else:
            final_img = img_original  # If not vis_global, just use the original image

        writer.append_data(final_img)
        bar.next()
        frame_i += 1

    writer.close()
