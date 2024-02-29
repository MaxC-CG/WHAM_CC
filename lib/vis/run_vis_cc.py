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
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    # Define camera rotations for top, front, and side views
    top_view_rotation = torch.tensor(R.from_euler('x', -90, degrees=True).as_matrix())
    front_view_rotation = torch.eye(3)
    side_view_rotation = torch.tensor(R.from_euler('y', 90, degrees=True).as_matrix())

    writer = imageio.get_writer(
        osp.join(output_pth, 'output.mp4'), 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1,
        codec='libx264', bitrate='16M',
        ffmpeg_params=['-vf', f'scale={width * 3}:{height * 2}']
    )
    bar = Bar('Rendering results ...', fill='#', max=length)

    frame_i = 0
    while (cap.isOpened()):
        flag, org_img = cap.read()
        if not flag: break
        img_original = org_img[..., ::-1].copy()
        img_keypoints = img_original.copy()

        # Draw keypoints on the image
        if 'keypoints' in results and 'contact' in results:
            keypoints = results['keypoints'][frame_i].reshape(-1, 3)
            contact_probs = results['contact'][frame_i]
            max_contact_idx = np.argmax(contact_probs)  # Index of the foot keypoint with the highest contact probability

            for idx, joint in enumerate(keypoints):
                x, y, confidence = joint
                radius = int(max(1, confidence * 5))  # Scale the radius based on confidence
                color = (0, 255, 0) if idx == max_contact_idx else (0, 0, 255)  # Use green for the keypoint with the highest contact probability, red otherwise
                cv2.circle(img_keypoints, (int(x), int(y)), radius, color, -1)  # Draw the circle

        # Render 3D mesh onto the original image
        img_overlaid = img_original.copy()
        if 'verts' in results:
            img_overlaid = renderer.render_mesh(torch.from_numpy(results['verts'][frame_i]).to(cfg.DEVICE), img_overlaid)

        # Render global views
        if vis_global and 'pose_world' in results:
            verts = tt(results['pose_world'][frame_i]['verts'])
            faces = renderer.faces.clone().squeeze(0)
            colors = torch.ones((1, 4)).float().to(cfg.DEVICE); colors[..., :3] *= 0.9

            # Render top view
            renderer.create_camera(top_view_rotation, default_T)
            img_top = np.ones_like(img_original) * 255  # Use a white background
            img_top = renderer.render_mesh(verts, img_top, faces=faces, colors=colors)

            # Render front view
            renderer.create_camera(front_view_rotation, default_T)
            img_front = np.ones_like(img_original) * 255  # Use a white background
            img_front = renderer.render_mesh(verts, img_front, faces=faces, colors=colors)

            # Render side view
            renderer.create_camera(side_view_rotation, default_T)
            img_side = np.ones_like(img_original) * 255  # Use a white background
            img_side = renderer.render_mesh(verts, img_side, faces=faces, colors=colors)
        else:
            img_top = np.ones_like(img_original) * 255
            img_front = np.ones_like(img_original) * 255
            img_side = np.ones_like(img_original) * 255

        # Concatenate all views in a 3x2 layout
        top_row = np.concatenate([img_top, img_front, img_side], axis=1)
        bottom_row = np.concatenate([img_original, img_keypoints, img_overlaid], axis=1)
        final_img = np.concatenate([top_row, bottom_row], axis=0)

        writer.append_data(final_img)
        bar.next()
        frame_i += 1

    writer.close()
