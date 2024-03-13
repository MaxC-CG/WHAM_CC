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
            body_pose=tt(results[sid]['pose_world_after'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world_after'][:, :3]),
            betas=tt(results[sid]['betas_after']),
            transl=tt(results[sid]['trans_world_after']))
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
        top_view_rotation = torch.tensor(R.from_euler('xyz', [-90, 0, 180], degrees=True).as_matrix())
        front_view_rotation = torch.tensor(R.from_euler('y', 180, degrees=True).as_matrix())  # Rotate 180 degrees around y-axis to look from behind
        side_view_rotation = torch.tensor(R.from_euler('y', -90, degrees=True).as_matrix())  

        # Define custom translation matrices for top, front, and side views
        # test0
        translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test1
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test2
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test3
        # translation_top = torch.tensor([0.5, 0, 6])  # Move camera up along z-axis
        # translation_front = torch.tensor([0.5, -1, 6])  # Move camera back along y-axis
        # translation_side = torch.tensor([0.5, -1, 5])  # Move camera right along x-axis
        # test4
        # translation_top = torch.tensor([-1.5, 0, 6])  # Move camera up along z-axis
        # translation_front = torch.tensor([-1.5, -1, 6])  # Move camera back along y-axis
        # translation_side = torch.tensor([-0.5, -1, 7])  # Move camera right along x-axis
        # test5
        # translation_top = torch.tensor([-0.5, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([-0.5, -1, 4])  # Move camera back along y-axis
        # translation_side = torch.tensor([-0.5, -1, 5])  # Move camera right along x-axis
        # test6
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test7
        # translation_top = torch.tensor([0, 1, 7])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 7])  # Move camera back along y-axis
        # translation_side = torch.tensor([1, -1, 8])  # Move camera right along x-axis
   
    
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
            img_overlaid = renderer.render_mesh(torch.from_numpy(val['verts_after'][frame_i2]).to(cfg.DEVICE), img_overlaid)

            # Draw keypoints on the image
            if 'keypoints' in val and 'contact' in val:
                keypoints = val['keypoints'][frame_i2].reshape(-1, 3)
                contact_probs = val['contact'][frame_i2]
                max_contact_idx = np.argmax(contact_probs)  # Index of the foot keypoint with the highest contact probability
                left_foot_prob = contact_probs[0] + contact_probs[1]  # Sum of the first two values for left foot probability
                right_foot_prob = contact_probs[2] + contact_probs[3]  # Sum of the last two values for right foot probability

                for idx, joint in enumerate(keypoints):
                    x, y, confidence = joint
                    radius = int(max(1, confidence * 20))  # Scale the radius based on confidence
                    # color = (0, 255, 0) if idx in [20, 21, 22, 23] and idx == max_contact_idx + 20 else (255, 0, 0)  # Use green for the keypoint with the highest contact probability, red otherwise
                    # cv2.circle(img_keypoints, (int(x), int(y)), radius, color, -1)  # Draw the circle
                    color = (0, 255, 0) if idx == 15 and left_foot_prob > right_foot_prob else (0, 255, 0) if idx == 16 and right_foot_prob > left_foot_prob else (255, 0, 0)
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
            top_row = np.concatenate([img_original, img_keypoints, img_top], axis=1)
            bottom_row = np.concatenate([img_overlaid, img_side, img_front], axis=1)
            final_img = np.concatenate([top_row, bottom_row], axis=0)
        else:
            final_img = img_original  # If not vis_global, just use the original image

        writer.append_data(final_img)
        bar.next()
        frame_i += 1

    writer.close()

def run_vis_on_demo_smplify(cfg, video, results, output_pth, smpl_before, smpl, vis_global=True):
    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)
    
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # create renderer with cliff focal length estimation
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer_before = Renderer(width, height, focal_length, cfg.DEVICE, smpl_before.faces)
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)
    
    if vis_global:
        # setup global coordinate subject
        # current implementation only visualize the subject appeared longest
        n_frames = {k: len(results[k]['frame_ids']) for k in results.keys()}
        sid = max(n_frames, key=n_frames.get)

        global_output_before = smpl_before.get_output(
            body_pose=tt(results[sid]['pose_world_before'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world_before'][:, :3]),
            betas=tt(results[sid]['betas_before']),
            transl=tt(results[sid]['trans_world_before']))
        verts_glob_before = global_output_before.vertices.cpu()
        verts_glob_before[..., 1] = verts_glob_before[..., 1] - verts_glob_before[..., 1].min()
        cx_before, cz_before = (verts_glob_before.mean(1).max(0)[0] + verts_glob_before.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx_before, sz_before = (verts_glob_before.mean(1).max(0)[0] - verts_glob_before.mean(1).min(0)[0])[[0, 2]]
        scale_before = max(sx_before.item(), sz_before.item()) * 1.5
        
        global_output = smpl.get_output(
            body_pose=tt(results[sid]['pose_world_after'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world_after'][:, :3]),
            betas=tt(results[sid]['betas_after']),
            transl=tt(results[sid]['trans_world_after']))
        verts_glob = global_output.vertices.cpu()
        verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
        cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
        scale = max(sx.item(), sz.item()) * 1.5
        
        # set default ground
        renderer_before.set_ground(scale_before, cx_before.item(), cz_before.item())
        renderer.set_ground(scale, cx.item(), cz.item())
        
        # build global camera
        global_R, global_T, global_lights = get_global_cameras(verts_glob, cfg.DEVICE)

        # Define camera rotations for top, front, and side views
        top_view_rotation = torch.tensor(R.from_euler('xyz', [-90, 0, 180], degrees=True).as_matrix())
        front_view_rotation = torch.tensor(R.from_euler('y', 180, degrees=True).as_matrix())  # Rotate 180 degrees around y-axis to look from behind
        side_view_rotation = torch.tensor(R.from_euler('y', -90, degrees=True).as_matrix())  

        # Define custom translation matrices for top, front, and side views
        # test0
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test1
        translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test2
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test3
        # translation_top = torch.tensor([0.5, 0, 6])  # Move camera up along z-axis
        # translation_front = torch.tensor([0.5, -1, 6])  # Move camera back along y-axis
        # translation_side = torch.tensor([0.5, -1, 5])  # Move camera right along x-axis
        # test4
        # translation_top = torch.tensor([-1.5, 0, 6])  # Move camera up along z-axis
        # translation_front = torch.tensor([-1.5, -1, 6])  # Move camera back along y-axis
        # translation_side = torch.tensor([-0.5, -1, 7])  # Move camera right along x-axis
        # test5
        # translation_top = torch.tensor([-0.5, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([-0.5, -1, 4])  # Move camera back along y-axis
        # translation_side = torch.tensor([-0.5, -1, 5])  # Move camera right along x-axis
        # test6
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test7
        # translation_top = torch.tensor([0, 1, 7])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 7])  # Move camera back along y-axis
        # translation_side = torch.tensor([1, -1, 8])  # Move camera right along x-axis
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    writer = imageio.get_writer(
        osp.join(output_pth, 'output.mp4'), 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1,
        codec='libx264', bitrate='16M',
        ffmpeg_params=['-vf', f'scale={width * 5}:{height * 2}']
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
        img_overlaid_before = img_original.copy()
        renderer_before.create_camera(default_R, default_T)
        
        img_overlaid = img_original.copy()
        renderer.create_camera(default_R, default_T)
        for _id, val in results.items():
            # Render onto the image
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            img_overlaid_before = renderer_before.render_mesh(torch.from_numpy(val['verts_before'][frame_i2]).to(cfg.DEVICE), img_overlaid_before)
            img_overlaid = renderer.render_mesh(torch.from_numpy(val['verts_after'][frame_i2]).to(cfg.DEVICE), img_overlaid)

            # Draw keypoints on the image
            if 'keypoints' in val and 'contact' in val:
                keypoints = val['keypoints'][frame_i2].reshape(-1, 3)
                contact_probs = val['contact'][frame_i2]
                max_contact_idx = np.argmax(contact_probs)  # Index of the foot keypoint with the highest contact probability
                left_foot_prob = contact_probs[0] + contact_probs[1]  # Sum of the first two values for left foot probability
                right_foot_prob = contact_probs[2] + contact_probs[3]  # Sum of the last two values for right foot probability

                for idx, joint in enumerate(keypoints):
                    x, y, confidence = joint
                    radius = int(max(1, confidence * 20))  # Scale the radius based on confidence
                    # color = (0, 255, 0) if idx in [20, 21, 22, 23] and idx == max_contact_idx + 20 else (255, 0, 0)  # Use green for the keypoint with the highest contact probability, red otherwise
                    # cv2.circle(img_keypoints, (int(x), int(y)), radius, color, -1)  # Draw the circle
                    color = (0, 255, 0) if idx == 15 and left_foot_prob > right_foot_prob else (0, 255, 0) if idx == 16 and right_foot_prob > left_foot_prob else (255, 0, 0)
                    cv2.circle(img_keypoints, (int(x), int(y)), radius, color, -1)  # Draw the circle
            # if 'full_joints_wham' in val:
            #     keypoints = val['full_joints_wham'].reshape(-1, 3)

            #     for idx, joint in enumerate(keypoints):
            #         x, y, confidence = joint
            #         radius = int(max(1, confidence * 20))  # Scale the radius based on confidence
            #         color = (0, 0, 255)
            #         cv2.circle(img_keypoints, (int(x), int(y)), radius, color, -1)  # Draw the circle



        # Render global views
        if vis_global:
            # Render the global coordinate
            if frame_i in results[sid]['frame_ids']:
                frame_i3 = np.where(results[sid]['frame_ids'] == frame_i)[0]
                verts_before = verts_glob_before[[frame_i3]].to(cfg.DEVICE)
                faces_before = renderer_before.faces.clone().squeeze(0)
                colors_before = torch.ones((1, 4)).float().to(cfg.DEVICE); colors_before[..., :3] *= 0.9
                
                verts = verts_glob[[frame_i3]].to(cfg.DEVICE)
                faces = renderer.faces.clone().squeeze(0)
                colors = torch.ones((1, 4)).float().to(cfg.DEVICE); colors[..., :3] *= 0.9

                if _global_R is None:
                    _global_R = global_R[frame_i3].clone(); _global_T = global_T[frame_i3].clone()

                cameras_before = renderer_before.create_camera(global_R[frame_i3], global_T[frame_i3])
                cameras = renderer.create_camera(global_R[frame_i3], global_T[frame_i3])
                img_glob_before = renderer_before.render_with_ground(verts_before, faces_before, colors_before, cameras_before, global_lights)
                img_glob = renderer.render_with_ground(verts, faces, colors, cameras, global_lights)
                
                # Render top, front, and side views using render_with_ground
                camera_top_before = renderer_before.create_camera(top_view_rotation, translation_top)
                camera_top = renderer.create_camera(top_view_rotation, translation_top)
                img_top_before = renderer_before.render_with_ground(verts_before, faces_before, colors_before, camera_top_before, global_lights)
                img_top = renderer.render_with_ground(verts, faces, colors, camera_top, global_lights)

                camera_front_before = renderer_before.create_camera(front_view_rotation, translation_front)
                camera_front = renderer.create_camera(front_view_rotation, translation_front)
                img_front_before = renderer_before.render_with_ground(verts_before, faces_before, colors_before, camera_front_before, global_lights)
                img_front = renderer.render_with_ground(verts, faces, colors, camera_front, global_lights)

                camera_side_before = renderer_before.create_camera(side_view_rotation, translation_side)
                camera_side = renderer.create_camera(side_view_rotation, translation_side)
                img_side_before = renderer_before.render_with_ground(verts_before, faces_before, colors_before, camera_side_before, global_lights)
                img_side = renderer.render_with_ground(verts, faces, colors, camera_side, global_lights)
            else:
                img_top_before = np.ones_like(img_original) * 255
                img_top = np.ones_like(img_original) * 255
                img_front_before = np.ones_like(img_original) * 255
                img_front = np.ones_like(img_original) * 255
                img_side_before = np.ones_like(img_original) * 255
                img_side = np.ones_like(img_original) * 255
                img_glob_before = np.ones_like(img_original) * 255
                img_glob = np.ones_like(img_original) * 255

            # Concatenate all views in a 3x2 layout
            top_row = np.concatenate([img_original, img_overlaid_before, img_overlaid, img_top_before, img_top], axis=1)
            bottom_row = np.concatenate([img_keypoints, img_side_before, img_side, img_front_before, img_front], axis=1)
            final_img = np.concatenate([top_row, bottom_row], axis=0)
        else:
            final_img = img_original  # If not vis_global, just use the original image

        writer.append_data(final_img)
        bar.next()
        frame_i += 1

    writer.close()

def run_vis_on_demo_smplify_rtm(cfg, video, results, output_pth, smpl_before, smpl_rtm, smpl, vis_global=True):
    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)
    
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # create renderer with cliff focal length estimation
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer_before = Renderer(width, height, focal_length, cfg.DEVICE, smpl_before.faces)
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)
    renderer_rtm = Renderer(width, height, focal_length, cfg.DEVICE, smpl_rtm.faces)
    
    if vis_global:
        # setup global coordinate subject
        # current implementation only visualize the subject appeared longest
        n_frames = {k: len(results[k]['frame_ids']) for k in results.keys()}
        sid = max(n_frames, key=n_frames.get)

        global_output_before = smpl_before.get_output(
            body_pose=tt(results[sid]['pose_world_before'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world_before'][:, :3]),
            betas=tt(results[sid]['betas_before']),
            transl=tt(results[sid]['trans_world_before']))
        verts_glob_before = global_output_before.vertices.cpu()
        verts_glob_before[..., 1] = verts_glob_before[..., 1] - verts_glob_before[..., 1].min()
        cx_before, cz_before = (verts_glob_before.mean(1).max(0)[0] + verts_glob_before.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx_before, sz_before = (verts_glob_before.mean(1).max(0)[0] - verts_glob_before.mean(1).min(0)[0])[[0, 2]]
        scale_before = max(sx_before.item(), sz_before.item()) * 1.5
        
        global_output = smpl.get_output(
            body_pose=tt(results[sid]['pose_world_after'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world_after'][:, :3]),
            betas=tt(results[sid]['betas_after']),
            transl=tt(results[sid]['trans_world_after']))
        verts_glob = global_output.vertices.cpu()
        verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
        cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
        scale = max(sx.item(), sz.item()) * 1.5

        global_output_rtm = smpl_rtm.get_output(
            body_pose=tt(results[sid]['pose_world_rtm'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world_rtm'][:, :3]),
            betas=tt(results[sid]['betas_rtm']),
            transl=tt(results[sid]['trans_world_rtm']))
        verts_glob_rtm = global_output_rtm.vertices.cpu()
        verts_glob_rtm[..., 1] = verts_glob_rtm[..., 1] - verts_glob_rtm[..., 1].min()
        cx_rtm, cz_rtm = (verts_glob_rtm.mean(1).max(0)[0] + verts_glob_rtm.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx_rtm, sz_rtm = (verts_glob_rtm.mean(1).max(0)[0] - verts_glob_rtm.mean(1).min(0)[0])[[0, 2]]
        scale_rtm = max(sx_rtm.item(), sz_rtm.item()) * 1.5
        
        # set default ground
        renderer_before.set_ground(scale_before, cx_before.item(), cz_before.item())
        renderer.set_ground(scale, cx.item(), cz.item())
        renderer_rtm.set_ground(scale_rtm, cx_rtm.item(), cz_rtm.item())
        
        # build global camera
        global_R, global_T, global_lights = get_global_cameras(verts_glob, cfg.DEVICE)

        # Define camera rotations for top, front, and side views
        top_view_rotation = torch.tensor(R.from_euler('xyz', [-90, 0, 180], degrees=True).as_matrix())
        front_view_rotation = torch.tensor(R.from_euler('y', 180, degrees=True).as_matrix())  # Rotate 180 degrees around y-axis to look from behind
        side_view_rotation = torch.tensor(R.from_euler('y', -90, degrees=True).as_matrix())  

        # Define custom translation matrices for top, front, and side views
        # test0
        translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test1
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test2
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test3
        # translation_top = torch.tensor([0.5, 0, 6])  # Move camera up along z-axis
        # translation_front = torch.tensor([0.5, -1, 6])  # Move camera back along y-axis
        # translation_side = torch.tensor([0.5, -1, 5])  # Move camera right along x-axis
        # test4
        # translation_top = torch.tensor([-1.5, 0, 6])  # Move camera up along z-axis
        # translation_front = torch.tensor([-1.5, -1, 6])  # Move camera back along y-axis
        # translation_side = torch.tensor([-0.5, -1, 7])  # Move camera right along x-axis
        # test5
        # translation_top = torch.tensor([-0.5, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([-0.5, -1, 4])  # Move camera back along y-axis
        # translation_side = torch.tensor([-0.5, -1, 5])  # Move camera right along x-axis
        # test6
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test7
        # translation_top = torch.tensor([0, 1, 7])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 7])  # Move camera back along y-axis
        # translation_side = torch.tensor([1, -1, 8])  # Move camera right along x-axis
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    writer = imageio.get_writer(
        osp.join(output_pth, 'output.mp4'), 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1,
        codec='libx264', bitrate='16M',
        ffmpeg_params=['-vf', f'scale={width * 7}:{height * 2}']
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
        img_overlaid_before = img_original.copy()
        renderer_before.create_camera(default_R, default_T)
        
        img_overlaid = img_original.copy()
        renderer.create_camera(default_R, default_T)

        img_overlaid_rtm = img_original.copy()
        renderer_rtm.create_camera(default_R, default_T)
        for _id, val in results.items():
            # Render onto the image
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            img_overlaid_before = renderer_before.render_mesh(torch.from_numpy(val['verts_before'][frame_i2]).to(cfg.DEVICE), img_overlaid_before)
            img_overlaid = renderer.render_mesh(torch.from_numpy(val['verts_after'][frame_i2]).to(cfg.DEVICE), img_overlaid)
            img_overlaid_rtm = renderer_rtm.render_mesh(torch.from_numpy(val['verts_rtm'][frame_i2]).to(cfg.DEVICE), img_overlaid_rtm)

            # Draw keypoints on the image
            if 'keypoints' in val and 'contact' in val:
                keypoints = val['keypoints'][frame_i2].reshape(-1, 3)
                contact_probs = val['contact'][frame_i2]
                max_contact_idx = np.argmax(contact_probs)  # Index of the foot keypoint with the highest contact probability
                left_foot_prob = contact_probs[0] + contact_probs[1]  # Sum of the first two values for left foot probability
                right_foot_prob = contact_probs[2] + contact_probs[3]  # Sum of the last two values for right foot probability

                for idx, joint in enumerate(keypoints):
                    x, y, confidence = joint
                    radius = int(max(1, confidence * 20))  # Scale the radius based on confidence
                    # color = (0, 255, 0) if idx in [20, 21, 22, 23] and idx == max_contact_idx + 20 else (255, 0, 0)  # Use green for the keypoint with the highest contact probability, red otherwise
                    # cv2.circle(img_keypoints, (int(x), int(y)), radius, color, -1)  # Draw the circle
                    color = (0, 255, 0) if idx == 15 and left_foot_prob > right_foot_prob else (0, 255, 0) if idx == 16 and right_foot_prob > left_foot_prob else (255, 0, 0)
                    cv2.circle(img_keypoints, (int(x), int(y)), radius, color, -1)  # Draw the circle

            if 'keypoints_rtm' in val:
                keypoints = val['keypoints_rtm'][frame_i2].reshape(-1, 3)

                for idx, joint in enumerate(keypoints):
                    x, y, confidence = joint
                    radius = int(max(1, confidence * 20))  # Scale the radius based on confidence
                    color = (0, 0, 255)
                    cv2.circle(img_keypoints, (int(x), int(y)), radius, color, -1)  # Draw the circle

        # Render global views
        if vis_global:
            # Render the global coordinate
            if frame_i in results[sid]['frame_ids']:
                frame_i3 = np.where(results[sid]['frame_ids'] == frame_i)[0]
                verts_before = verts_glob_before[[frame_i3]].to(cfg.DEVICE)
                faces_before = renderer_before.faces.clone().squeeze(0)
                colors_before = torch.ones((1, 4)).float().to(cfg.DEVICE); colors_before[..., :3] *= 0.9
                
                verts = verts_glob[[frame_i3]].to(cfg.DEVICE)
                faces = renderer.faces.clone().squeeze(0)
                colors = torch.ones((1, 4)).float().to(cfg.DEVICE); colors[..., :3] *= 0.9

                verts_rtm = verts_glob_rtm[[frame_i3]].to(cfg.DEVICE)
                faces_rtm = renderer_rtm.faces.clone().squeeze(0)
                colors_rtm = torch.ones((1, 4)).float().to(cfg.DEVICE); colors_rtm[..., :3] *= 0.9

                if _global_R is None:
                    _global_R = global_R[frame_i3].clone(); _global_T = global_T[frame_i3].clone()

                cameras_before = renderer_before.create_camera(global_R[frame_i3], global_T[frame_i3])
                cameras = renderer.create_camera(global_R[frame_i3], global_T[frame_i3])
                cameras_rtm = renderer_rtm.create_camera(global_R[frame_i3], global_T[frame_i3])
                img_glob_before = renderer_before.render_with_ground(verts_before, faces_before, colors_before, cameras_before, global_lights)
                img_glob = renderer.render_with_ground(verts, faces, colors, cameras, global_lights)
                img_glob_rtm = renderer_rtm.render_with_ground(verts_rtm, faces_rtm, colors_rtm, cameras_rtm, global_lights)
                
                # Render top, front, and side views using render_with_ground
                camera_top_before = renderer_before.create_camera(top_view_rotation, translation_top)
                camera_top = renderer.create_camera(top_view_rotation, translation_top)
                camera_top_rtm = renderer_rtm.create_camera(top_view_rotation, translation_top)
                img_top_before = renderer_before.render_with_ground(verts_before, faces_before, colors_before, camera_top_before, global_lights)
                img_top = renderer.render_with_ground(verts, faces, colors, camera_top, global_lights)
                img_top_rtm = renderer_rtm.render_with_ground(verts_rtm, faces_rtm, colors_rtm, camera_top_rtm, global_lights)

                camera_front_before = renderer_before.create_camera(front_view_rotation, translation_front)
                camera_front = renderer.create_camera(front_view_rotation, translation_front)
                camera_front_rtm = renderer_rtm.create_camera(front_view_rotation, translation_front)
                img_front_before = renderer_before.render_with_ground(verts_before, faces_before, colors_before, camera_front_before, global_lights)
                img_front = renderer.render_with_ground(verts, faces, colors, camera_front, global_lights)
                img_front_rtm = renderer_rtm.render_with_ground(verts_rtm, faces_rtm, colors_rtm, camera_front_rtm, global_lights)

                camera_side_before = renderer_before.create_camera(side_view_rotation, translation_side)
                camera_side = renderer.create_camera(side_view_rotation, translation_side)
                camera_side_rtm = renderer_rtm.create_camera(side_view_rotation, translation_side)
                img_side_before = renderer_before.render_with_ground(verts_before, faces_before, colors_before, camera_side_before, global_lights)
                img_side = renderer.render_with_ground(verts, faces, colors, camera_side, global_lights)
                img_side_rtm = renderer_rtm.render_with_ground(verts_rtm, faces_rtm, colors_rtm, camera_side_rtm, global_lights)
            else:
                img_top_before = np.ones_like(img_original) * 255
                img_top = np.ones_like(img_original) * 255
                img_top_rtm = np.ones_like(img_original) * 255
                img_front_before = np.ones_like(img_original) * 255
                img_front = np.ones_like(img_original) * 255
                img_front_rtm = np.ones_like(img_original) * 255
                img_side_before = np.ones_like(img_original) * 255
                img_side = np.ones_like(img_original) * 255
                img_side_rtm = np.ones_like(img_original) * 255
                img_glob_before = np.ones_like(img_original) * 255
                img_glob = np.ones_like(img_original) * 255
                img_glob_rtm = np.ones_like(img_original) * 255

            # Concatenate all views in a 3x2 layout
            top_row = np.concatenate([img_original, img_overlaid_before, img_overlaid, img_overlaid_rtm, img_top_before, img_top, img_top_rtm], axis=1)
            bottom_row = np.concatenate([img_keypoints, img_side_before, img_side, img_side_rtm, img_front_before, img_front, img_front_rtm], axis=1)
            final_img = np.concatenate([top_row, bottom_row], axis=0)
        else:
            final_img = img_original  # If not vis_global, just use the original image

        writer.append_data(final_img)
        bar.next()
        frame_i += 1

    writer.close()

def run_vis_on_demo_smplify_rtm_mix(cfg, video, results, output_pth, smpl_before, smpl_rtm, smpl, vis_global=True):
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
            body_pose=tt(results[sid]['pose_world_after'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world_after'][:, :3]),
            betas=tt(results[sid]['betas_after']),
            transl=tt(results[sid]['trans_world_after']))
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
        top_view_rotation = torch.tensor(R.from_euler('xyz', [-90, 0, 180], degrees=True).as_matrix())
        front_view_rotation = torch.tensor(R.from_euler('y', 180, degrees=True).as_matrix())  # Rotate 180 degrees around y-axis to look from behind
        side_view_rotation = torch.tensor(R.from_euler('y', -90, degrees=True).as_matrix())  

        # Define custom translation matrices for top, front, and side views
        # test0
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test1
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test2
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test3
        # translation_top = torch.tensor([0.5, 0, 6])  # Move camera up along z-axis
        # translation_front = torch.tensor([0.5, -1, 6])  # Move camera back along y-axis
        # translation_side = torch.tensor([0.5, -1, 5])  # Move camera right along x-axis
        # test4
        # translation_top = torch.tensor([-1.5, 0, 6])  # Move camera up along z-axis
        # translation_front = torch.tensor([-1.5, -1, 6])  # Move camera back along y-axis
        # translation_side = torch.tensor([-0.5, -1, 7])  # Move camera right along x-axis
        # test5
        # translation_top = torch.tensor([-0.5, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([-0.5, -1, 4])  # Move camera back along y-axis
        # translation_side = torch.tensor([-0.5, -1, 5])  # Move camera right along x-axis
        # test6
        # translation_top = torch.tensor([0, 0, 5])  # Move camera up along z-axis
        # translation_front = torch.tensor([0, -1, 5])  # Move camera back along y-axis
        # translation_side = torch.tensor([0, -1, 5])  # Move camera right along x-axis
        # test7
        translation_top = torch.tensor([0, 1, 7])  # Move camera up along z-axis
        translation_front = torch.tensor([0, -1, 7])  # Move camera back along y-axis
        translation_side = torch.tensor([1, -1, 8])  # Move camera right along x-axis
   
    
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
            img_overlaid = renderer.render_mesh(torch.from_numpy(val['verts_after'][frame_i2]).to(cfg.DEVICE), img_overlaid)

            # Draw keypoints on the image
            if 'keypoints' in val and 'contact' in val:
                keypoints = val['keypoints'][frame_i2].reshape(-1, 3)
                contact_probs = val['contact'][frame_i2]
                max_contact_idx = np.argmax(contact_probs)  # Index of the foot keypoint with the highest contact probability
                left_foot_prob = contact_probs[0] + contact_probs[1]  # Sum of the first two values for left foot probability
                right_foot_prob = contact_probs[2] + contact_probs[3]  # Sum of the last two values for right foot probability

                for idx, joint in enumerate(keypoints):
                    x, y, confidence = joint
                    radius = int(max(1, confidence * 20))  # Scale the radius based on confidence
                    # color = (0, 255, 0) if idx in [20, 21, 22, 23] and idx == max_contact_idx + 20 else (255, 0, 0)  # Use green for the keypoint with the highest contact probability, red otherwise
                    # cv2.circle(img_keypoints, (int(x), int(y)), radius, color, -1)  # Draw the circle
                    color = (0, 255, 0) if idx == 15 and left_foot_prob > right_foot_prob else (0, 255, 0) if idx == 16 and right_foot_prob > left_foot_prob else (255, 0, 0)
                    cv2.circle(img_keypoints, (int(x), int(y)), radius, color, -1)  # Draw the circle

            if 'keypoints_rtm' in val:
                keypoints = val['keypoints_rtm'][frame_i2].reshape(-1, 3)

                for idx, joint in enumerate(keypoints):
                    x, y, confidence = joint
                    radius = int(max(1, confidence * 20))  # Scale the radius based on confidence
                    color = (0, 0, 255)
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
            top_row = np.concatenate([img_original, img_keypoints, img_top], axis=1)
            bottom_row = np.concatenate([img_overlaid, img_side, img_front], axis=1)
            final_img = np.concatenate([top_row, bottom_row], axis=0)
        else:
            final_img = img_original  # If not vis_global, just use the original image

        writer.append_data(final_img)
        bar.next()
        frame_i += 1

    writer.close()
