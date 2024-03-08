import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import json
import copy
import numpy as np
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.data._custom import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

def run(cfg,
        video,
        rtm_pre,
        output_pth,
        network,
        calib=None,
        run_global=True,
        save_pkl=False,
        visualize=False):

    network_rtm = copy.deepcopy(network)
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    
    # Preprocess
    with torch.no_grad():
        if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and 
                osp.exists(osp.join(output_pth, 'slam_results.pth'))):
            
            detector = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            
            if run_global: slam = SLAMModel(video, output_pth, width, height, calib)
            else: slam = None
            
            bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break
                
                # 2D detection and tracking
                detector.track(img, fps, length)
                
                # SLAM
                if slam is not None: 
                    slam.track()
                
                bar.next()

            tracking_results = detector.process(fps)
            
            if slam is not None: 
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion
        
            # Extract image features
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
            tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')
            
            # Save the processed data
            joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
            joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Save processed data at {output_pth}')
        
        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
            slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
            
    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # run WHAM
    results = defaultdict(dict)
    
    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2
                
                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
            
            else:
                # data
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                
                # inference
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)

        pred_body_pose_before= matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root_before= matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world_before= matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose_before= np.concatenate((pred_root_before, pred_body_pose_before), axis=-1)
        pred_pose_world_before= np.concatenate((pred_root_world_before, pred_body_pose_before), axis=-1)
        pred_trans_before= (pred['trans_cam'] - network.output.offset).cpu().numpy()

        pred_contact = pred['contact'].cpu().squeeze().numpy()
        pred_keypoints = tracking_results[_id]['keypoints']

        smpl_before = copy.deepcopy(network.smpl)
        pred_before = copy.deepcopy(pred)
        network_before = copy.deepcopy(network)
        results[_id]['pose_before'] = pred_pose_before
        results[_id]['trans_before'] = pred_trans_before
        results[_id]['pose_world_before'] = pred_pose_world_before
        results[_id]['trans_world_before'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas_before'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts_before'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()       

        results[_id]['contact'] = pred_contact
        results[_id]['keypoints'] = pred_keypoints
        results[_id]['frame_ids'] = frame_id
        
        # if False:
        if args.run_smplify_rtm:
            tracking_results_rtm = 
            
            # Build dataset
            dataset_rtm = CustomDataset(cfg, tracking_results_rtm, slam_results, width, height, fps)
            
            # run WHAM
            # results_rtm = defaultdict(dict)
            
            n_subjs_rtm = len(dataset_rtm)
            for subj in range(n_subjs_rtm):
        
                with torch.no_grad():
                    if cfg.FLIP_EVAL:
                        # Forward pass with flipped input
                        flipped_batch_rtm = dataset_rtm.load_data(subj, True)
                        _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch_rtm
                        flipped_pred_rtm = network_rtm(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                        
                        # Forward pass with normal input
                        batch_rtm = dataset_rtm.load_data(subj)
                        _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                        pred_rtm = network_rtm(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                        
                        # Merge two predictions
                        flipped_pose_rtm, flipped_shape_rtm = flipped_pred_rtm['pose'].squeeze(0), flipped_pred_rtm['betas'].squeeze(0)
                        pose_rtm, shape_rtm = pred_rtm['pose'].squeeze(0), pred_rtm['betas'].squeeze(0)
                        flipped_pose_rtm, pose_rtm = flipped_pose_rtm.reshape(-1, 24, 6), pose_rtm.reshape(-1, 24, 6)
                        avg_pose_rtm, avg_shape_rtm = avg_preds(pose_rtm, shape_rtm, flipped_pose_rtm, flipped_shape_rtm)
                        avg_pose_rtm = avg_pose_rtm.reshape(-1, 144)
                        avg_contact_rtm = (flipped_pred_rtm['contact'][..., [2, 3, 0, 1]] + pred_rtm['contact']) / 2
                        
                        # Refine trajectory with merged prediction
                        network_rtm.pred_pose = avg_pose.view_as(network_rtm.pred_pose)
                        network_rtm.pred_shape = avg_shape.view_as(network_rtm.pred_shape)
                        network_rtm.pred_contact = avg_contact.view_as(network_rtm.pred_contact)
                        output_rtm = network_rtm.forward_smpl(**kwargs)
                        pred_rtm = network_rtm.refine_trajectory(output_rtm, cam_angvel, return_y_up=True)
                    
                    else:
                        # data
                        batch_rtm = dataset_rtm.load_data(subj)
                        _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                        
                        # inference
                        pred_rtm = network_rtm(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
            
            # smplify = TemporalSMPLify(network_before.smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            smplify = TemporalSMPLify(network_rtm.smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            # with open(rtm_pre, 'r') as f:
            #   data = json.load(f)

            # def select_keypoints_with_highest_confidence(instances):
            #   highest_confidence = 0
            #   selected_keypoints = None
            #   selected_scores = None
            #   for instance in instances:
            #     total_confidence = sum(instance['keypoint_scores'])
            #     if total_confidence > highest_confidence:
            #       highest_confidence = total_confidence
            #       selected_keypoints = instance['keypoints']
            #       selected_scores = instance['keypoint_scores']
            #   return selected_keypoints, selected_scores
            # formatted_data = [
            #   [
            #     [keypoint[0], keypoint[1], score]
            #     for keypoint, score in zip(instance['keypoints'], instance['keypoint_scores'])
            #   ]
            #   for frame in data
            #   for instance in frame['instances']
            # ]
            # 提取并处理关键点数据
            # formatted_data = []
            # for frame in data:
            #   # 检查 'instances' 是否非空
            #   if frame['instances']:
            #     # 选择置信度最高的关键点组及其对应的置信度分数
            #     selected_keypoints, selected_scores = select_keypoints_with_highest_confidence(frame['instances'])
            #     # 格式化关键点数据
            #     keypoints = [
            #         [keypoint[0], keypoint[1], score]
            #         for keypoint, score in zip(selected_keypoints, selected_scores)
            #     ]
            #     formatted_data.append(keypoints)
                # 只取第一组关键点
                # instance = frame['instances'][0]
                # keypoints = [
                #   [keypoint[0], keypoint[1], score]
                #   for keypoint, score in zip(instance['keypoints'], instance['keypoint_scores'])
                # ]
                # formatted_data.append(keypoints)
            #   else:
            #     # 如果 'instances' 为空，可以添加一个空列表或者其他占位符
            #     formatted_data.append([])
            # input_keypoints = np.array(formatted_data)
            
            # print(input_keypoints)
            # print(input_keypoints.shape)
            input_keypoints_rtm = dataset_rtm.tracking_results_rtm[_id]['keypoints']
            # results[_id]['keypoints_rtm'] = input_keypoints
            pred_rtm = smplify.fit(pred_rtm, input_keypoints_rtm, **kwargs)
            
            with torch.no_grad():
                network_rtm.pred_pose = pred_rtm['pose']
                network_rtm.pred_shape = pred_rtm['betas']
                network_rtm.pred_cam = pred_rtm['cam']
                output_rtm = network_rtm.forward_smpl(**kwargs)
                pred_rtm = network_rtm.refine_trajectory(output_rtm, cam_angvel, return_y_up=True)

        pred_body_pose_rtm = matrix_to_axis_angle(pred_rtm['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root_rtm = matrix_to_axis_angle(pred_rtm['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world_rtm = matrix_to_axis_angle(pred_rtm['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose_rtm = np.concatenate((pred_root_rtm, pred_body_pose_rtm), axis=-1)
        pred_pose_world_rtm = np.concatenate((pred_root_world_rtm, pred_body_pose_rtm), axis=-1)
        pred_trans_rtm = (pred_rtm['trans_cam'] - network_rtm.output.offset).cpu().numpy()

        pred_contact_rtm = pred_rtm['contact'].cpu().squeeze().numpy()
        pred_keypoints_rtm = tracking_results_rtm[_id]['keypoints']
        
        results[_id]['pose_rtm'] = pred_pose_rtm
        results[_id]['trans_rtm'] = pred_trans_rtm
        results[_id]['pose_world_rtm'] = pred_pose_world_rtm
        results[_id]['trans_world_rtm'] = pred_rtm['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas_rtm'] = pred_rtm['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts_rtm'] = (pred_rtm['verts_cam'] + pred_rtm['trans_cam'].unsqueeze(1)).cpu().numpy()

        results[_id]['contact_rtm'] = pred_contact_rtm
        results[_id]['keypoints_rtm'] = pred_keypoints_rtm

        smpl_rtm = copy.deepcopy(network_rtm.smpl)

        # if False:
        if args.run_smplify:
            smplify = TemporalSMPLify(network.smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            input_keypoints = dataset.tracking_results[_id]['keypoints']
            # print(input_keypoints)
            # print(input_keypoints.shape)
            pred_after = smplify.fit(pred_before, input_keypoints, **kwargs)
            
            with torch.no_grad():
                network.pred_pose = pred_after['pose']
                network.pred_shape = pred_after['betas']
                network.pred_cam = pred_after['cam']
                output = network.forward_smpl(**kwargs)
                pred_after = network.refine_trajectory(output, cam_angvel, return_y_up=True)
        
        # ========= Store results ========= #
        pred_body_pose_after = matrix_to_axis_angle(pred_after['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root_after = matrix_to_axis_angle(pred_after['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world_after = matrix_to_axis_angle(pred_after['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose_after = np.concatenate((pred_root_after, pred_body_pose_after), axis=-1)
        pred_pose_world_after = np.concatenate((pred_root_world_after, pred_body_pose_after), axis=-1)
        pred_trans_after = (pred_after['trans_cam'] - network.output.offset).cpu().numpy()
        
        results[_id]['pose_after'] = pred_pose_after
        results[_id]['trans_after'] = pred_trans_after
        results[_id]['pose_world_after'] = pred_pose_world_after
        results[_id]['trans_world_after'] = pred_after['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas_after'] = pred_after['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts_after'] = (pred_after['verts_cam'] + pred_after['trans_cam'].unsqueeze(1)).cpu().numpy()

        smpl_after = copy.deepcopy(network.smpl)
    
    if save_pkl:
        joblib.dump(results, osp.join(output_pth, "wham_output.pkl"))
     
    # Visualize
    if visualize:
        if args.run_smplify:
            from lib.vis.run_vis_cc import run_vis_on_demo_smplify
            with torch.no_grad():
                run_vis_on_demo_smplify(cfg, video, results, output_pth, smpl_before, network.smpl, vis_global=run_global)
        elif args.run_smplify_rtm:
            from lib.vis.run_vis_cc import run_vis_on_demo_smplify_rtm
            with torch.no_grad():
                run_vis_on_demo_smplify_rtm(cfg, video, results, output_pth, smpl_before, smpl_after, smpl_rtm, vis_global=run_global)
        else:
            from lib.vis.run_vis_cc import run_vis_on_demo
            with torch.no_grad():
                run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, 
                        default='examples/demo_video.mp4', 
                        help='input video path or youtube link')

    parser.add_argument('--rtm_pre', type=str, 
                        default='examples/demo_video.json', 
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='output/demo', 
                        help='output folder to write results')
    
    parser.add_argument('--calib', type=str, default=None, 
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')
    
    parser.add_argument('--save_pkl', action='store_true',
                        help='Save output as pkl file')
    
    parser.add_argument('--run_smplify', action='store_true',
                        help='Run Temporal SMPLify for post processing')

    parser.add_argument('--run_smplify_rtm', action='store_true',
                        help='Run Temporal SMPLify for post processing')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Output folder
    sequence = '.'.join(args.video.split('/')[-1].split('.')[:-1])
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    run(cfg, 
        args.video, 
        args.rtm_pre,
        output_pth, 
        network, 
        args.calib, 
        run_global=not args.estimate_local_only, 
        save_pkl=args.save_pkl,
        visualize=args.visualize)
        
    print()
    logger.info('Done !')
