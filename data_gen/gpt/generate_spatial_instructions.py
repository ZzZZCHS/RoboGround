import h5py
import os
from PIL import Image
import numpy as np
import json
import cv2
import base64
# from gpt_utils import get_response_with_image
import re
import logging
from datetime import datetime
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import argparse
from tqdm import tqdm

def pose_to_matrix(position, quaternion):
    """
    Convert position and quaternion (x, y, z, w format) to a 4x4 transformation matrix.
    """
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix

def matrix_to_pose(matrix):
    """
    Convert a 4x4 transformation matrix back to position and quaternion.
    """
    rotation_matrix = matrix[:3, :3]
    quaternion = R.from_matrix(rotation_matrix).as_quat()

    position = matrix[:3, 3]

    return position, quaternion

def transform_pose(world_obj_pos, world_obj_quat, world_robot_pos, world_robot_quat):
    obj_matrix = pose_to_matrix(world_obj_pos, world_obj_quat)
    robot_matrix = pose_to_matrix(world_robot_pos, world_robot_quat)

    robot_matrix_inv = np.linalg.inv(robot_matrix)

    obj_in_robot_frame_matrix = robot_matrix_inv @ obj_matrix

    obj_pos_in_robot_frame, obj_quat_in_robot_frame = matrix_to_pose(obj_in_robot_frame_matrix)

    return obj_pos_in_robot_frame, obj_quat_in_robot_frame

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f'./logs/output_{current_time}.log'

logging.basicConfig(
    filename=log_filename,
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def isvalid(pos_a, pos_b, limit, exclude_idx=-1):
    for index in range(3):
        if index == exclude_idx:
            continue
        
        if abs(pos_a[index] - pos_b[index]) > limit * (1/math.sqrt(3)):
            return False
    return True

# parser = argparse.ArgumentParser()
# parser.add_argument('--file_path', type=str)
# args = parser.parse_args()
# file_path = "./data/demo_gentex_im128_randcams_addobj_use_actions_hhf.hdf5"
# file_path = args.file_path

file_paths = [
    '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPCabToCounter.hdf5',
    '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPCounterToCab.hdf5',
    '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPCounterToMicrowave.hdf5',
    '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPCounterToSink.hdf5',
    '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPCounterToStove.hdf5',
    '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPMicrowaveToCounter.hdf5',
    '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPSinkToCounter.hdf5',
    '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPStoveToCounter.hdf5'
]

output_dir = '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/data_gen/gpt/spatial_results'
os.makedirs(output_dir, exist_ok=True)

for file_path in file_paths:
    logging.info(f"generate instructions for {file_path}")
    print(f"generate instructions for {file_path}")
    f = h5py.File(file_path, 'r')

    env_args = json.loads(f['data'].attrs['env_args'])
    env_name = env_args['env_name']
    env_kwargs = env_args['env_kwargs']
    camera_names = list(env_kwargs['camera_names'])
    views = ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
    view_names = ['robot0_agentview_left_image', 'robot0_agentview_right_image', 'robot0_eye_in_hand_image']
    mask_names = ['robot0_agentview_left_mask', 'robot0_agentview_right_mask', 'robot0_eye_in_hand_mask']


    output_filename = os.path.join(output_dir, f'{env_name}.json')
    cnt = 0
    gen_instructions = dict()

    for demo_id in tqdm(f['data']):
        try:
            demo_name = f"{env_name}_{demo_id}"
            demo_item = {}
            cnt += 1
            if cnt % 100 == 0:
                save_to_json(gen_instructions, output_filename)
                logging.info(f"save to json, count: {cnt}")
            ep_meta = json.loads(f[f"data/{demo_id}"].attrs['ep_meta'])
            cam_configs = ep_meta['cam_configs']['robot0_frontview']
            cam_pos = cam_configs['pos']
            cam_quat = cam_configs['quat']

            object_cfgs = ep_meta['object_cfgs']
            attr_lang = ep_meta['lang']
            unique_attr = ep_meta['unique_attr']
            target_obj_phrase = ep_meta['target_obj_phrase']
            target_obj_class = ''
            target_obj_name = ''
            target_place_phrase = ep_meta['target_place_phrase']
            base_pos = f[f"data/{demo_id}/obs/robot0_base_pos"][0]
            base_quat = f[f"data/{demo_id}/obs/robot0_base_quat"][0]
            logging.info(f"{demo_id} robot base pos: {base_pos}")
            logging.info(f"{demo_id} base quat: {base_quat}")


            obj_infos = json.loads(f[f"data/{demo_id}"].attrs['obj_infos'])
            obj_dict = {}
            
            for object_cfg in ep_meta['object_cfgs']:
                obj_dict[object_cfg['name']] = object_cfg['info']
                if object_cfg['name'] == 'obj':
                    target_obj_class = object_cfg['info']['cat']
                    target_obj_name = object_cfg['target_obj_name']

            for name, item in obj_infos.items():
                obj_dict[name].update(item)
                obj_class = obj_dict[name]['mjcf_path'].split('/')[-3]
                obj_dict[name].update({'class': obj_class})
                obj_pos = item['qpos'][:3]
                obj_quat = item['qpos'][3:]
                relative_pos, _ = transform_pose(obj_pos, obj_quat, base_pos, base_quat)
                # relative_pos, _ = transform_pose(obj_pos, obj_quat, cam_pos, cam_quat)
                # relative_pos = item['qpos'][:3]
                # print(relative_pos)
                
                # obj_dict[name].update({'relative_robot_pos': relative_pos.flatten()})
                obj_dict[name].update({'relative_robot_pos': relative_pos})
            obj_list = []
            pos_dict = dict()
            occur_num = defaultdict(int)
            id2class = dict()
            for name, item in obj_dict.items():
                obj_list.append(item['class'])
                pos_dict[item['id']] = item['relative_robot_pos']
                class_name = ' '.join(item['id'].split('_')[1:-1])
                occur_num[class_name] += 1
                id2class[item['id']] = class_name

            obj_pos_flag = False
            for name, item in pos_dict.items():
                if name == target_obj_name:
                    obj_pos = item
                    obj_pos_flag = True
                    break

            # x+: behind, x-: front, y+: left, y-: right
            if obj_pos_flag:
                spatial_instr_list =[]
                left_min = 100
                left_name = ''
                right_min = 100
                right_name = ''
                for name, pos in pos_dict.items():
                    if name == target_obj_name:
                        continue
                    dis = pos[1] - obj_pos[1]
                    if dis > 0:
                        if left_min > abs(dis) and isvalid(pos, obj_pos, abs(dis), 1):
                            left_min = abs(dis)
                            left_name = name
                            # print(f"left: {name}: {dis} {occur_num[id2class[name]]}")
                    else:
                        if right_min > abs(dis) and isvalid(pos, obj_pos, abs(dis), 1):
                            right_min = abs(dis)
                            right_name = name
                            # print(f"right: {name}: {dis} {occur_num[id2class[name]]}")
                            # print(pos)
            
                if left_min != 100 and left_name != '' and occur_num[id2class[left_name]] == 1:
                    left_instr = f"pick the object to the right of the {id2class[left_name]} and place it to the {target_place_phrase}"
                    spatial_instr_list.append(left_instr)

                if right_min != 100 and right_name != '' and occur_num[id2class[right_name]] == 1:
                    right_instr = f"pick the object to the left of the {id2class[right_name]} and place it to the {target_place_phrase}"
                    spatial_instr_list.append(right_instr)

                front_min = 100
                front_name = ''
                behind_min = 100
                behind_name = ''
                for name, pos in pos_dict.items():
                    if name == target_obj_name:
                        continue
                    dis = pos[0] - obj_pos[0]
                    if dis < 0:
                        if front_min > abs(dis) and isvalid(pos, obj_pos, abs(dis), 0):
                            front_min = abs(dis)
                            front_name = name
                    else:
                        if behind_min > abs(dis) and isvalid(pos, obj_pos, abs(dis), 0):
                            behind_min = abs(dis)
                            behind_name = name
                if front_min != 100 and front_name != '' and occur_num[id2class[front_name]] == 1:
                    front_str = f"pick the object behind the {id2class[front_name]} and place it to the {target_place_phrase}"
                    spatial_instr_list.append(front_str)
                if behind_min != 100 and behind_name != '' and occur_num[id2class[behind_name]] == 1:
                    behind_str = f"pick the object in front of the {id2class[behind_name]} and place it to the {target_place_phrase}"
                    spatial_instr_list.append(behind_str)

                demo_item.update({"spatial": spatial_instr_list})

                # print(pos_dict)
            else:
                demo_item.update({"spatial": "fail"})

            gen_instructions[demo_name] = demo_item

        except Exception as e:
            logging.error(e)
            gen_instructions[demo_name] = demo_item
            logging.error(f"{demo_name} fails.")


    save_to_json(gen_instructions, output_filename)
    logging.info("finish generating instructions")
