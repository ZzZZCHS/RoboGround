import os

import h5py
import os
from PIL import Image
import numpy as np
import json
import cv2
import base64
from gpt_utils import get_response_with_image
import re
import logging
from datetime import datetime
from scipy.spatial.transform import Rotation as R

def transform_pose(world_obj_pos, world_obj_quat, world_robot_pos, world_robot_quat):
    # Convert world object position and quaternion to arrays
    world_obj_pos = np.array(world_obj_pos)
    world_obj_quat = np.array(world_obj_quat)

    # Convert world robot position and quaternion to arrays
    world_robot_pos = np.array(world_robot_pos)
    world_robot_quat = np.array(world_robot_quat)

    # Step 1: Compute the inverse of the robot's world pose
    # Inverse rotation (quaternion)
    robot_rotation_inv = R.from_quat(world_robot_quat).inv()

    # Inverse translation: Rotate the translation part and negate it
    robot_translation_inv = -robot_rotation_inv.apply(world_robot_pos)

    # Step 2: Apply the inverse transformation to the object's world pose
    # Transform position
    obj_pos_in_robot_frame = robot_rotation_inv.apply(world_obj_pos + robot_translation_inv)

    # Transform orientation
    obj_rotation_in_world = R.from_quat(world_obj_quat)
    obj_rotation_in_robot_frame = robot_rotation_inv * obj_rotation_in_world

    # Get quaternion for object's orientation in robot base frame
    obj_quat_in_robot_frame = obj_rotation_in_robot_frame.as_quat()

    return obj_pos_in_robot_frame, obj_quat_in_robot_frame

# 生成带时间戳的日志文件名
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f'./logs/output_{current_time}.log'

# 配置日志设置
logging.basicConfig(
    filename=log_filename,  # 设置日志文件名
    filemode='w',           # 以写模式打开（每次运行会覆盖旧日志）
    level=logging.INFO,     # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 设置日志格式
)

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)  # Assumes OpenCV; adjust if needed
    return base64.b64encode(buffer).decode('utf-8')

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def isvalid(pos_a, pos_b):
    flag = True
    for index in range(3):
        if abs(abs(pos_a[index] - pos_b[index])) > 0.5:
            flag = False
    return flag


# directory_path = "/mnt/hwfile/OpenRobotLab/huanghaifeng/data/generated_data"
# file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.hdf5')]
file_paths = [
    # '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPCabToCounter.hdf5',
    # '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPCounterToCab.hdf5',
    # '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPCounterToMicrowave.hdf5',
    # '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPCounterToSink.hdf5',
    # '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPCounterToStove.hdf5',
    '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPMicrowaveToCounter.hdf5',
    '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPSinkToCounter.hdf5',
    '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_0412/PnPStoveToCounter.hdf5'
]

output_dir = '/ailab/user/chenxinyi1/group/haifeng/robocasa_exps_haifeng/data_gen/gpt'
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
    instruction_template="pick {} from the counter and place it in the cabinet"


    common_sense_prompt_template="""
    These images depict a kitchen scene where a Franka robotic arm is interacting with various items. The robot is given a pick-and-place instruction to perform a manipulation task.
    Three viewpoints are provided: left view, right view and robot hand view. Each viewpoint includes two types of images.
    The target object, <TARGET_OBJECT>, is highlighted in red, while the target placement area, '<TARGET_PLACE>', is marked in green. Other objects present in the scene are listed as <OBJECTS_LIST>.

    Please generate five new instructions that requires common-sense reasoning of the target object for me, remember that you cannot change the target place, remember don't mention the mask thing.
    ATTENTION: ONLY mention objects that exist in the scene.

    For example, the target object is kettle, the expected answer format is:
    <ANSWER>
    <1> I want to drink, please pick the related object to the <TARGET_PLACE>. </1>
    <2> I'm thirsty, please pick the related object to the <TARGET_PLACE>. </2>
    <3> I'm going for a hike and need to stay hydrated, please pick up the appropriate item and place it in the <TARGET_PLACE>. </3> 
    <4> I need to refill something for gardening, could you put the refillable item into the <TARGET_PLACE>? </4>
    <5> I'm heading to the gym and need to fill up my container, place it in the <TARGET_PLACE>.</5>
    </ANSWER>
    """


    output_filename = os.path.join(output_dir, f'output/{env_name}.json')
    logging.info(f"save at {output_filename}")
    cnt = 0
    gen_instructions = dict()
    if os.path.exists(output_filename):
        gen_instructions = json.load(open(output_filename, 'r'))

    for demo_id in f['data']:
        try:
            demo_name = f"{env_name}_{demo_id}"
            if demo_name in gen_instructions:
                continue
            demo_item = {}
            cnt += 1
            if cnt % 100 == 0:
                save_to_json(gen_instructions, output_filename)
                logging.info(f"save to json, count: {cnt}")
            ep_meta = json.loads(f[f"data/{demo_id}"].attrs['ep_meta'])
            cam_configs = ep_meta['cam_configs']['robot0_frontview']
            cam_pos = cam_configs['pos']
            cam_quat =cam_configs['quat']

            object_cfgs = ep_meta['object_cfgs']
            attr_lang = ep_meta['lang']
            unique_attr = ep_meta['unique_attr']
            target_obj_phrase = ep_meta['target_obj_phrase']
            target_obj_class = ''
            target_obj_name = ''
            target_place_phrase = ep_meta['target_place_phrase']
            base_pos = f[f"data/{demo_id}/obs/robot0_base_pos"][:1]
            base_quat = f[f"data/{demo_id}/obs/robot0_base_quat"][:1]
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
                # obj_pos = item['qpos'][:3]
                # obj_quat = item['qpos'][3:]
                # relative_pos, _ = transform_pose(obj_pos, obj_quat, base_pos, base_quat)
                # relative_pos, _ = transform_pose(obj_pos, obj_quat, cam_pos, cam_quat)
                relative_pos = item['qpos'][:3]
                # print(relative_pos)
                
                # obj_dict[name].update({'relative_robot_pos': relative_pos.flatten()})
                obj_dict[name].update({'relative_robot_pos': relative_pos})
            obj_list = []
            pos_dict = dict()
            for name, item in obj_dict.items():
                obj_list.append(item['class'])
                pos_dict[item['id']] = item['relative_robot_pos']

            encoded_images = []
            prompt = common_sense_prompt_template
            prompt = prompt.replace('<TARGET_OBJECT>', f"'{target_obj_class}'")
            prompt = prompt.replace('<TARGET_PLACE>', f"{target_place_phrase}")
            prompt = prompt.replace('<OBJECTS_LIST>', ', '.join(obj_list))

            for view in views:
                image_name = view + '_image'
                mask_name = view + '_mask'
                first_frame = f[f'data/{demo_id}/obs/{image_name}'][0]
                encoded_images.append(encode_image_to_base64(first_frame))
                mask = f[f'data/{demo_id}/obs/{mask_name}'][0].squeeze()

                mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                mask_rgb[mask[...] == 1] = [255, 0, 0]  # 红色 target_object
                mask_rgb[mask[...] == 2] = [0, 255, 0]  # 绿色 target_place

                alpha = 0.7  # 透明度
                combined_image = first_frame * (1 - alpha) + mask_rgb * alpha
                combined_image = combined_image.astype(np.uint8)

                encoded_images.append(encode_image_to_base64(first_frame))
                encoded_images.append(encode_image_to_base64(combined_image))

            # todo: 再调一下prompt
            # 生成common sense的instructions
            try:
                response = get_response_with_image(
                    image_paths=[],
                    text_prompt=prompt,
                    encoded_images=encoded_images
                )
                common_sense_list = re.findall(r"<\d+>\s*(.*?)\s*</\d+>", response)
                logging.info(common_sense_list)     
                demo_item.update({"common-sense": common_sense_list})
            except Exception as e:
                logging.error(e)
                logging.error(f"Scene {demo_id} fails to generate.")
                demo_item.update({"common-sense": "fail"})
                continue
            logging.info(f'{demo_id}: {target_obj_class}/{target_obj_phrase} -> {target_place_phrase}')
            logging.info(response)

            # 存储
            gen_instructions[demo_name] = demo_item
            print(f"{demo_name} save.")
    
        except Exception as e:
            logging.error(e)
            gen_instructions[demo_name] = demo_item
            logging.error(f"{demo_name} fails.")

    save_to_json(gen_instructions, output_filename)
    logging.info(f"finish generating {env_name} instructions")
