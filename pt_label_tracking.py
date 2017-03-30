import pandas as pd
from pathlib import Path
import shutil
import os
import csv

#This file will generate training labels for MOTC based tracker

#specify the path to the root data directory here
data_dir = '/Users/junhanh/ML/playerTracking/data/'
target_dir = '/Users/junhanh/ML/playerTracking/tracking_data/'

clip_counter = 0
frame_counter = 0
data_path = Path(data_dir)

for game_path in data_path.iterdir():
    game_name = game_path.name
    if (game_name == ".DS_Store") or (game_name == "events.pkl"):
        continue
    for clip_path in game_path.iterdir():
        clip_name = clip_path.name

        if (clip_name == ".DS_Store"):
            continue

        cp_path = bb_path = data_dir + game_name + '/' + clip_name + '/clip_info.csv'
        cp_data = pd.read_csv(cp_path).set_index('index').to_dict()
        cp_dict = cp_data[list(cp_data.keys())[0]]
        v_w = float(cp_dict['vid_w'])
        v_h = float(cp_dict['vid_h'])
        if (v_w != 490) or (v_h != 360):
            continue 
        
        clip_tar_path = target_dir + '{0:03d}'.format(clip_counter) + "/"
        dect_path = clip_tar_path + 'detection.csv'
        os.makedirs(os.path.dirname(dect_path), exist_ok=True)
        gt_path = clip_tar_path + 'groundtruth.csv'
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)

        tid_dict = {}
        tid_counter = 0

        dict_file = open(dect_path, 'w', newline='')
        gt_file = open(gt_path, 'w', newline='')
        dect_writer = csv.writer(dict_file, delimiter=',')
        gt_writer = csv.writer(gt_file, delimiter=',')

        for img_path in clip_path.iterdir():
            if (img_path.suffix == ".png"):
                img_name = img_path.stem
                im_path = data_dir + game_name + '/' + clip_name + '/' + img_name + '.png'
                bb_path = data_dir + game_name + '/' + clip_name + '/' + img_name + '_info.csv'

                td_path = clip_tar_path + '{0:06d}'.format(frame_counter) + '.png'

                bbs_data = pd.read_csv(bb_path)

                skip = False
                for _, bbox in bbs_data.iterrows():
                    if (bbox.x < 0) or (bbox.x > v_w) or (bbox.y < 0) or (bbox.y > v_h):
                        skip = True
                        break
                if (skip):
                    continue

                
                shutil.copyfile(im_path, td_path)
                #iterating over bounding boxes
                for _, bbox in bbs_data.iterrows():
                    if bbox.id not in tid_dict:
                        tid_dict[bbox.id] = tid_counter
                        tid_counter = tid_counter + 1
                    dect_writer.writerow([frame_counter, '-1', round(bbox.x, 3), round(bbox.y + bbox.h, 3), round(bbox.w, 3), round(bbox.h, 3), '1', '-1', '-1'])
                    gt_writer.writerow([frame_counter,tid_dict[bbox.id] , round(bbox.x, 3), round(bbox.y + bbox.h, 3), round(bbox.w, 3), round(bbox.h, 3), '1', '-1', '-1'])
                frame_counter = frame_counter + 1
        dict_file.close()
        gt_file.close()
        clip_counter = clip_counter + 1
        frame_counter = 0
