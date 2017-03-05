import pandas as pd
from pathlib import Path
import shutil

#This file will generate training labels for YOLO2 detector
#Check out https://pjreddie.com/darknet/yolo/ for training instruction

#specify the path to the root data directory here
data_dir = '/Users/junhanh/ML/playerTracking/data/'
target_dir = '/Users/junhanh/ML/playerTracking/training_data/'
train_num = 6000

train_file_path = target_dir + 'pt_train.txt'
test_file_path = target_dir + 'pt_test.txt'
train = open(train_file_path, 'w+')
test = open(test_file_path, 'w+')

counter = 0
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

        for img_path in clip_path.iterdir():
            if (img_path.suffix == ".png"):
                img_name = img_path.stem
                im_path = data_dir + game_name + '/' + clip_name + '/' + img_name + '.png'
                bb_path = data_dir + game_name + '/' + clip_name + '/' + img_name + '_info.csv'
                tl_path = target_dir + 'labels/' + '{0:05d}'.format(counter) + '.txt'
                td_path = target_dir + 'data/' + '{0:05d}'.format(counter) + '.png'
                bbs_data = pd.read_csv(bb_path)

                skip = False
                for _, bbox in bbs_data.iterrows():
                    if (bbox.x < 0) or (bbox.x > v_w) or (bbox.y < 0) or (bbox.y > v_h):
                        skip = True
                        break
                if (skip):
                    continue

                tlf = open(tl_path, 'w+')
                shutil.copyfile(im_path, td_path)
                #iterating over bounding boxes
                for _, bbox in bbs_data.iterrows():
                    print("0", round(bbox.x / v_w, 3), round(bbox.y/ v_h, 3), round(bbox.w / v_w, 3), round(bbox.h / v_h, 3), file = tlf)
                tlf.close()

                if (counter < train_num):
                    print(td_path, file = train)
                else:
                    print(td_path, file = test)

                counter = counter + 1

train.close()
test.close()
