import pandas as pd
from pathlib import Path

#specify the path to the root data directory here
data_dir = '/Users/junhanh/ML/playerTracking/data/'
data_path = Path(data_dir)
for game_path in data_path.iterdir():
    game_name = game_path.name
    if (game_name == ".DS_Store") or (game_name == "events.pkl"):
        continue
    for clip_path in game_path.iterdir():
        clip_name = clip_path.name
        if (clip_name == ".DS_Store"):
            continue
        for img_path in clip_path.iterdir():
            if (img_path.suffix == ".png"):
                img_name = img_path.stem
                bb_path = data_dir + game_name+ '/' + clip_name + '/' + img_name + '_info.csv'
                tl_path = data_dir + game_name+ '/' + clip_name + '/' + img_name + '_tl.txt'
                bbs_data = pd.read_csv(bb_path)
                tlf = open(tl_path, 'w+')

                #iterating over bounding boxes
                for _, bbox in bbs_data.iterrows():
                    print("person", round(bbox.x, 3), round(bbox.y, 3), round(bbox.w, 3), round(bbox.h, 3), file = tlf)
                tlf.close()
