import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
#specify the path to the root data directory here
data_dir = '/Users/junhanh/ML/playerTracking/data/'
#specify the game name
game_name = '0En5pOUZN5M'
#specify the clip name
clip_name = 'clip_46'
#specify the image number
img_num = 1
im_path = data_dir + game_name + '/' + clip_name + '/' + '{:02}.png'.format(img_num)
bb_path = data_dir + game_name+ '/' + clip_name + '/' + '{:02}_info.csv'.format(img_num)
bbs_data = pd.read_csv(bb_path)
im = mpimg.imread(im_path)
fig = plt.figure(frameon = False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(im)

#iterating over bounding boxes
for _, bbox in bbs_data.iterrows():
    print(bbox)
    rect = patches.Rectangle((bbox.x, bbox.y), bbox.w, bbox.h, linewidth=3, facecolor='red',fill=False)
    ax.add_patch(rect)

plt.show()