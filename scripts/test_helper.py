from havsim import helper
from havsim import plotting
from IPython import embed
import numpy as np, pandas as pd
import pickle

with open('/home/jiwonkim/github/havsim/data/recon-ngsim.pkl', 'rb') as f:
    meas, platooninfo = pickle.load(f) #load data

txt = np.loadtxt('data/trajectories-0400-0415.txt')
# 7-> global_y, 13->lane, 1->frame_id, 0->veh_id
print('global_y of 2022 in frame 6100', txt[(txt[:, 0] == 2022) & (txt[:, 1] == 6100), 7])
print('lane of 2022 in frame 6100', txt[(txt[:, 0] == 2022) & (txt[:, 1] == 6100), 13])

lane1_6100 = txt[(txt[:, 13] == 1) & (txt[:, 1] == 6100), :]

print('pos of lane 1 in frame 6100', lane1_6100[lane1_6100[:, 7].argsort(), 7])
print('vehid of lane 1 in frame 6100', lane1_6100[lane1_6100[:, 7].argsort(), 0])
print('this implies that lfol of 2022 in frame 6100 is 1962')

embed()
1/0

# res = pd.read_csv('data/RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv')
all_veh_dict = helper.extract_lc_data(txt)

embed()
