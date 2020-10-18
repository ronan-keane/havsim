from havsim import helper
from havsim import plotting
from havsim.calibration.deep_learning import make_dataset2, make_dataset, generate_lane_data
from IPython import embed
import numpy as np
import pickle
import time

with open('/home/jiwonkim/github/havsim/data/recon-ngsim.pkl', 'rb') as f:
    meas, platooninfo = pickle.load(f) #load data

# res = pd.read_csv('data/RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv')

# txt = np.loadtxt('data/trajectories-0400-0415.txt')

# start = time.time()
# txt = np.loadtxt('C:/Users/rlk268/OneDrive - Cornell University/important misc/datasets/ngsim trajectory data/i 80/trajectories-0400-0415.txt')
# txt = np.loadtxt('/home/rlk268/Downloads/trajectories-0400-0415.txt')
# txt = np.loadtxt('data/trajectories-0400-0415.txt')
# print(time.time()-start)

# all_veh_dict = helper.extract_lc_data(txt)
# with open('data/veh_dict.pckl', 'wb') as f:
#     pickle.dump(all_veh_dict, f)
with open('data/veh_dict.pckl', 'rb') as f:
    all_veh_dict = pickle.load(f)

# res = generate_lane_data(all_veh_dict[1588.0])

# res = make_dataset(meas, platooninfo, list(meas.keys()))
ds = make_dataset2(all_veh_dict, dt=0.1)
embed()


#%%

from havsim import helper
import numpy as np
import pickle

txt = np.loadtxt('C:/Users/rlk268/OneDrive - Cornell University/important misc/datasets/ngsim trajectory data/recon ngsim/DATA (NO MOTORCYCLES).txt')

all_veh_dict = helper.load_dataset(txt, column_dict={'veh_id':0, 'frame_id':1, 'lane':2, 'pos':3, 'veh_len':6,
                                      'lead':9, 'fol':8}, alpha=0, None_val=-1)

# plotting.animatetraj(meas, platooninfo, usetime=[2801, 2801])


