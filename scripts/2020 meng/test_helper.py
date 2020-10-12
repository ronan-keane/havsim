from havsim import helper
from havsim import plotting
# from IPython import embed
import numpy as np
import pickle
import time

# with open('/home/jiwonkim/github/havsim/data/recon-ngsim.pkl', 'rb') as f:
    # meas, platooninfo = pickle.load(f) #load data

# res = pd.read_csv('data/RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv')

# txt = np.loadtxt('data/trajectories-0400-0415.txt')

# start = time.time()
txt = np.loadtxt('C:/Users/rlk268/OneDrive - Cornell University/important misc/datasets/ngsim trajectory data/i 80/trajectories-0400-0415.txt')
# print(time.time()-start)

all_veh_dict = helper.extract_lc_data(txt)

# embed()
