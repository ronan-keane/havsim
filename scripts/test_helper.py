from havsim import helper
from havsim import plotting
from IPython import embed
import numpy as np, pandas as pd
import pickle

with open('/home/jiwonkim/github/havsim/data/recon-ngsim.pkl', 'rb') as f:
    meas, platooninfo = pickle.load(f) #load data

txt = np.loadtxt('data/trajectories-0400-0415.txt')
# res = pd.read_csv('data/RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv')
all_veh_dict = helper.extract_lc_data(txt)

embed()
