"""Example of shockwave in ngsim I-80 data."""
import havsim
import pickle

with open('recon-ngsim-old.pkl', 'rb') as f:
    meas, platooninfo = pickle.load(f)

platoon = [1732, 1739, 1746, 1748, 1755, 1765, 1772, 1780, 1793, 1795, 1804, 1810, 1817, 1821, 
           1829, 1845, 1851, 1861, 1868, 1873, 1882, 1887, 1898, 1927, 1929 ,1951 ,1941 ,1961, 1992 ,
           1984 ,1998, 2006, 2019 ,2022, 2035, 2042, 2050, 2058, 2065, 2071, 2095, 2092, 2097, 2108, 2113, 
           2122, 2128, 2132, 2146, 2151, 2158, 2169, 2176, 2186, 2190, 2199, 2234, 2253,]

havsim.plotting.animatetraj(meas,platooninfo,platoon=platoon,speed_limit=[0,50], show_ID=False)