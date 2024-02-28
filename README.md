## havsim - a traffic simulator written in python
havsim implements a parametric model of human driving, which also includes a stochastic model of human driving errors. havsim can realistically simulate traffic (including crashes and near misses) on an arbitrary highway network.

# Install
1. Download the repository
     - click code -> download zip
     - extract the havsim folder
2. With a console in the havsim file directory (the folder that contains setup.py), run 
```
pip install -e .
```

# Description
Refer to the provided scripts for examples of using havsim. The quickstart guide also explains how to create a simulation in havsim ([broken-link.asdf.lol.com]).

### The deterministic havsim model (havsim.Vehicle) 
- Based on traditional traffic flow models
- Car following model uses 6 parameters
       - output scalar acceleration from IDM - 5 parameters
       - bounded maximum deceleration - 1 parameter
- Lane changing model uses - 21 parameters
       - decision model based on MOBIL can output {stay, change left, change right} - 9 parameters
       - dynamics model outputs an acceleration which is added to the car following acceleration - 10 parameters
       - route model adjusts the lane changing state so vehicles will stay on their planned route - 2 parameters

### The stochastic havsim model (havsim.CrashesStochasticVehicle)
- Incorporates human driving errors into the model, which can lead to rear ends and sideswipe crashes.
- Stochastic human behavior - 7 parameters
       - human perception response time - 4 parameters
       - errors in lane changing decisions - 3 parameters

## Code Structure
scripts


# Usage
## Quickstart Example

# References
```
@article{havsim2021,
title = {A formulation of the relaxation phenomenon for lane changing dynamics in an arbitrary car following model},
journal = {Transportation Research Part C: Emerging Technologies},
volume = {125},
year = {2021},
doi = {https://doi.org/10.1016/j.trc.2021.103081},
author = {Ronan Keane and H. Oliver Gao}
}
```
