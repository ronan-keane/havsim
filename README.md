## havsim - a traffic simulator written in python

# Install
1. Download the repository
     - click code -> download zip
     - extract the havsim folder
2. With a console in the havsim file directory (the folder that contains setup.py), run 
```
pip install -e .
```

# Description
havsim implements a parametric model of human driving, and can simulate traffic on an arbitrary highway network.
The **deterministic havsim model** (havsim.Vehicle) is based on traditional traffic flow models. It consists of the following parts:
1. Car following model (IDM) - 5 parameters
      - outputs scalar acceleration
2. Lane changing model (lc_havsim) - 21 parameters
      - decision model based on MOBIL can output {stay, change left, change right} - 9 parameters
      - dynamics model outputs an acceleration which is added to the car following acceleration - 10 parameters
      - route model adjusts the lane changing state so the vehicle will stay on the planned route - 2 parameters

The **stochastic havsim model** (havsim.CrashesStochasticVehicle) incorporates human driving errors into the model, which can lead to rear ends and sideswipe crashes.
3. Stochastic human behavior - 7 parameters
      - human perception response time - 4 parameters
      - errors in lane changing decisions - 3 parameters

## Code Structure
scripts


# Usage
## Example

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
