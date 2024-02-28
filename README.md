## havsim - a traffic simulator written in python
havsim implements a parametric model of human driving, which includes a stochastic model of human driving errors. havsim can realistically simulate traffic (including crashes and near misses) on an arbitrary highway network.

# Install
1. Download the repository from github
2. Open a console and navigate to the havsim project folder (the folder that contains setup.py). Run 
```
pip install -e .
```
  
# Description
Refer to the provided scripts for examples of using havsim.

## Code Structure
To use havsim, you should try using some of the included scripts, which are in the scripts folder. The havsim folder contains all the code used to implement the simulation and vehicle models.

     /scripts/
        analyze_crashes.py - Used to analyze output from safety_sim.py 
        safety_sim.py - Do parallel simulations, and save crashes only, or run a single simulation and save full output.
        bottleneck_simulation.py - Example of creating simple simulation from scratch
        quickstart_simulation.ipynb - quickstart guide for creating simulations in havsim
        make_simulation.py - Setup more complicated simulations
        plot_saved_simulation.py - Example of making plots from saved simulation
        safety_calibration.py - Example of calibrating stochastic safety parameters using crashes data
        cf_calibration.py - Example of calibrating car following parameters using trajectory data
     
     /havsim/
        vehicles.py - the Vehicle class and subclassed Vehicles
        road.py - the Road and Lane classes (used to define road networks and boundary conditions)
        simulation.py - the Simulation class which implements the simulation logic
        models.py - the havsim lane changing model (lc_havsim), IDM, default parameters
        plotting.py - included plotting functions
         ...

# The havsim model
## The deterministic havsim model (havsim.Vehicle) 
- Based on traditional traffic flow models
- 6 parameter car following model
  - acceleration from IDM - 5 parameters
  - bounded maximum deceleration - 1 parameter
- 21 parameter lane changing model
   - decision model based on MOBIL can output {stay, change left, change right} - 9 parameters
   - dynamics model outputs an acceleration which is added to the car following acceleration - 10 parameters
   - route model adjusts the lane changing state so vehicles will stay on their planned route - 2 parameters
- Default parameters given in havsim.models.default_parameters()

## The stochastic havsim model (havsim.CrashesStochasticVehicle)
- Incorporates human driving errors into the model, which can lead to rear ends and sideswipe crashes.
- 7 parameter model of human driving errors
  - human perception response time - 4 parameters
  - errors in lane changing decisions - 3 parameters
- Default parameters given in havsim.models.stochastic_default_parameters()
     
# Usage
You can see what havsim can do by running the included scripts. The beginning of scripts define what they will do, and can be changed accordingly.
### safety_sim.py
```
# -------  SETTINGS  ------- #
save_name: str, the output will be saved at havsim/scripts/pickle files/save_name.pkl
n_simulation: int number of simulations to run
n_workers: int number of workers to use for multiprocessing
batch_size: int number of simulations per batch (may run out of memory if batch_size is too large)
save_crashes_only: bool, by default, only save Vehicles which crash or have near misses, or have interaction with
     those crashed/near missed Vehicles. By default, if the n_simulation=1, then all Vehicles will be saved.
sim_name: str, name of simulation to run
use_times: list of \[start, end\], where start is the start time of the simulation and end is the end
     time of the simulation (times are floats between \[0-24\))
gamma_parameters: stochastic parameters to use for simulation (see havsim.vehicles.StochasticVehicle)
xi_parameters: stochastic parameters to use for simulation (see havsim.vehicles.StochasticVehicle)
# -------------------------- #
```
     
Example output:

### analyze_crashes.py
```
# -------  SETTINGS  ------- #
saved_sim: str, the name of the output (the save_name from safety_sim.py)
min_crash_plots: int, plot all crashes with index from min_crash_plots to max_crash_plots
max_crash_plots: int, plot all crashes with index from min_crash_plots to max_crash_plots
show_plots: bool, whether or not to show figures
save_plots: bool, whether or not to save figures/animations
# -------------------------- #
```
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
