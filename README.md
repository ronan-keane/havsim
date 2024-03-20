## havsim - a traffic simulator written in python
havsim implements a parametric model of human driving, which includes a stochastic model of human driving errors. It can realistically simulate traffic flow, including safety events, on an arbitrary highway network.

# Install
1. Download the repository from github
2. Open a console and navigate to the havsim project folder (the folder that contains setup.py). Run 
```
pip install -e .
```
  
# Usage
If you've never used havsim before, the first thing you should do is look at the included scripts. Those scripts explain how to use havsim to simulate traffic, generate safety events, and do plotting/saving. 
If you want to try setting up your own simulation in havsim, you should also look at the included notebook.

## Scripts
You can see the documentation for any script by running
```
python path_to_script.py -h
```
where "path_to_script" would be replaced with the relevant filepath. 

All scripts can be run with default settings like
```
python path_to_script.py
```

To manually specify arguments, one can use positional arguments (like normal), but you can also pass arguments as keyword arguments using python syntax
```
python path_to_script.py 0 1 some_arg_name="..."
```
In the example above, the first argument gets 0, the second arguments gets 1 and the argument "some_arg_name" gets the value ...

*Note that any saved files (e.g. saved Vehicles) will go to the folder "scripts\pickle files" and any saved plots/animations will go to the folder "scripts\plots and animations"*

### bottleneck_simulation.py - Run a single simulation of a merge bottleneck, save the result, and plot it.
```
python scripts\bottleneck_simulation.py
```
### safety_sim.py - Run multiple simulations of I94 in Ann Arbor, and save the result 
```
python scripts\safety_sim.py
```
### plot_saved_simulation.py - Load a saved simulation, and plot the result
```
python scripts\plot_saved_simulation.py save_name='e94_16_17_test'
```
### analyze_crashes.py - From saved vehicles/simulations, plot and save all the crashes/near misses and print out statistics
```
python scripts\analyze_crashes.py
```
### safety_calibration.py - calibrate the StochasticCrashesVehicle parameters using bayesian optimization
```
python scripts\safety_calibration.py
```

# The havsim code structure and model explanation
```
     /havsim/
        vehicles.py - the Vehicle class and subclassed Vehicles
        road.py - the Road and Lane classes (used to define road networks and boundary conditions)
        simulation.py - the Simulation class which implements the simulation logic
        models.py - the havsim lane changing model (lc_havsim), IDM, default parameters
        opt.py - wrappers for calling optimization algorithms
        vehicle_orders.py - used internally in simulation update code
        update_lane_routes.py - used internally in simulation update code
        plotting.py - included plotting functions
        helper.py - miscellaneous functions
```

## The havsim model
### The deterministic havsim model (havsim.Vehicle) 
- Based on traditional traffic flow models
- 6 parameter car following model
  - acceleration from IDM - 5 parameters
  - bounded maximum deceleration - 1 parameter
- 21 parameter lane changing model
   - decision model based on MOBIL can output {stay, change left, change right} - 9 parameters
   - dynamics model outputs an acceleration which is added to the car following acceleration - 10 parameters
   - route model adjusts the lane changing state so vehicles will stay on their planned route - 2 parameters
- Default parameters given in havsim.models.default_parameters()

### The stochastic havsim model (havsim.CrashesStochasticVehicle)
- Incorporates human driving errors into the model, which can lead to rear ends and sideswipe crashes.
- 7 parameter model of human driving errors
  - human perception response time - 4 parameters
  - errors in lane changing decisions - 3 parameters
- Default parameters given in havsim.models.stochastic_default_parameters()
     

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
