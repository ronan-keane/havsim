

#ifndef HAVSIM_CPP_MODEL_H
#define HAVSIM_CPP_MODEL_H

#include "vehicle.h"
#include <unordered_map>
#include <string>

std::pair<double, double>
mobil_helper(Vehicle *fol, Vehicle *curlead, Vehicle *newlead, double newhd, int timeind, double dt, bool userelax_cur,
             bool userelax_new);


void
mobil(Vehicle *veh, std::unordered_map<Vehicle *, std::string> &lc_actions, HeadwayParams headway_params, int timeind,
      double dt, bool userelax_cur = true, bool userelax_new = false, bool use_coop = false, bool use_tact = false);

#endif //HAVSIM_CPP_MODEL_H
