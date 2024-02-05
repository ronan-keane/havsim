
#ifndef HAVSIM_CPP_LANE_H
#define HAVSIM_CPP_LANE_H

#include <vector>

class Vehicle;

class Lane {
public:
    explicit Lane(std::vector<double> time_series);


    double call_downstream(Vehicle *vehicle, int timeind, double dt);

    double get_headway(Vehicle *veh, Vehicle *lead);

private:
    std::vector<double> time_series_;


};

#endif //HAVSIM_CPP_LANE_H
