
#include "lane.h"

#include <utility>
#include "vehicle.h"

Lane::Lane(std::vector<double> time_series) : time_series_(std::move(time_series)) {}


double Lane::call_downstream(Vehicle *vehicle, int timeind, double dt) {
    return (time_series_[timeind] - vehicle->get_speed()) / dt;
}

double Lane::get_headway(Vehicle *veh, Vehicle *lead) {
    return lead->get_pos() - veh->get_pos() - lead->get_len();
}

