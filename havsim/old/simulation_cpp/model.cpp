
#include "model.h"
#include <limits>

std::pair<double, double>
mobil_helper(Vehicle *fol, Vehicle *curlead, Vehicle *newlead, double newhd, int timeind, double dt, bool userelax_cur,
             bool userelax_new) {
    double fola;
    double newfola;
    if (fol->get_cf_parameters().empty()) {
        fola = 0;
        newfola = 0;
    } else {
        if (!userelax_cur && fol->get_in_relax()) {
            fola = fol->get_cf(fol->get_headway(), fol->get_speed(), curlead, fol->get_lane(), timeind, dt, false);
        } else {
            fola = fol->get_acc();
        }
        bool userelax = userelax_new && fol->get_in_relax();
        newfola = fol->get_cf(newhd, fol->get_speed(), newlead, fol->get_lane(), timeind, dt, userelax);
    }
    return {fola, newfola};
}

void
mobil(Vehicle *veh, std::unordered_map<Vehicle *, std::string> &lc_actions, HeadwayParams headway_params, int timeind,
      double dt, bool userelax_cur, bool userelax_new, bool use_coop, bool use_tact) {

    auto p = veh->get_lc_parameters();
    double lincentive = std::numeric_limits<double>::lowest();
    double rincentive = std::numeric_limits<double>::lowest();
    double cura, fola, newfola;
    if (!userelax_cur && veh->get_in_relax()) {
        cura = veh->get_cf(veh->get_headway(), veh->get_speed(), veh->get_lead(), veh->get_lane(), timeind, dt, false);
    } else {
        cura = veh->get_acc();
    }
    auto pair_a = mobil_helper(veh->get_fol(), veh, veh->get_lead(), headway_params.newfolhd, timeind, dt, userelax_cur,
                               userelax_new);
    fola = pair_a.first;
    newfola = pair_a.second;

    double lfola, newlfola, newla;
    if (headway_params.lside) {
        auto pair = mobil_helper(veh->get_lfol(), veh->get_lfol()->get_lead(), veh, headway_params.newlfolhd, timeind,
                                 dt, userelax_cur, userelax_new);
        lfola = pair.first;
        newlfola = pair.second;
        bool userelax = userelax_new && veh->get_in_relax();
        newla = veh->get_cf(headway_params.newlhd, veh->get_speed(), veh->get_lfol()->get_lead(), veh->get_llane(),
                            timeind, dt, userelax);
        lincentive = newla - cura + p[2] * (newlfola - lfola + newfola - fola) + p[3];
    }

    double rfola, newrfola, newra;
    if (headway_params.rside) {
        Vehicle *rfol = veh->get_rfol();
        Vehicle *rlead = rfol->get_lead();
        auto pair = mobil_helper(rfol, rlead, veh, headway_params.newrfolhd, timeind, dt, userelax_cur, userelax_new);
        rfola = pair.first;
        newrfola = pair.second;
        bool userelax = userelax_new && veh->get_in_relax();
        newra = veh->get_cf(headway_params.newrhd, veh->get_speed(), rlead, veh->get_rlane(), timeind, dt, userelax);

        rincentive = newra - cura + p[2] * (newrfola - rfola + newfola - fola) + p[4];
    }
    double incentive;
    std::string side;
    std::string lctype;
    double selfsafe;
    double folsafe;
    double newhd;
    double newlcsidehd;
    if (rincentive > lincentive) {
        side = "r";
        lctype = veh->get_r();
        incentive = rincentive;
        newhd = headway_params.newrhd;
        newlcsidehd = headway_params.newrfolhd;
        selfsafe = newra;
        folsafe = newrfola;
    } else {
        side = "l";
        lctype = veh->get_l();
        incentive = lincentive;
        newhd = headway_params.newlhd;
        newlcsidehd = headway_params.newlfolhd;
        selfsafe = newla;
        folsafe = newlfola;
    }
    if (lctype == "discretionary") {
        if (incentive > p[1]) {
            if (selfsafe > p[0] && folsafe > p[0]) {
                lc_actions[veh] = side;
            }
        }
    } else {
        if (selfsafe > p[0] && folsafe > p[0]) {
            lc_actions[veh] = side;
        }
    }

}