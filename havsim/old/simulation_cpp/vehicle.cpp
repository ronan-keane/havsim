#include "vehicle.h"
#include "lane.h"
#include "model.h"
#include <cmath>

void Vehicle::set_lc(std::unordered_map<Vehicle *, std::string> &lc_actions, int timeind, int dt) {
    auto res = set_lc_helper(get_lc_parameters().back() * dt);
    if (res.call_model) {
        mobil(this, lc_actions, res, timeind, dt);
    }

}

std::pair<double, double> get_new_hd(Vehicle *lcsidefol, Vehicle *veh, Lane *lcsidelane) {
    Vehicle *lcsidelead = lcsidefol->get_lead();
    double newlcsidefolhd = -1;
    double newlcsidehd = -1;
    if (lcsidelead) {
        newlcsidehd = lcsidelane->get_headway(veh, lcsidelead);
    }
    if (!lcsidefol->get_cf_parameters().empty()) {
        newlcsidefolhd = lcsidefol->get_lane()->get_headway(lcsidefol, veh);
    }
    return {newlcsidefolhd, newlcsidehd};
}

HeadwayParams::HeadwayParams() {
    call_model = false;
    lside = false;
    rside = false;
    newlfolhd = -1.0;
    newlhd = -1.0;
    newrfolhd = -1.0;
    newrhd = -1.0;
    newfolhd = -1.0;
}

Vehicle::Vehicle(int vehid,
                 Lane *lane,
                 std::vector<double> cf_parameters,
                 std::vector<double> lc_parameters,
                 double length,
                 double relaxp,
                 double minacc,
                 double maxacc,
                 double maxspeed) :
        vehid_(vehid),
        lane_(lane),
        cf_parameters_(std::move(cf_parameters)),
        lc_parameters_(std::move(lc_parameters)),
        relaxp_(relaxp),
        length_(length),
        lead_(nullptr),
        fol_(nullptr),
        lfol_(nullptr),
        rfol_(nullptr),
        llead_(nullptr),
        rlead_(nullptr),
        pos_(-1),
        speed_(-1),
        headway_(-1),
        acc_(-1),
        inittime_(-1),
        endtime_(-1),
        in_relax_(false),
        relax_start_(-1),
        minacc_(minacc),
        maxacc_(maxacc),
        maxspeed_(maxspeed) {}

void Vehicle::initialize(double pos,
                         double speed,
                         double headway,
                         int inittime,
                         Vehicle *lead,
                         Vehicle *fol,
                         Vehicle *lfol,
                         Vehicle *rfol,
                         Vehicle *llead,
                         Vehicle *rlead) {
    pos_ = pos;
    speed_ = speed;
    headway_ = headway;
    inittime_ = inittime;
    lead_ = lead;
    fol_ = fol;
    lfol_ = lfol;
    rfol_ = rfol;
    llead_ = llead;
    rlead_ = rlead;
    leadmem_.emplace_back(lead_, inittime_);
    lanemem_.emplace_back(lane_, inittime_);
    posmem_.push_back(pos_);
    speedmem_.push_back(speed_);
}


void Vehicle::set_cf(int timeind, double dt) {
    acc_ = get_cf(headway_, speed_, lead_, lane_, timeind, dt, in_relax_);
}

double Vehicle::get_speed() const {
    return speed_;
}

double Vehicle::cfmodel(const std::vector<double> &p, const std::vector<double> &state) {
// state = headway, velocity, lead velocity
// p = parameters
// returns acceleration
    double t5 = 2 * std::sqrt(p[3] * p[4]);
    double t4 = state[1] * (state[1] - state[2]);
    double t3 = p[2] + state[1] * p[1] + t4 / t5;
    double t2 = t3 / state[0];
    double t1 = 1 - std::pow(state[1] / p[0], 4) - std::pow(t2, 2);
    return p[3] * t1;
}

HeadwayParams Vehicle::set_lc_helper(double chk_lc, bool get_fol) {
    HeadwayParams res;

    bool chk_cond;
    if (l_.empty()) {
        if (r_.empty()) {
            res.call_model = false;
            return res;
        } else if (r_ == "discretionary") {
            res.lside = false;
            res.rside = true;
            chk_cond = lcside_.empty();
        } else {
            res.lside = false;
            res.rside = true;
            chk_cond = false;
        }
    } else if (l_ == "discretionary") {
        if (r_.empty()) {
            res.lside = true;
            res.rside = false;
            chk_cond = lcside_.empty();
        } else if (r_ == "discretionary") {
            if (!lcside_.empty()) {
                chk_cond = false;
                if (lcside_ == "l") {
                    res.lside = true;
                    res.rside = false;
                } else {
                    res.lside = false;
                    res.rside = true;
                }
            } else {
                chk_cond = true;
                res.lside = true;
                res.rside = true;
            }
        } else {
            res.lside = false;
            res.rside = true;
            chk_cond = false;
        }
    } else {
        if (r_.empty()) {
            res.lside = true;
            res.rside = false;
            chk_cond = false;
        } else if (r_ == "discretionary") {
            res.lside = true;
            res.rside = false;
            chk_cond = false;
        }
    }
    if (chk_cond) {
        if (chk_lc < 1 && (double) rand() / RAND_MAX > chk_lc) {
            res.call_model = false;
            return res;
        }
    }
    if (res.lside) {
        auto hd = get_new_hd(lfol_, this, get_llane());
        res.newlfolhd = hd.first;
        res.newlhd = hd.second;
    }
    if (res.rside) {
        auto hd = get_new_hd(rfol_, this, get_rlane());
        res.newrfolhd = hd.first;
        res.newrhd = hd.second;
    }

    if (get_fol) {
        if (fol_->get_cf_parameters().empty()) {
            // newfolhd already set to -1
        } else if (!lead_) {
            // newfolhd already set to -1
        } else {
            res.newfolhd = fol_->get_lane()->get_headway(fol_, lead_);
        }
    }
    res.call_model = true;
    return res;
}

void Vehicle::update(int timeind, double dt) {
    double acc = acc_;
    acc_ = std::min(acc_, maxacc_);
    acc_ = std::max(acc_, minacc_);

    double temp = acc * dt;
    double nextspeed = speed_ + temp;
    if (nextspeed < 0) {
        nextspeed = 0;
        temp = -speed_;
    } else if (nextspeed > maxspeed_) {
        nextspeed = maxspeed_;
        temp = maxspeed_ - speed_;
    }
    pos_ += speed_ * dt + .5 * temp * dt;
    speed_ = nextspeed;

    posmem_.push_back(pos_);
    speedmem_.push_back(speed_);
    if (in_relax_ && timeind == relax_start_ + relax_.size() - 1) {
        in_relax_ = false;
        relaxmem_.push_back({{relax_start_, timeind}, relax_});
    }
}

double Vehicle::get_cf(double headway, double speed, Vehicle *lead, Lane *lane, int timeind, double dt, bool in_relax) {
    double acc;
    if (!lead) {
        acc = lane_->call_downstream(this, timeind, dt);
    } else {


        if (in_relax) {
            headway += relax_[timeind - relax_start_];
        }
        std::vector<double> state = {headway, speed, lead->get_speed()};
        acc = cfmodel(cf_parameters_, state);

    }


    return acc;
}


void Vehicle::update_headway() {
    if (lead_) {
        headway_ = lane_->get_headway(this, lead_);
    }
}



