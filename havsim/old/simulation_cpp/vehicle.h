#ifndef HAVSIM_CPP_VEHICLE_H
#define HAVSIM_CPP_VEHICLE_H

#include <vector>
#include <string>
#include <unordered_map>

class Lane;

struct HeadwayParams {
    bool call_model;
    bool lside;
    bool rside;
    double newlfolhd;
    double newlhd;
    double newrfolhd;
    double newrhd;
    double newfolhd;

    HeadwayParams();

};

class Vehicle {
public:
    Vehicle(int vehid,
            Lane *lane,
            std::vector<double> cf_parameters,
            std::vector<double> lc_parameters,
            double length = 3,
            double relaxp = 12,
            double minacc = -7,
            double maxacc = 3,
            double maxspeed = 1e4
    );

    void initialize(double pos,
                    double speed,
                    double headway,
                    int inittime,
                    Vehicle *lead = nullptr,
                    Vehicle *fol = nullptr,
                    Vehicle *lfol = nullptr,
                    Vehicle *rfol = nullptr,
                    Vehicle *llead = nullptr,
                    Vehicle *rlead = nullptr);

    // car following model: updates the action (acceleration by default)
    virtual void set_cf(int timeind, double dt);

    virtual double
    get_cf(double headway, double speed, Vehicle *lead, Lane *lane, int timeind, double dt, bool in_relax);

    virtual double cfmodel(const std::vector<double> &p, const std::vector<double> &state);

    double get_speed() const;

    double get_pos() const {
        return pos_;
    }

    double get_len() const {
        return length_;
    }

    int get_vehid() const {
        return vehid_;
    }

    void update(int timeind, double dt);

    void update_headway();

    HeadwayParams set_lc_helper(double chk_lc = 1, bool get_fol = true);

    void set_lc(std::unordered_map<Vehicle *, std::string> &lc_actions, int timeind, int dt);

    std::vector<double> get_posmem() {
        return posmem_;
    }

    Vehicle *get_lead() const {
        return lead_;
    }

    void set_l(const std::string &l) {
        l_ = l;
    };

    void set_r(const std::string &r) {
        r_ = r;
    };

    void set_lcside(const std::string &lcside) {
        lcside_ = lcside;
    }

    std::vector<double> get_cf_parameters() const {
        return cf_parameters_;
    }

    void set_headway(double headway) {
        headway_ = headway;
    }

    Lane *get_lane() const {
        return lane_;
    }

    Lane *get_llane() const {
        return llane_;
    }

    void set_pos(double pos) {
        pos_ = pos;
    }

    void set_speed(double speed) {
        speed_ = speed;
    }

    void set_llane(Lane *llane) {
        llane_ = llane;
    }

    void set_fol(Vehicle *fol) {
        fol_ = fol;
    }

    void set_lfol(Vehicle *lfol) {
        lfol_ = lfol;
    }

    void set_rfol(Vehicle *rfol) {
        rfol_ = rfol;
    }

    void set_lead(Vehicle *lead) {
        lead_ = lead;
    }

    void set_rlane(Lane *rlane) {
        rlane_ = rlane;
    }

    Lane *get_rlane() const {
        return rlane_;
    }

    bool get_in_relax() const {
        return in_relax_;
    }

    double get_headway() const {
        return headway_;
    }

    void set_in_relax(bool in_relax) {
        in_relax_ = in_relax;
    }

    void set_relax_start(int start) {
        relax_start_ = start;
    }

    void set_relax(const std::vector<double> &relax) {
        relax_ = relax;
    }

    double get_acc() const {
        return acc_;
    }

    std::vector<double> get_lc_parameters() const {
        return lc_parameters_;
    }

    Vehicle *get_fol() const {
        return fol_;
    }

    Vehicle *get_lfol() const {
        return lfol_;
    }

    Vehicle *get_rfol() const {
        return rfol_;
    }

    Vehicle *get_rlead() const {
        return rlead_;
    }

    Vehicle *get_llead() const {
        return llead_;
    }

    std::string get_r() const {
        return r_;
    }

    std::string get_l() const {
        return l_;
    }


private:
    // id of the vehicle
    int vehid_;
    Lane *lane_;
    Lane *llane_;
    Lane *rlane_;

    // ====== Model parameters =====
    std::vector<double> cf_parameters_;
    std::vector<double> lc_parameters_;
    double relaxp_;
    double length_;

    // Leader/follower relationships
    Vehicle *lead_;
    Vehicle *fol_;
    Vehicle *lfol_;
    Vehicle *rfol_;
    Vehicle *llead_;
    Vehicle *rlead_;

    // State
    double pos_;
    double speed_;
    double headway_;
    double acc_;

    // Memory
    int inittime_;
    int endtime_;
    std::vector<std::pair<Vehicle *, int>> leadmem_;
    std::vector<std::pair<Lane *, int>> lanemem_;
    std::vector<double> posmem_;
    std::vector<double> speedmem_;
    std::vector<std::pair<std::pair<int, int>, std::vector<double>>> relaxmem_;

    bool in_relax_;
    std::vector<double> relax_;
    int relax_start_;

    std::string l_;
    std::string r_;
    std::string lcside_;

    double maxacc_;
    double minacc_;
    double maxspeed_;

};

#endif //HAVSIM_CPP_VEHICLE_H
