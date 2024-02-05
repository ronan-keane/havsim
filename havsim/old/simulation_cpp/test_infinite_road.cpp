#include "vehicle.h"
#include "lane.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <cassert>

using namespace std;

double IDM_eql(vector<double> &p, double v) {
    return sqrt(pow(p[2] + p[1] * v, 2.0) / (1 - pow(v / p[0], 4)));
}


int main() {


    int simlen = 2000;
    const double dt = .25;
    int nveh = 100;
    vector<double> time_series(simlen, 30.0);
    for (int i = 0; i < 200; i++) {
        time_series[i] = 1.0 / 10 * pow(.1 * i - 10.0, 2.0) + 20;
    }
    for (int i = 0; i < simlen; i++) {
        time_series[i] -= 10;
    }

    Lane lane(time_series);
    // cf parameters
    vector<double> p = {33.33, 1.1, 2.0, .9, 1.5};
    // lc parameters (not used)
    vector<double> lcp;
    const double initspeed = time_series[0];
    const double car_len = 2;
    const double eql_hd = IDM_eql(p, initspeed);

    vector<Vehicle *> vehicles;
    double curpos = 0.0;
    auto v1 = new Vehicle(-1, &lane, p, lcp, 2);
    v1->initialize(curpos, initspeed, -1.0, 0.0);
    vehicles.push_back(v1);
    for (int i = 0; i < nveh - 1; i++) {
        curpos -= eql_hd + car_len;
        auto v_new = new Vehicle(i, &lane, p, lcp, car_len);
        v_new->initialize(curpos, initspeed, eql_hd, 0.0, vehicles.back());
        vehicles.push_back(v_new);
    }

    for (int i = 0; i < simlen; i++) {
        for (auto v:vehicles) {
            v->set_cf(i, dt);
        }
        for (auto v:vehicles) {
            v->update(i, dt);
        }
        for (auto v:vehicles) {
            v->update_headway();
        }
    }
    unordered_map<int, vector<double>> mp;
    for (Vehicle *v:vehicles) {
        mp[v->get_vehid()] = v->get_posmem();
    }
    string line;
    // Current folder is cmake-build-debug
    // Will jump to the source folder (parent folder)
    ifstream test_file("../test_infinite_road_result");
    bool id_line = true;
    int vehid_cur;
    if (test_file.is_open()) {
        while (getline(test_file, line)) {
            if (!line.empty()) {
                if (id_line) {
                    vehid_cur = stoi(line);
                } else {
                    auto &expected_pos = mp[vehid_cur];
                    vector<double> actual_pos;
                    size_t pos = 0;
                    std::string token;
                    while ((pos = line.find(',')) != string::npos) {
                        actual_pos.push_back(stod(line.substr(0, pos)));
                        line.erase(0, pos + 1);
                    }
                    actual_pos.push_back(stod(line));
                    assert(actual_pos.size() == expected_pos.size());
                    for (int i = 0; i < actual_pos.size(); i++) {
                        assert(fabs(actual_pos[i] - expected_pos[i]) < 0.01);
                    }
                }
                id_line = !id_line;
            }
        }
        test_file.close();
    } else {
        cerr << "Unable to open file";
    }

    for (auto v:vehicles) {
        delete v;
    }
    return 0;


}
