#include "vehicle.h"
#include "lane.h"
#include <iostream>
#include <cassert>
#include <fstream>

using namespace std;

void set_state(Vehicle *veh, double pos, double speed) {
    veh->set_pos(pos);
    veh->set_speed(speed);
}

void verify_output(const string &expected_output_path, const string &output) {
    ifstream test_file(expected_output_path);
    string line;
    if (test_file.is_open()) {
        getline(test_file, line);
        assert(line == output);
    } else {
        cerr << "Unable to open file";
    }
}

vector<vector<vector<double>>> read_input(const string &input_path) {

    const int x = 2000;
    const int y = 7;
    const int z = 2;
    vector<vector<vector<double>>> res(x, vector<vector<double>>(y, vector<double>(x, 0.0)));
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    ifstream test_file(input_path);
    string line;
    if (test_file.is_open()) {
        getline(test_file, line);
        size_t pos;
        while ((pos = line.find(',')) != string::npos) {

            res[i][j][k] = stod(line.substr(0, pos));

            line.erase(0, pos + 1);
            k++;
            if (k == z) {
                k = 0;
                j++;
                if (j == y) {
                    j = 0;
                    i++;
                }
            }
        }
        res[i][j][k] = stod(line);
        test_file.close();
    } else {
        cerr << "Unable to open file";
    }
    return res;

}

char maketest(vector<Vehicle *> &vehlist, vector<Vehicle *> &vehlist2, vector<vector<double>> &data, Vehicle *testveh) {
    for (int i = 0; i < vehlist.size(); i++) {
        set_state(vehlist[i], data[i][0], data[i][1]);
    }

    for (auto veh:vehlist2) {
        veh->set_headway(veh->get_lane()->get_headway(veh, veh->get_lead()));
        veh->set_cf(0, 1);
    }
    unordered_map<Vehicle *, std::string> lc_actions;
    testveh->set_lc(lc_actions, 0, 1);
    if (lc_actions.count(testveh)) {
        return lc_actions[testveh][0];
    }

    return 'n';
}

int main() {
    Lane lane0({});
    Lane lane1({});
    Lane lane2({});
    Vehicle testveh(0, &lane1, {27, 1.2, 2, 1.1, 1.5}, {-2, .2, .2, 0, .1, 1});
    Vehicle lead(0, &lane1, {28, 1.2, 2, 1.1, 1.5}, {-1.5, .2, .2, 0, .1, 1});
    Vehicle fol(0, &lane1, {25, 1.2, 2, 1.1, 1.5}, {-1.5, .2, .2, 0, .1, 1});
    Vehicle lfol(0, &lane0, {20, 1.2, 1.1, 1.1, 1.5}, {-1.5, .2, .2, 0, .1, 1});
    Vehicle llead(0, &lane0, {23, 1.2, 2, 1.1, 1.5}, {-1.5, .2, .2, 0, .1, 1});
    Vehicle rfol(0, &lane2, {33, .9, 1.5, 1.1, 1.5}, {-1.5, .2, .2, 0, .1, 1});
    Vehicle rlead(0, &lane2, {27, 1.2, 2, 1.1, 1.5}, {-1.5, .2, .2, 0, .1, 1});
    testveh.set_llane(&lane0);
    testveh.set_rlane(&lane2);
    testveh.set_lead(&lead);
    testveh.set_fol(&fol);
    testveh.set_lfol(&lfol);
    testveh.set_rfol(&rfol);
    lead.set_fol(&testveh);
    fol.set_lead(&testveh);
    rfol.set_lead(&rlead);
    lfol.set_lead(&llead);

    // test1
    auto test1_input = read_input("../test1_input");
    testveh.set_l("discretionary");
    testveh.set_r("discretionary");
    vector<Vehicle *> vehlist = {&testveh, &llead, &lead, &rlead, &lfol, &fol, &rfol};
    vector<Vehicle *> vehlist2 = {&testveh, &lfol, &fol, &rfol};
    string test1_output;
    for (auto &i:test1_input) {
        test1_output += maketest(vehlist, vehlist2, i, &testveh);
    }
    verify_output("../test1_output", test1_output);


    // test2
    auto test2_input = read_input("../test2_input");
    testveh.set_l("discretionary");
    testveh.set_r("mandatory");

    string test2_output;
    for (auto &i:test2_input) {
        test2_output += maketest(vehlist, vehlist2, i, &testveh);
    }
    verify_output("../test2_output", test2_output);

    // test3
    auto test3_input = read_input("../test3_input");
    testveh.set_l("discretionary");
    testveh.set_r("");

    string test3_output;
    for (auto &i:test3_input) {
        test3_output += maketest(vehlist, vehlist2, i, &testveh);
    }
    verify_output("../test3_output", test3_output);

    // test4
    // same input as test1
    auto test4_input = read_input("../test1_input");
    testveh.set_in_relax(true);
    testveh.set_relax({20});
    testveh.set_relax_start(0);
    rfol.set_in_relax(true);
    rfol.set_relax({-30});
    rfol.set_relax_start(0);
    testveh.set_l("discretionary");
    testveh.set_r("discretionary");
    string test4_output;
    for (auto &i:test4_input) {
        test4_output += maketest(vehlist, vehlist2, i, &testveh);
    }
    verify_output("../test4_output", test4_output);


}
