
"""
Fit a trajectory using least squares.
"""
# Uses new VehicleData format
platoon = [1013]



def cf_model(self, p, state):
        s = state[0]
        v = state[1]
        vl = state[2]
        vvl = v*vl
        vs = v*s
        hi = 1+p[1]*v+p[2]*s+p[3]*vs+p[4]*vl+p[5]*vvl
        lo = p[0] + p[6]*v+p[7]*s+p[8]*vs+p[9]*vl+p[10]*vvl
