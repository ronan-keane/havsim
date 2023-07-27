def IDM(s, v, vl, p):
    """Intelligent Driver Model with parameters p = [c1, c2, c3, c4, c5].
    Using Treiber, Kesting notation, p = [v_0, T, s_0, a, b] =
    [free flow speed, desired time gap, jam distance, acceleration, comfortable deceleration]
    """
    s_star = p[2] + p[1]*v + (v*(v-vl))/(2*(p[3]*p[4])**.5)
    return p[3]*(1 - (v/p[0])**4 - (s_star/s)**2)
