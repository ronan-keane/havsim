"""Configure simulations."""
import havsim as hs
import numpy as np


def e94(times=None, gamma_parameters=None, xi_parameters=None):
    """Simulation of 12km length of E94 in Ann Arbor area"""
    # veh parameters
    def make_parameters(truck_prob=0.):
        def p_f(timeind):
            kwargs = hs.models.default_parameters(truck_prob=truck_prob)
            kwargs['gamma_parameters'] = gamma_parameters
            kwargs['xi_parameters'] = xi_parameters
            return kwargs
        return p_f

    # road network
    main_road = hs.Road(num_lanes=2, length=12000, name='E94')
    main_road.connect('exit', is_exit=True)
    offramp1 = hs.Road(num_lanes=1, length=[(475, 675)], name='jackson off ramp')
    main_road.merge(offramp1, self_index=1, new_lane_index=0, self_pos=(475, 660), new_lane_pos=(475, 660))
    offramp1.connect('jackson offramp exit', is_exit=True)
    onramp1 = hs.Road(num_lanes=1, length=[(1000, 1350)], name='jackson on ramp')
    onramp1.merge(main_road, self_index=0, new_lane_index=1, self_pos=(1100, 1350), new_lane_pos=(1100, 1350))
    offramp2 = hs.Road(num_lanes=1, length=[(5330, 5530)], name='ann arbor saline off ramp')
    main_road.merge(offramp2, self_index=1, new_lane_index=0, self_pos=(5330, 5480), new_lane_pos=(5330, 5480))
    offramp2.connect('saline offramp exit', is_exit=True)
    onramp2 = hs.Road(num_lanes=1, length=[(5950, 6330)], name='ann arbor saline on ramp SW')
    onramp2.merge(main_road, self_index=0, new_lane_index=1, self_pos=(6050, 6330), new_lane_pos=(6050, 6330))
    onramp3 = hs.Road(num_lanes=1, length=[(6410, 6810)], name='ann arbor saline on ramp NE')
    onramp3.merge(main_road, self_index=0, new_lane_index=1, self_pos=(6610, 6810), new_lane_pos=(6610, 6810))
    offramp3 = hs.Road(num_lanes=1, length=[(7810, 7990)], name='state off ramp')
    main_road.merge(offramp3, self_index=1, new_lane_index=0, self_pos=(7810, 7940), new_lane_pos=(7810, 7940))
    offramp3.connect('state offramp exit', is_exit=True)
    onramp4 = hs.Road(num_lanes=1, length=[(8310, 8710)], name='state on ramp S')
    onramp4.merge(main_road, self_index=0, new_lane_index=1, self_pos=(8410, 8710), new_lane_pos=(8410, 8710))
    onramp5 = hs.Road(num_lanes=1, length=[(8830, 9230)], name='state on ramp N')
    onramp5.merge(main_road, self_index=0, new_lane_index=1, self_pos=(8980, 9230), new_lane_pos=(8980, 9230))

    # downstream boundary conditions
    main_road.set_downstream({'method': 'free'})
    offramp1.set_downstream({'method': 'free'})
    offramp2.set_downstream({'method': 'free'})
    offramp3.set_downstream({'method': 'free'})
    onramp1.set_downstream({'method': 'free merge', 'self_lane': onramp1[0], 'minacc': -2})
    onramp2.set_downstream({'method': 'free merge', 'self_lane': onramp2[0], 'minacc': -2})
    onramp3.set_downstream({'method': 'free merge', 'self_lane': onramp3[0], 'minacc': -2})
    onramp4.set_downstream({'method': 'free merge', 'self_lane': onramp4[0], 'minacc': -2})
    onramp5.set_downstream({'method': 'free merge', 'self_lane': onramp5[0], 'minacc': -2})

    # upstream boundary conditions
    dt = .2
    init_timeind = int(times[0]*3600/dt)
    timesteps = int((times[1]-times[0])*3600/dt)
    interval = int(3600/dt*.25)
    # inflow amounts
    upstream_flows = \
        [142, 122, 126, 114, 120, 128, 156, 132, 106, 130, 130, 130, 133, 193, 121, 177, 308, 296, 292, 308, 576, 704,
         704, 708, 1373, 1229, 1397, 1345, 2121, 2109, 2177, 2037, 2468, 2268, 2772, 2404, 1412, 1580, 1532, 1332, 1216,
         1344, 1164, 1404, 1249, 1205, 1273, 1153, 1173, 1225, 1229, 1333, 1322, 1398, 1366, 1522, 1489, 1249, 1197,
         1445, 1496, 1458, 1630, 1762, 1662, 1620, 1772, 1920, 1669, 2117, 2337, 1549, 1395, 1643, 1499, 1311, 771,
         783, 907, 919, 716, 640, 692, 712, 501, 565, 585, 685, 302, 442, 522, 506, 226, 230, 210, 282, 142]
    onramp1_flows = \
        [88, 76, 80, 80, 36, 36, 12, 12, 32, 16, 16, 4, 24, 40, 56, 16, 24, 44, 52, 40, 56, 48, 136, 136, 152, 272, 312,
         360, 360, 556, 552, 668, 660, 668, 508, 600, 512, 444, 448, 540, 540, 488, 536, 532, 528, 580, 560, 564, 608,
         620, 536, 516, 608, 500, 584, 564, 620, 672, 616, 620, 708, 676, 580, 620, 616, 568, 576, 356, 536, 388, 380,
         412, 672, 524, 620, 576, 520, 508, 460, 440, 432, 368, 368, 344, 320, 276, 248, 184, 204, 152, 172, 128, 128,
         156, 132, 112, 88]
    onramp2_flows = \
        [40, 32, 20, 40, 16, 16, 20, 12, 8, 16, 12, 20, 12, 8, 16, 8, 16, 24, 16, 8, 36, 32, 60, 56, 72, 100, 116, 136,
         176, 204, 204, 260, 252, 228, 184, 236, 248, 176, 172, 168, 172, 188, 200, 176, 164, 224, 132, 168, 264, 196,
         240, 248, 208, 184, 192, 132, 292, 328, 292, 248, 260, 300, 312, 260, 292, 352, 192, 208, 204, 200, 112, 204,
         188, 224, 176, 236, 220, 180, 168, 196, 212, 224, 140, 208, 156, 216, 140, 164, 116, 100, 60, 92, 96, 68, 84,
         44, 40]
    onramp3_flows = \
        [40, 52, 32, 28, 12, 16, 24, 24, 0, 8, 0, 4, 8, 8, 16, 0, 16, 24, 84, 72, 96, 108, 108, 160, 212, 320, 392, 428,
         548, 688, 648, 604, 404, 544, 460, 420, 388, 380, 376, 480, 312, 332, 384, 292, 328, 356, 412, 376, 432, 488,
         416, 364, 420, 436, 388, 396, 316, 432, 480, 352, 400, 472, 444, 344, 428, 412, 340, 476, 468, 428, 408, 600,
         504, 284, 296, 332, 412, 412, 328, 300, 216, 260, 248, 208, 248, 232, 208, 232, 200, 136, 80, 100, 108, 72, 76,
         48, 40]
    onramp4_flows = \
        [80, 76, 60, 68, 56, 64, 60, 60, 24, 20, 20, 24, 28, 20, 16, 12, 28, 16, 48, 40, 8, 88, 60, 92, 100, 88, 168,
         152, 256, 200, 208, 248, 288, 312, 332, 264, 312, 268, 324, 276, 380, 364, 332, 376, 416, 332, 420, 456, 576,
         592, 464, 536, 544, 608, 488, 516, 508, 560, 616, 564, 748, 868, 916, 792, 1084, 904, 888, 936, 972, 988, 940,
         800, 756, 684, 640, 520, 564, 504, 500, 432, 448, 348, 248, 328, 368, 244, 192, 192, 248, 224, 200, 192, 208,
         152, 140, 120, 80]
    onramp5_flows = \
        [28, 44, 32, 16, 12, 8, 20, 8, 16, 20, 12, 32, 16, 12, 24, 20, 36, 16, 24, 48, 44, 40, 52, 104, 164, 140, 220,
         212, 176, 236, 340, 300, 396, 348, 352, 344, 280, 324, 292, 240, 292, 324, 320, 376, 348, 308, 336, 476, 432,
         348, 440, 384, 440, 324, 452, 384, 420, 448, 532, 540, 648, 644, 744, 604, 952, 736, 1056, 1108, 920, 928, 864,
         716, 612, 532, 432, 268, 248, 300, 280, 276, 332, 240, 204, 196, 168, 180, 128, 88, 104, 68, 60, 72, 68, 72,
         44, 24, 28]

    def make_inflow(flow):
        def inflow(timeind):
            return flow[timeind // interval]/3600
        return inflow

    main_inflow = make_inflow(list(np.array(upstream_flows)/2))
    onramp1_inflow = make_inflow(onramp1_flows)
    onramp2_inflow = make_inflow(onramp2_flows)
    onramp3_inflow = make_inflow(onramp3_flows)
    onramp4_inflow = make_inflow(onramp4_flows)
    onramp5_inflow = make_inflow(onramp5_flows)

    # OD pairs definition
    main_routes = [['jackson off ramp', 'jackson offramp exit'], ['ann arbor saline off ramp', 'saline offramp exit'],
                   ['state off ramp', 'state offramp exit'], ['exit']]
    onramp1_routes = [['E94', 'ann arbor saline off ramp', 'saline offramp exit'],
                      ['E94', 'state off ramp', 'state offramp exit'], ['E94', 'exit']]
    onramp2_routes = [['E94', 'exit']]
    onramp3_routes = [['E94', 'exit']]
    onramp4_routes = [['E94', 'exit']]
    onramp5_routes = [['E94', 'exit']]
    main_od = \
        np.stack([
            [0.450704225352113, 0.295081967213115, 0.0952380952380952, 0.245614035087719, 0.0666666666666667, 0.21875,
             0.256410256410256, 0.151515151515152, 0.0377358490566038, 0.123076923076923, 0.0923076923076923,
             0.0615384615384615, 0.0300751879699248, 0.352331606217617, 0.165289256198347, 0.15819209039548,
             0.0779220779220779, 0.135135135135135, 0.205479452054795, 0.207792207792208, 0.0833333333333333,
             0.210227272727273, 0.329545454545455, 0.361581920903955, 0.227239621267298, 0.335231895850285,
             0.443808160343593, 0.389591078066915, 0.243281471004243, 0.386913229018492, 0.376665135507579,
             0.325969563082965, 0.247974068071313, 0.248677248677249, 0.248196248196248, 0.171381031613977,
             0.243626062322946, 0.225316455696203, 0.198433420365535, 0.255255255255255, 0.236842105263158, 0.25,
             0.219931271477663, 0.236467236467236, 0.214571657325861, 0.315352697095436, 0.24509033778476,
             0.287944492627927, 0.238704177323103, 0.303673469387755, 0.231082180634662, 0.228057014253563,
             0.229954614220877, 0.180257510729614, 0.228404099560761, 0.202365308804205, 0.22834116856951,
             0.243394715772618, 0.227234753550543, 0.226989619377163, 0.168449197860963, 0.219178082191781,
             0.169325153374233, 0.163402350307778, 0.16364699006429, 0.206706981858164, 0.158119658119658,
             0.195833333333333, 0.215698022768125, 0.207841284837034, 0.176294394522893, 0.198837959974177,
             0.240860215053763, 0.180158247108947, 0.216144096064043, 0.158657513348589, 0.264591439688716,
             0.229885057471264, 0.229327453142227, 0.182807399347116, 0.268156424581006, 0.21875, 0.190751445086705,
             0.168539325842697, 0.111776447105788, 0.276106194690265, 0.136752136752137, 0.169343065693431,
             0.291390728476821, 0.226244343891403, 0.0996168582375479, 0.189723320158103, 0.353982300884956,
             0.260869565217391, 0.285714285714286, 0.156028368794326, .045070],
            [0.0661802138129985, 0.0870269176280105, 0.0932744231713304, 0.0363559501162545, 0.100900900900901,
             0.0459558823529412, 0.0697115384615385, 0, 0.0287243030132357, 0.10792899408284, 0.0541905855338691,
             0.0893772893772894, 0.0253575114256229, 0.0471031559114461, 0.106332578828236, 0.0204074644752611,
             0.0119750379490639, 0.0345945945945946, 0.134285163032992, 0.0781049935979514, 0.0125570776255708,
             0.0470725466586394, 0.07498504784689, 0.104231523117722, 0.0815443763213404, 0.0854614642616714,
             0.116447878275171, 0.194338488269975, 0.110908364555103, 0.112736345123695, 0.117548743434925,
             0.166443576779121, 0.123145740840469, 0.121630664632351, 0.138062726025689, 0.156005423060332,
             0.185742460644745, 0.183917675985794, 0.189391626214348, 0.171116285998792, 0.160117596443425,
             0.186497326203209, 0.207442099551646, 0.203735675057371, 0.174886628985097, 0.155933905287872,
             0.19852982569763, 0.135727547975615, 0.239394822320783, 0.204217964171412, 0.213905564874085,
             0.215844252325217, 0.178067075678321, 0.193232129913068, 0.160160321214494, 0.192006551086502,
             0.146567194664016, 0.157216682436859, 0.136399835843123, 0.131727733255245, 0.13120781099325,
             0.13340257004732, 0.146033840668439, 0.136070728838829, 0.150355597291813, 0.14832330921001,
             0.189344835255616, 0.247175438596491, 0.258458320952293, 0.300751783383906, 0.327338106124525,
             0.265599998750243, 0.194718697005274, 0.175273490730316, 0.125766295450449, 0.134294095546503,
             0.181315082027069, 0.174679536919209, 0.135649007384802, 0.161929012387977, 0.153105350506066,
             0.151209677419355, 0.11510863065577, 0.149236531259003, 0.171839327880187, 0.118359279116336,
             0.0917129203981794, 0.141200647531528, 0.115276149434393, 0.10024364775496, 0.162686050536518,
             0.0963897909105603, 0.132032814417673, 0.0997599359829288, 0.0506585612968592, 0.0482269503546099,
             .06618021],
            [0.154804935809548, 0.217612070216161, 0.123050990307627, 0.113336438441236, 0.129166666666667, 0.096875,
             0.166025641025641, 0.0840909090909091, 0.0869565217391304, 0.19041248606466, 0, 0, 0.0258041553588112,
             0.0158199711806727, 0.0442075407110372, 0.122755992377896, 0.0482374768089054, 0.129182754182754,
             0.196836268754077, 0.228818800247372, 0.134298493408663, 0.111228813559322, 0.18361581920904,
             0.360592422356283, 0.170930876546908, 0.224289911268998, 0.260865791404204, 0.407697801287674,
             0.280548671276957, 0.322091968555065, 0.379211157806893, 0.460636901209343, 0.361320555281516,
             0.389011578866651, 0.251964782399565, 0.303407364537365, 0.385448887621574, 0.29517768606487,
             0.327199181859213, 0.306805711619716, 0.251824817518248, 0.274149866759356, 0.264027892743372,
             0.264007597340931, 0.236193278947482, 0.199010628886148, 0.277351978844516, 0.383107131415891,
             0.251727640016309, 0.269827861579414, 0.282664591219443, 0.317573306370071, 0.33966027527917,
             0.346409978655039, 0.272384097596116, 0.250082915954709, 0.188902515227625, 0.300736424514428,
             0.384965353118219, 0.302581847218525, 0.248189616755793, 0.138277342900631, 0.13955351056578,
             0.091618479139464, 0.10077645346578, 0.0939882180155237, 0.0925519180000214, 0.104676427678199,
             0.0741202233060201, 0.0481739655011577, 0.058948300877787, 0.0931498751623667, 0.107941993077855,
             0.121716243318078, 0.208899245951739, 0.149251349698146, 0.266304109782511, 0.258962639057223,
             0.189757738456617, 0.215321091078983, 0.170315566963612, 0.198141891891892, 0.175285111701297,
             0.151457637412694, 0.268602853059796, 0.165307307602299, 0.103535163280511, 0.138824824667386,
             0.238887894517034, 0.111749758808582, 0.113676895286091, 0.134102481928569, 0.345258566455563,
             0.186358953224809, 0.183254344391785, 0.0937783603912473, .154804935]
        ], axis=1)
    last_column = 1 - np.sum(main_od, axis=1, keepdims=True)
    main_od = np.concatenate([main_od, last_column], axis=1)
    onramp1_od = \
        np.stack([
            [0.120481927710843, 0.123456790123457, 0.103092783505155, 0.0481927710843374, 0.108108108108108,
             0.0588235294117647, 0.09375, 0, 0.0298507462686567, 0.123076923076923, 0.0597014925373134,
             0.0952380952380952, 0.0261437908496732, 0.0727272727272727, 0.127388535031847, 0.0242424242424242,
             0.012987012987013, 0.04, 0.169014084507042, 0.0985915492957746, 0.0136986301369863, 0.0596026490066225,
             0.111842105263158, 0.163265306122449, 0.105523495465787, 0.128558310376492, 0.209366391184573,
             0.318374259102456, 0.146564885496183, 0.18388318009735, 0.188580408590885, 0.246937775600196,
             0.163751987281399, 0.161888701517707, 0.183641975308642, 0.188271604938272, 0.245569620253165,
             0.237410071942446, 0.236276849642005, 0.22976501305483, 0.209809264305177, 0.248663101604278,
             0.265927977839335, 0.266832917705736, 0.222664015904573, 0.227758007117438, 0.262984878369494,
             0.190613718411552, 0.314457028647568, 0.293279022403259, 0.278190411883862, 0.279611650485437,
             0.231242312423124, 0.235722964763062, 0.207570207570208, 0.240719910011249, 0.189937817976258,
             0.207792207792208, 0.176508760545101, 0.170408750719632, 0.157786885245902, 0.170848905499199,
             0.175801447776629, 0.162647754137116, 0.179775280898876, 0.186971655892591, 0.224907063197026,
             0.307368421052632, 0.329539295392954, 0.379661016949153, 0.397396963123644, 0.331518451300665,
             0.256499133448873, 0.213789417423838, 0.160445682451253, 0.159618820726623, 0.24655013799448,
             0.226822682268227, 0.176013805004314, 0.198152812762385, 0.209205020920502, 0.193548387096774,
             0.142241379310345, 0.179487179487179, 0.193464052287582, 0.163503649635036, 0.106241699867198,
             0.169986719787517, 0.162679425837321, 0.129554655870445, 0.180685358255452, 0.118959107806691,
             0.204379562043796, 0.134969325153374, 0.0709219858156028, 0.0571428571428571, .12048192],
            [0.15929203539823, 0.176991150442478, 0.106194690265487, 0.0884955752212389, 0.125, 0.1, 0.175, 0.075,
             0.0869565217391304, 0.202898550724638, 0, 0, 0.0236686390532544, 0.0236686390532544, 0.0473372781065089,
             0.142011834319527, 0.0476190476190476, 0.130952380952381, 0.202380952380952, 0.238095238095238,
             0.11864406779661, 0.11864406779661, 0.225988700564972, 0.446327683615819, 0.192841490138787,
             0.251278305332359, 0.344777209642075, 0.543462381300219, 0.291545189504373, 0.368179925031237,
             0.446480633069554, 0.541441066222407, 0.42463768115942, 0.421739130434783, 0.310144927536232,
             0.33768115942029, 0.413566739606127, 0.336980306345733, 0.363238512035011, 0.317286652078775,
             0.277372262773723, 0.318734793187348, 0.277372262773723, 0.333333333333333, 0.259459459459459,
             0.221021021021021, 0.31951951951952, 0.401201201201201, 0.264347826086957, 0.299130434782609,
             0.292173913043478, 0.350144927536232, 0.35356762513312, 0.351437699680511, 0.287539936102236,
             0.289669861554846, 0.197942185203332, 0.27437530622244, 0.323370896619304, 0.307692307692308,
             0.232638888888889, 0.130208333333333, 0.131944444444444, 0.0972222222222222, 0.0967069612338474,
             0.0933722384326803, 0.0950395998332639, 0.098374322634431, 0.0900995285489785, 0.0670508119434259,
             0.0900995285489785, 0.0963855421686747, 0.115209701869631, 0.13744315310763, 0.230419403739262,
             0.139464375947448, 0.220537560303239, 0.215024121295658, 0.173673328738801, 0.195727084769125,
             0.162162162162162, 0.155405405405405, 0.148648648648649, 0.128378378378378, 0.191968658178257,
             0.125367286973555, 0.0783545543584721, 0.113614103819785, 0.156156156156156, 0.0960960960960961,
             0.12012012012012, 0.126126126126126, 0.265402843601896, 0.161137440758294, 0.132701421800948,
             0.0853080568720379, .15929203]
        ], axis=1)
    last_column = 1 - np.sum(onramp1_od, axis=1, keepdims=True)
    onramp1_od = np.concatenate([onramp1_od, last_column], axis=1)

    vehicle = hs.vehicles.CrashesStochasticVehicle
    main_newveh = hs.road.make_newveh(make_parameters(.08), vehicle, main_routes, main_od, interval)
    onramp1_newveh = hs.road.make_newveh(make_parameters(), vehicle, onramp1_routes, onramp1_od, interval)
    onramp2_newveh = hs.road.make_newveh(make_parameters(), vehicle, onramp2_routes, None, interval)
    onramp3_newveh = hs.road.make_newveh(make_parameters(), vehicle, onramp3_routes, None, interval)
    onramp4_newveh = hs.road.make_newveh(make_parameters(), vehicle, onramp4_routes, None, interval)
    onramp5_newveh = hs.road.make_newveh(make_parameters(), vehicle, onramp5_routes, None, interval)

    # define set_upstream method
    # deterministic inflow
    # main_get_inflow = {'time_series': main_inflow}
    # onramp1_get_inflow = {'time_series': onramp1_inflow}
    # onramp2_get_inflow = {'time_series': onramp2_inflow}
    # onramp3_get_inflow = {'time_series': onramp3_inflow}
    # onramp4_get_inflow = {'time_series': onramp4_inflow}
    # onramp5_get_inflow = {'time_series': onramp5_inflow}
    # stochastic inflow
    main_get_inflow = {'inflow_type': 'stochastic',
                       'args': (hs.road.M3Arrivals(main_inflow, 1.2, .95), .2, init_timeind)}
    onramp1_get_inflow = {'inflow_type': 'stochastic',
                          'args': (hs.road.M3Arrivals(onramp1_inflow, 1.2, .95), .2, init_timeind)}
    onramp2_get_inflow = {'inflow_type': 'stochastic',
                          'args': (hs.road.M3Arrivals(onramp2_inflow, 1.2, .95), .2, init_timeind)}
    onramp3_get_inflow = {'inflow_type': 'stochastic',
                          'args': (hs.road.M3Arrivals(onramp3_inflow, 1.2, .95), .2, init_timeind)}
    onramp4_get_inflow = {'inflow_type': 'stochastic',
                          'args': (hs.road.M3Arrivals(onramp4_inflow, 1.2, .95), .2, init_timeind)}
    onramp5_get_inflow = {'inflow_type': 'stochastic',
                          'args': (hs.road.M3Arrivals(onramp5_inflow, 1.2, .95), .2, init_timeind)}

    increment_inflow = {'boundary_type': 'seql', 'kwargs': {'c': .9}}
    increment_inflow2 = {'boundary_type': 'heql', 'kwargs': {}}

    main_road.set_upstream(increment_inflow=increment_inflow,  get_inflow=main_get_inflow, new_vehicle=main_newveh)
    onramp1.set_upstream(increment_inflow=increment_inflow2, get_inflow=onramp1_get_inflow, new_vehicle=onramp1_newveh)
    onramp2.set_upstream(increment_inflow=increment_inflow2, get_inflow=onramp2_get_inflow, new_vehicle=onramp2_newveh)
    onramp3.set_upstream(increment_inflow=increment_inflow2, get_inflow=onramp3_get_inflow, new_vehicle=onramp3_newveh)
    onramp4.set_upstream(increment_inflow=increment_inflow2, get_inflow=onramp4_get_inflow, new_vehicle=onramp4_newveh)
    onramp5.set_upstream(increment_inflow=increment_inflow2, get_inflow=onramp5_get_inflow, new_vehicle=onramp5_newveh)

    simulation = hs.simulation.CrashesSimulation(
        roads=[main_road, onramp1, onramp2, onramp3, onramp4, onramp5, offramp1, offramp2, offramp3], dt=dt,
        timeind=init_timeind, timesteps=timesteps)
    lanes = {main_road[0]: 0, main_road[1]: 1, onramp1[0]: 2, onramp2[0]: 2, onramp3[0]: 2, onramp4[0]: 2,
             onramp5[0]: 2, offramp1[0]: 2, offramp2[0]: 2, offramp3[0]: 2}

    return simulation, lanes


def merge_bottleneck(main_inflow=None, onramp_inflow=None, timesteps=18000):
    """Test simulation of merge bottleneck."""
    main_road = hs.Road(num_lanes=2, length=2000, name='main road')
    main_road.connect('exit', is_exit=True)
    onramp = hs.Road(num_lanes=1, length=[(950, 1300)], name='on ramp')
    onramp.merge(main_road, self_index=0, new_lane_index=1, self_pos=(1050, 1300), new_lane_pos=(1050, 1300))

    main_road.set_downstream({'method': 'free'})
    onramp.set_downstream({'method': 'free merge', 'self_lane': onramp[0]})

    def veh_parameters(route):
        def newveh(self, vehid, timeind):
            kwargs = hs.models.default_parameters(truck_prob=.05)
            self.newveh = hs.Vehicle(vehid, self, **kwargs, route=route.copy())
        return newveh

    mainroad_newveh = veh_parameters(['exit'])
    onramp_newveh = veh_parameters(['main road', 'exit'])
    increment_inflow = {'boundary_type': 'seql2', 'kwargs': {'c': .8, 'eql_speed': True, 'transition': 20}}
    if main_inflow is None:
        main_inflow = lambda timeind: 3200 / 3600 / 2 * min(timeind, 10000) / 10000
    if onramp_inflow is None:
        onramp_inflow = lambda timeind: 600 / 3600 * min(timeind, 12000) / 12000

    # main_get_inflow = {'time_series': main_inflow}
    # onramp_get_inflow = {'time_series': onramp_inflow}
    main_get_inflow = {'inflow_type': 'stochastic', 'args': (hs.road.M3Arrivals(main_inflow, 1.1, .95), .2, 0)}
    onramp_get_inflow = {'inflow_type': 'stochastic', 'args': (hs.road.M3Arrivals(onramp_inflow, 1.1, .95), .2, 0)}
    main_road.set_upstream(increment_inflow=increment_inflow, get_inflow=main_get_inflow, new_vehicle=mainroad_newveh)
    onramp.set_upstream(increment_inflow=increment_inflow, get_inflow=onramp_get_inflow, new_vehicle=onramp_newveh)
    simulation = hs.Simulation(roads=[main_road, onramp], dt=.2, timesteps=timesteps)
    lanes = {main_road[0]: 0, main_road[1]: 1, onramp[0]: 2}

    return simulation, lanes


if __name__ == '__main__':
    raise ValueError('This script is not meant to be called. The functions here are imported by other scripts.')
