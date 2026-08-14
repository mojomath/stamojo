# ===----------------------------------------------------------------------=== #
# Stamojo - Special - Bessel functions
# Licensed under Apache 2.0
# ===----------------------------------------------------------------------=== #
"""Bessel functions.

This module provides implementations of Bessel functions of the first and
second kind, as well as modified Bessel functions and their exponentially
scaled variants.

Functions:
    - j0, j1, jn: Bessel functions of the first kind (orders 0, 1, n)
    - i0, i1, i0e, i1e: Modified Bessel functions of the first kind
    - y0, y1: Bessel functions of the second kind (orders 0, 1)

References:
    - fdlibm e_j0.c, e_j1.c (Sun Microsystems, 1993) -- used by scipy
    - Cephes Mathematical Library -- used by scipy for i0/i1
"""

from std.math import cos, exp, inf, log, nan, sin, sqrt

# === --------------------------------------------------------------------=== #
# General notes:
# TODO: Asymptotic expansions need to be implemented for large arguments to
# ensure accuracy and efficiency. The threshold for switching to asymptotic
# expansions should be determined empirically based on accuracy requirements.
# === ----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

comptime _INV_SQRT_PI: Float64 = 0.56418958354775627928  # 1/sqrt(pi)
comptime _TWO_OVER_PI: Float64 = 0.63661977236758138243  # 2/pi

# fdlibm constants for j0 (R0/S0 on [0, 2])
comptime _J0_R02: Float64 = 1.56249999999999947958e-02
comptime _J0_R03: Float64 = -1.89979294238854721751e-04
comptime _J0_R04: Float64 = 1.82954049532700665670e-06
comptime _J0_R05: Float64 = -4.61832688532103189199e-09
comptime _J0_S01: Float64 = 1.56191029464890010492e-02
comptime _J0_S02: Float64 = 1.16926784663337450260e-04
comptime _J0_S03: Float64 = 5.13546550207318111446e-07
comptime _J0_S04: Float64 = 1.16614003333790000205e-09

# fdlibm constants for j1 (R0/S0 on [0, 2])
comptime _J1_R00: Float64 = -6.25000000000000000000e-02
comptime _J1_R01: Float64 = 1.40705666955189706048e-03
comptime _J1_R02: Float64 = -1.59955631084035597520e-05
comptime _J1_R03: Float64 = 4.96727999609584448412e-08
comptime _J1_S01: Float64 = 1.91537599538363460805e-02
comptime _J1_S02: Float64 = 1.85946785588630915560e-04
comptime _J1_S03: Float64 = 1.17718464042623683263e-06
comptime _J1_S04: Float64 = 5.04636257076217042715e-09
comptime _J1_S05: Float64 = 1.23542274426137913908e-11

# fdlibm constants for y0 (U/V on [0, 2])
comptime _Y0_U00: Float64 = -7.38042951086872317523e-02
comptime _Y0_U01: Float64 = 1.76666452509181115538e-01
comptime _Y0_U02: Float64 = -1.38185671945596898896e-02
comptime _Y0_U03: Float64 = 3.47453432093683650238e-04
comptime _Y0_U04: Float64 = -3.81407053724364161125e-06
comptime _Y0_U05: Float64 = 1.95590137035022920206e-08
comptime _Y0_U06: Float64 = -3.98205194132103398453e-11
comptime _Y0_V01: Float64 = 1.27304834834123699328e-02
comptime _Y0_V02: Float64 = 7.60068627350353253702e-05
comptime _Y0_V03: Float64 = 2.59150851840457805467e-07
comptime _Y0_V04: Float64 = 4.41110311332675467403e-10

# fdlibm constants for y1 (U0/V0 on [0, 2])
comptime _Y1_U00: Float64 = -1.96057090646238940668e-01
comptime _Y1_U01: Float64 = 5.04438716639811282616e-02
comptime _Y1_U02: Float64 = -1.91256895875763547298e-03
comptime _Y1_U03: Float64 = 2.35252600561610495928e-05
comptime _Y1_U04: Float64 = -9.19099158039878874504e-08
comptime _Y1_V00: Float64 = 1.99167318236649903973e-02
comptime _Y1_V01: Float64 = 2.02552581025135171496e-04
comptime _Y1_V02: Float64 = 1.35608801097516229404e-06
comptime _Y1_V03: Float64 = 6.22741452364621501295e-09
comptime _Y1_V04: Float64 = 1.66559246207992079114e-11

# fdlibm pzero/qzero asymptotic coefficients (4 ranges)
# pzero: pR8/pS8 for x >= 8
comptime _PZ_P8_0: Float64 = 0.0
comptime _PZ_P8_1: Float64 = -7.03124999999900357484e-02
comptime _PZ_P8_2: Float64 = -8.08167041275349795626e00
comptime _PZ_P8_3: Float64 = -2.57063105679704847262e02
comptime _PZ_P8_4: Float64 = -2.48521641009428822144e03
comptime _PZ_P8_5: Float64 = -5.25304380490729545272e03
comptime _PZ_S8_0: Float64 = 1.16534364619668181717e02
comptime _PZ_S8_1: Float64 = 3.83374475364121826715e03
comptime _PZ_S8_2: Float64 = 4.05978572648472545552e04
comptime _PZ_S8_3: Float64 = 1.16752972564375915681e05
comptime _PZ_S8_4: Float64 = 4.76277284146730962675e04

# pzero: pR5/pS5 for x in [4.5454, 8]
comptime _PZ_P5_0: Float64 = -1.14125464691894502584e-11
comptime _PZ_P5_1: Float64 = -7.03124940873599280078e-02
comptime _PZ_P5_2: Float64 = -4.15961064470587782438e00
comptime _PZ_P5_3: Float64 = -6.76747652265167261021e01
comptime _PZ_P5_4: Float64 = -3.31231299649172967747e02
comptime _PZ_P5_5: Float64 = -3.46433388365604912451e02
comptime _PZ_S5_0: Float64 = 6.07539382692300335975e01
comptime _PZ_S5_1: Float64 = 1.05125230595704579173e03
comptime _PZ_S5_2: Float64 = 5.97897094333855784498e03
comptime _PZ_S5_3: Float64 = 9.62544514357774460223e03
comptime _PZ_S5_4: Float64 = 2.40605815922939109441e03

# pzero: pR3/pS3 for x in [2.8571, 4.547]
comptime _PZ_P3_0: Float64 = -2.54704601771951915620e-09
comptime _PZ_P3_1: Float64 = -7.03119616381481654654e-02
comptime _PZ_P3_2: Float64 = -2.40903221549529611423e00
comptime _PZ_P3_3: Float64 = -2.19659774734883086467e01
comptime _PZ_P3_4: Float64 = -5.80791704701737572236e01
comptime _PZ_P3_5: Float64 = -3.14479470594888503854e01
comptime _PZ_S3_0: Float64 = 3.58560338055209726349e01
comptime _PZ_S3_1: Float64 = 3.61513983050303863820e02
comptime _PZ_S3_2: Float64 = 1.19360783792111533330e03
comptime _PZ_S3_3: Float64 = 1.12799679856907414432e03
comptime _PZ_S3_4: Float64 = 1.73580930813335754692e02

# pzero: pR2/pS2 for x in [2, 2.8570]
comptime _PZ_P2_0: Float64 = -8.87534333032526411254e-08
comptime _PZ_P2_1: Float64 = -7.03030995483624743247e-02
comptime _PZ_P2_2: Float64 = -1.45073846780952986357e00
comptime _PZ_P2_3: Float64 = -7.63569613823527770791e00
comptime _PZ_P2_4: Float64 = -1.11931668860356747786e01
comptime _PZ_P2_5: Float64 = -3.23364579351335335033e00
comptime _PZ_S2_0: Float64 = 2.22202997532088808441e01
comptime _PZ_S2_1: Float64 = 1.36206794218215208048e02
comptime _PZ_S2_2: Float64 = 2.70470278658083486789e02
comptime _PZ_S2_3: Float64 = 1.53875394208320329881e02
comptime _PZ_S2_4: Float64 = 1.46576176948256193810e01

# fdlibm qzero asymptotic coefficients
# qzero: qR8/qS8 for x >= 8
comptime _QZ_R8_0: Float64 = 0.0
comptime _QZ_R8_1: Float64 = 7.32421874999935051953e-02
comptime _QZ_R8_2: Float64 = 1.17682064682252693899e01
comptime _QZ_R8_3: Float64 = 5.57673380256401856059e02
comptime _QZ_R8_4: Float64 = 8.85919720756468632317e03
comptime _QZ_R8_5: Float64 = 3.70146267776887834771e04
comptime _QZ_S8_0: Float64 = 1.63776026895689824414e02
comptime _QZ_S8_1: Float64 = 8.09834494656449805916e03
comptime _QZ_S8_2: Float64 = 1.42538291419120476348e05
comptime _QZ_S8_3: Float64 = 8.03309257119514397345e05
comptime _QZ_S8_4: Float64 = 8.40501579819060512818e05
comptime _QZ_S8_5: Float64 = -3.43899293537866615225e05

# qzero: qR5/qS5 for x in [4.5454, 8]
comptime _QZ_R5_0: Float64 = 1.84085963594515531381e-11
comptime _QZ_R5_1: Float64 = 7.32421766612684765896e-02
comptime _QZ_R5_2: Float64 = 5.83563508962056953777e00
comptime _QZ_R5_3: Float64 = 1.35111577286449829671e02
comptime _QZ_R5_4: Float64 = 1.02724376596164097464e03
comptime _QZ_R5_5: Float64 = 1.98997785864605384631e03
comptime _QZ_S5_0: Float64 = 8.27766102236537761883e01
comptime _QZ_S5_1: Float64 = 2.07781416421392987104e03
comptime _QZ_S5_2: Float64 = 1.88472887785718085070e04
comptime _QZ_S5_3: Float64 = 5.67511122894947329769e04
comptime _QZ_S5_4: Float64 = 3.59767538425114471465e04
comptime _QZ_S5_5: Float64 = -5.35434275601944773371e03

# qzero: qR3/qS3 for x in [2.8571, 4.547]
comptime _QZ_R3_0: Float64 = 4.37741014089738620906e-09
comptime _QZ_R3_1: Float64 = 7.32411180042911447163e-02
comptime _QZ_R3_2: Float64 = 3.34423137516170720929e00
comptime _QZ_R3_3: Float64 = 4.26218440745412650017e01
comptime _QZ_R3_4: Float64 = 1.70808091340565596283e02
comptime _QZ_R3_5: Float64 = 1.66733948696651168575e02
comptime _QZ_S3_0: Float64 = 4.87588729724587182091e01
comptime _QZ_S3_1: Float64 = 7.09689221056606015736e02
comptime _QZ_S3_2: Float64 = 3.70414822620111362994e03
comptime _QZ_S3_3: Float64 = 6.46042516752568917582e03
comptime _QZ_S3_4: Float64 = 2.51633368920368957333e03
comptime _QZ_S3_5: Float64 = -1.49247451836156386662e02

# qzero: qR2/qS2 for x in [2, 2.8570]
comptime _QZ_R2_0: Float64 = 1.50444444886983272379e-07
comptime _QZ_R2_1: Float64 = 7.32234265963079278272e-02
comptime _QZ_R2_2: Float64 = 1.99819174093815998816e00
comptime _QZ_R2_3: Float64 = 1.44956029347885735348e01
comptime _QZ_R2_4: Float64 = 3.16662317504781540833e01
comptime _QZ_R2_5: Float64 = 1.62527075710929267416e01
comptime _QZ_S2_0: Float64 = 3.03655848355219184498e01
comptime _QZ_S2_1: Float64 = 2.69348118608049844624e02
comptime _QZ_S2_2: Float64 = 8.44783757595320139444e02
comptime _QZ_S2_3: Float64 = 8.82935845112488550512e02
comptime _QZ_S2_4: Float64 = 2.12666388511798828631e02
comptime _QZ_S2_5: Float64 = -5.31095493882666946917e00

# fdlibm pone/qone asymptotic coefficients (for j1/y1)
# pone: pr8/ps8 for x >= 8
comptime _PO_PR8_0: Float64 = 0.0
comptime _PO_PR8_1: Float64 = 1.17187499999988647970e-01
comptime _PO_PR8_2: Float64 = 1.32394806593073575129e01
comptime _PO_PR8_3: Float64 = 4.12051854307378562225e02
comptime _PO_PR8_4: Float64 = 3.87474538913960532227e03
comptime _PO_PR8_5: Float64 = 7.91447954031891731574e03
comptime _PO_PS8_0: Float64 = 1.14207370375678408436e02
comptime _PO_PS8_1: Float64 = 3.65093083420853463394e03
comptime _PO_PS8_2: Float64 = 3.69562060269033463555e04
comptime _PO_PS8_3: Float64 = 9.76027935934950801311e04
comptime _PO_PS8_4: Float64 = 3.08042720627888811578e04

# pone: pr5/ps5 for x in [4.5454, 8]
comptime _PO_PR5_0: Float64 = 1.31990519556243522749e-11
comptime _PO_PR5_1: Float64 = 1.17187493190614097638e-01
comptime _PO_PR5_2: Float64 = 6.80275127868432871736e00
comptime _PO_PR5_3: Float64 = 1.08308182990189109773e02
comptime _PO_PR5_4: Float64 = 5.17636139533199752805e02
comptime _PO_PR5_5: Float64 = 5.28715201363337541807e02
comptime _PO_PS5_0: Float64 = 5.92805987221131331921e01
comptime _PO_PS5_1: Float64 = 9.91401418733614377743e02
comptime _PO_PS5_2: Float64 = 5.35326695291487976647e03
comptime _PO_PS5_3: Float64 = 7.84469031749551231769e03
comptime _PO_PS5_4: Float64 = 1.50404688810361062679e03

# pone: pr3/ps3 for x in [2.8571, 4.547]
comptime _PO_PR3_0: Float64 = 3.02503916137373618024e-09
comptime _PO_PR3_1: Float64 = 1.17186865567253592491e-01
comptime _PO_PR3_2: Float64 = 3.93297750033315640650e00
comptime _PO_PR3_3: Float64 = 3.51194035591636932736e01
comptime _PO_PR3_4: Float64 = 9.10550110750781271918e01
comptime _PO_PR3_5: Float64 = 4.85590685197364919645e01
comptime _PO_PS3_0: Float64 = 3.47913095001251519989e01
comptime _PO_PS3_1: Float64 = 3.36762458747825746741e02
comptime _PO_PS3_2: Float64 = 1.04687139975775130551e03
comptime _PO_PS3_3: Float64 = 8.90811346398256432622e02
comptime _PO_PS3_4: Float64 = 1.03787932439639277504e02

# pone: pr2/ps2 for x in [2, 2.8570]
comptime _PO_PR2_0: Float64 = 1.07710830106873743082e-07
comptime _PO_PR2_1: Float64 = 1.17176219462683348094e-01
comptime _PO_PR2_2: Float64 = 2.36851496667608785174e00
comptime _PO_PR2_3: Float64 = 1.22426109148261232917e01
comptime _PO_PR2_4: Float64 = 1.76939711271687727390e01
comptime _PO_PR2_5: Float64 = 5.07352312588818499250e00
comptime _PO_PS2_0: Float64 = 2.14364859363821409488e01
comptime _PO_PS2_1: Float64 = 1.25290227168402751090e02
comptime _PO_PS2_2: Float64 = 2.32276469057162813669e02
comptime _PO_PS2_3: Float64 = 1.17679373287147100768e02
comptime _PO_PS2_4: Float64 = 8.36463893371618283368e00

# qone: qr8/qs8 for x >= 8
comptime _QO_QR8_0: Float64 = 0.0
comptime _QO_QR8_1: Float64 = -1.02539062499992714161e-01
comptime _QO_QR8_2: Float64 = -1.62717534544589987888e01
comptime _QO_QR8_3: Float64 = -7.59601722513950107896e02
comptime _QO_QR8_4: Float64 = -1.18498066702429587167e04
comptime _QO_QR8_5: Float64 = -4.84385124285750353010e04
comptime _QO_QS8_0: Float64 = 1.61395369700722909556e02
comptime _QO_QS8_1: Float64 = 7.82538599923348465381e03
comptime _QO_QS8_2: Float64 = 1.33875336287249578163e05
comptime _QO_QS8_3: Float64 = 7.19657723683240939863e05
comptime _QO_QS8_4: Float64 = 6.66601232617776375264e05
comptime _QO_QS8_5: Float64 = -2.94490264303834643215e05

# qone: qr5/qs5 for x in [4.5454, 8]
comptime _QO_QR5_0: Float64 = -2.08979931141764104297e-11
comptime _QO_QR5_1: Float64 = -1.02539050241375426231e-01
comptime _QO_QR5_2: Float64 = -8.05644828123936029840e00
comptime _QO_QR5_3: Float64 = -1.83669607474888380239e02
comptime _QO_QR5_4: Float64 = -1.37319376065508163265e03
comptime _QO_QR5_5: Float64 = -2.61244440453215656817e03
comptime _QO_QS5_0: Float64 = 8.12765501384335777857e01
comptime _QO_QS5_1: Float64 = 1.99179873460485964642e03
comptime _QO_QS5_2: Float64 = 1.74684851924908907677e04
comptime _QO_QS5_3: Float64 = 4.98514270910352279316e04
comptime _QO_QS5_4: Float64 = 2.79480751638918118260e04
comptime _QO_QS5_5: Float64 = -4.71918354795128470869e03

# qone: qr3/qs3 for x in [2.8571, 4.547]
comptime _QO_QR3_0: Float64 = -5.07831226461766561369e-09
comptime _QO_QR3_1: Float64 = -1.02537829820837089745e-01
comptime _QO_QR3_2: Float64 = -4.61011581139473403113e00
comptime _QO_QR3_3: Float64 = -5.78472216562783643212e01
comptime _QO_QR3_4: Float64 = -2.28244540737631695038e02
comptime _QO_QR3_5: Float64 = -2.19210128478909325622e02
comptime _QO_QS3_0: Float64 = 4.76651550323729509273e01
comptime _QO_QS3_1: Float64 = 6.73865112676699709482e02
comptime _QO_QS3_2: Float64 = 3.38015286679526343505e03
comptime _QO_QS3_3: Float64 = 5.54772909720722782367e03
comptime _QO_QS3_4: Float64 = 1.90311919338810798763e03
comptime _QO_QS3_5: Float64 = -1.35201191444307340817e02

# qone: qr2/qs2 for x in [2, 2.8570]
comptime _QO_QR2_0: Float64 = -1.78381727510958865572e-07
comptime _QO_QR2_1: Float64 = -1.02517042607985553460e-01
comptime _QO_QR2_2: Float64 = -2.75220568278187460720e00
comptime _QO_QR2_3: Float64 = -1.96636162643703720221e01
comptime _QO_QR2_4: Float64 = -4.23253133372830490089e01
comptime _QO_QR2_5: Float64 = -2.13719211703704061733e01
comptime _QO_QS2_0: Float64 = 2.95333629060523854548e01
comptime _QO_QS2_1: Float64 = 2.52981549982190529136e02
comptime _QO_QS2_2: Float64 = 7.57502834868645436472e02
comptime _QO_QS2_3: Float64 = 7.39393205320467245656e02
comptime _QO_QS2_4: Float64 = 1.55949003336666123687e02
comptime _QO_QS2_5: Float64 = -4.95949898822628210127e00

# Cephes Chebyshev coefficients for exp(-x) * I0(x) on [0, 8]
# Reverse order (T_N-1 first), zero-order T_0 doubled (last element).
comptime _I0_A: InlineArray[Float64, 30] = [
    -4.41534164647933937950e-18,
    3.33079451882223809783e-17,
    -2.43127984654795469359e-16,
    1.71539128555513303061e-15,
    -1.16853328779934516808e-14,
    7.67618549860493561688e-14,
    -4.85644678311192946090e-13,
    2.95505266312963983461e-12,
    -1.72682629144155570723e-11,
    9.67580903537323691224e-11,
    -5.18979560163526290666e-10,
    2.65982372468238665035e-9,
    -1.30002500998624804212e-8,
    6.04699502254191894932e-8,
    -2.67079385394061173391e-7,
    1.11738753912010371815e-6,
    -4.41673835845875056359e-6,
    1.64484480707288970893e-5,
    -5.75419501008210370398e-5,
    1.88502885095841655729e-4,
    -5.76375574538582365885e-4,
    1.63947561694133579842e-3,
    -4.32430999505057594430e-3,
    1.05464603945949983183e-2,
    -2.37374148058994688156e-2,
    4.93052842396707084878e-2,
    -9.49010970480476444210e-2,
    1.71620901522208775349e-1,
    -3.04682672343198398683e-1,
    6.76795274409476084995e-1,
]

# Cephes Chebyshev coefficients for exp(-x) * sqrt(x) * I0(x) on [8, inf]
# Reverse order, zero-order T_0 doubled (last element).
comptime _I0_B: InlineArray[Float64, 25] = [
    -7.23318048787475395456e-18,
    -4.83050448594418207126e-18,
    4.46562142029675999901e-17,
    3.46122286769746109310e-17,
    -2.82762398051658348494e-16,
    -3.42548561967721913462e-16,
    1.77256013305652638360e-15,
    3.81168066935262242075e-15,
    -9.55484669882830764870e-15,
    -4.15056934728722208663e-14,
    1.54008621752140982691e-14,
    3.85277838274214270114e-13,
    7.18012445138366623367e-13,
    -1.79417853150680611778e-12,
    -1.32158118404477131188e-11,
    -3.14991652796324136454e-11,
    1.18891471078464383424e-11,
    4.94060238822496958910e-10,
    3.39623202570838634515e-9,
    2.26666899049817806459e-8,
    2.04891858946906374183e-7,
    2.89137052083475648297e-6,
    6.88975834691682398426e-5,
    3.36911647825569408990e-3,
    8.04490411014108831608e-1,
]

# Cephes Chebyshev coefficients for exp(-x) * I1(x) / x on [0, 8]
# Reverse order, zero-order T_0 doubled (last element).
comptime _I1_A: InlineArray[Float64, 29] = [
    2.77791411276104639959e-18,
    -2.11142121435816608115e-17,
    1.55363195773620046921e-16,
    -1.10559694773538630805e-15,
    7.60068429473540693410e-15,
    -5.04218550472791168711e-14,
    3.22379336594557470981e-13,
    -1.98397439776494371520e-12,
    1.17361862988909016308e-11,
    -6.66348972350202774223e-11,
    3.62559028155211703701e-10,
    -1.88724975172282928790e-9,
    9.38153738649577178388e-9,
    -4.44505912879632808065e-8,
    2.00329475355213526229e-7,
    -8.56872026469545474066e-7,
    3.47025130813767847674e-6,
    -1.32731636560394358279e-5,
    4.78156510755005422638e-5,
    -1.61760815825896745588e-4,
    5.12285956168575772895e-4,
    -1.51357245063125314899e-3,
    4.15642294431288815669e-3,
    -1.05640848946261981558e-2,
    2.47264490306265168283e-2,
    -5.29459812080949914269e-2,
    1.02643658689847095384e-1,
    -1.76416518357834055153e-1,
    2.52587186443633654823e-1,
]

# Cephes Chebyshev coefficients for exp(-x) * sqrt(x) * I1(x) on [8, inf]
# Reverse order, zero-order T_0 doubled (last element).
comptime _I1_B: InlineArray[Float64, 25] = [
    7.51729631084210481353e-18,
    4.41434832307170791151e-18,
    -4.65030536848935832153e-17,
    -3.20952592199342395980e-17,
    2.96262899764595013876e-16,
    3.30820231092092828324e-16,
    -1.88035477551078244854e-15,
    -3.81440307243700780478e-15,
    1.04202769841288027642e-14,
    4.27244001671195135429e-14,
    -2.10154184277266431302e-14,
    -4.08355111109219731823e-13,
    -7.19855177624590851209e-13,
    2.03562854414708950722e-12,
    1.41258074366137813316e-11,
    3.25260358301548823856e-11,
    -1.89749581235054123450e-11,
    -5.58974346219658380687e-10,
    -3.83538038596423702205e-9,
    -2.63146884688951950684e-8,
    -2.51223623787020892529e-7,
    -3.88256480887769039346e-6,
    -1.10588938762623716291e-4,
    -9.76109749136146840777e-3,
    7.78576235018280120474e-1,
]


# ===----------------------------------------------------------------------=== #
# Helper functions
# ===----------------------------------------------------------------------=== #


def _pzero(x: Float64) -> Float64:
    """fdlibm pzero(x): asymptotic factor P(x) for j0/y0.

    Approximates 1 + R/S where s = 1/x.
    """
    var ix = abs(x)
    var z = 1.0 / (ix * ix)
    var r: Float64
    var s: Float64

    if ix >= 8.0:
        r = _PZ_P8_0 + z * (
            _PZ_P8_1
            + z * (_PZ_P8_2 + z * (_PZ_P8_3 + z * (_PZ_P8_4 + z * _PZ_P8_5)))
        )
        s = 1.0 + z * (
            _PZ_S8_0
            + z * (_PZ_S8_1 + z * (_PZ_S8_2 + z * (_PZ_S8_3 + z * _PZ_S8_4)))
        )
    elif ix >= 4.5454:
        r = _PZ_P5_0 + z * (
            _PZ_P5_1
            + z * (_PZ_P5_2 + z * (_PZ_P5_3 + z * (_PZ_P5_4 + z * _PZ_P5_5)))
        )
        s = 1.0 + z * (
            _PZ_S5_0
            + z * (_PZ_S5_1 + z * (_PZ_S5_2 + z * (_PZ_S5_3 + z * _PZ_S5_4)))
        )
    elif ix >= 2.8571:
        r = _PZ_P3_0 + z * (
            _PZ_P3_1
            + z * (_PZ_P3_2 + z * (_PZ_P3_3 + z * (_PZ_P3_4 + z * _PZ_P3_5)))
        )
        s = 1.0 + z * (
            _PZ_S3_0
            + z * (_PZ_S3_1 + z * (_PZ_S3_2 + z * (_PZ_S3_3 + z * _PZ_S3_4)))
        )
    else:
        r = _PZ_P2_0 + z * (
            _PZ_P2_1
            + z * (_PZ_P2_2 + z * (_PZ_P2_3 + z * (_PZ_P2_4 + z * _PZ_P2_5)))
        )
        s = 1.0 + z * (
            _PZ_S2_0
            + z * (_PZ_S2_1 + z * (_PZ_S2_2 + z * (_PZ_S2_3 + z * _PZ_S2_4)))
        )

    return 1.0 + r / s


def _qzero(x: Float64) -> Float64:
    """fdlibm qzero(x): asymptotic factor Q(x) for j0/y0.

    Approximates (-0.125 + R/S) / x where s = 1/x.
    """
    var ix = abs(x)
    var z = 1.0 / (ix * ix)
    var r: Float64
    var s: Float64

    if ix >= 8.0:
        r = _QZ_R8_0 + z * (
            _QZ_R8_1
            + z * (_QZ_R8_2 + z * (_QZ_R8_3 + z * (_QZ_R8_4 + z * _QZ_R8_5)))
        )
        s = 1.0 + z * (
            _QZ_S8_0
            + z
            * (
                _QZ_S8_1
                + z
                * (_QZ_S8_2 + z * (_QZ_S8_3 + z * (_QZ_S8_4 + z * _QZ_S8_5)))
            )
        )
    elif ix >= 4.5454:
        r = _QZ_R5_0 + z * (
            _QZ_R5_1
            + z * (_QZ_R5_2 + z * (_QZ_R5_3 + z * (_QZ_R5_4 + z * _QZ_R5_5)))
        )
        s = 1.0 + z * (
            _QZ_S5_0
            + z
            * (
                _QZ_S5_1
                + z
                * (_QZ_S5_2 + z * (_QZ_S5_3 + z * (_QZ_S5_4 + z * _QZ_S5_5)))
            )
        )
    elif ix >= 2.8571:
        r = _QZ_R3_0 + z * (
            _QZ_R3_1
            + z * (_QZ_R3_2 + z * (_QZ_R3_3 + z * (_QZ_R3_4 + z * _QZ_R3_5)))
        )
        s = 1.0 + z * (
            _QZ_S3_0
            + z
            * (
                _QZ_S3_1
                + z
                * (_QZ_S3_2 + z * (_QZ_S3_3 + z * (_QZ_S3_4 + z * _QZ_S3_5)))
            )
        )
    else:
        r = _QZ_R2_0 + z * (
            _QZ_R2_1
            + z * (_QZ_R2_2 + z * (_QZ_R2_3 + z * (_QZ_R2_4 + z * _QZ_R2_5)))
        )
        s = 1.0 + z * (
            _QZ_S2_0
            + z
            * (
                _QZ_S2_1
                + z
                * (_QZ_S2_2 + z * (_QZ_S2_3 + z * (_QZ_S2_4 + z * _QZ_S2_5)))
            )
        )

    return (-0.125 + r / s) / ix


def _pone(x: Float64) -> Float64:
    """fdlibm pone(x): asymptotic factor P(x) for j1/y1.

    Approximates 1 + R/S where s = 1/x.
    """
    var ix = abs(x)
    var z = 1.0 / (ix * ix)
    var r: Float64
    var s: Float64

    if ix >= 8.0:
        r = _PO_PR8_0 + z * (
            _PO_PR8_1
            + z
            * (_PO_PR8_2 + z * (_PO_PR8_3 + z * (_PO_PR8_4 + z * _PO_PR8_5)))
        )
        s = 1.0 + z * (
            _PO_PS8_0
            + z
            * (_PO_PS8_1 + z * (_PO_PS8_2 + z * (_PO_PS8_3 + z * _PO_PS8_4)))
        )
    elif ix >= 4.5454:
        r = _PO_PR5_0 + z * (
            _PO_PR5_1
            + z
            * (_PO_PR5_2 + z * (_PO_PR5_3 + z * (_PO_PR5_4 + z * _PO_PR5_5)))
        )
        s = 1.0 + z * (
            _PO_PS5_0
            + z
            * (_PO_PS5_1 + z * (_PO_PS5_2 + z * (_PO_PS5_3 + z * _PO_PS5_4)))
        )
    elif ix >= 2.8571:
        r = _PO_PR3_0 + z * (
            _PO_PR3_1
            + z
            * (_PO_PR3_2 + z * (_PO_PR3_3 + z * (_PO_PR3_4 + z * _PO_PR3_5)))
        )
        s = 1.0 + z * (
            _PO_PS3_0
            + z
            * (_PO_PS3_1 + z * (_PO_PS3_2 + z * (_PO_PS3_3 + z * _PO_PS3_4)))
        )
    else:
        r = _PO_PR2_0 + z * (
            _PO_PR2_1
            + z
            * (_PO_PR2_2 + z * (_PO_PR2_3 + z * (_PO_PR2_4 + z * _PO_PR2_5)))
        )
        s = 1.0 + z * (
            _PO_PS2_0
            + z
            * (_PO_PS2_1 + z * (_PO_PS2_2 + z * (_PO_PS2_3 + z * _PO_PS2_4)))
        )

    return 1.0 + r / s


def _qone(x: Float64) -> Float64:
    """fdlibm qone(x): asymptotic factor Q(x) for j1/y1.

    Approximates (0.375 + R/S) / x where s = 1/x.
    """
    var ix = abs(x)
    var z = 1.0 / (ix * ix)
    var r: Float64
    var s: Float64

    if ix >= 8.0:
        r = _QO_QR8_0 + z * (
            _QO_QR8_1
            + z
            * (_QO_QR8_2 + z * (_QO_QR8_3 + z * (_QO_QR8_4 + z * _QO_QR8_5)))
        )
        s = 1.0 + z * (
            _QO_QS8_0
            + z
            * (
                _QO_QS8_1
                + z
                * (
                    _QO_QS8_2
                    + z * (_QO_QS8_3 + z * (_QO_QS8_4 + z * _QO_QS8_5))
                )
            )
        )
    elif ix >= 4.5454:
        r = _QO_QR5_0 + z * (
            _QO_QR5_1
            + z
            * (_QO_QR5_2 + z * (_QO_QR5_3 + z * (_QO_QR5_4 + z * _QO_QR5_5)))
        )
        s = 1.0 + z * (
            _QO_QS5_0
            + z
            * (
                _QO_QS5_1
                + z
                * (
                    _QO_QS5_2
                    + z * (_QO_QS5_3 + z * (_QO_QS5_4 + z * _QO_QS5_5))
                )
            )
        )
    elif ix >= 2.8571:
        r = _QO_QR3_0 + z * (
            _QO_QR3_1
            + z
            * (_QO_QR3_2 + z * (_QO_QR3_3 + z * (_QO_QR3_4 + z * _QO_QR3_5)))
        )
        s = 1.0 + z * (
            _QO_QS3_0
            + z
            * (
                _QO_QS3_1
                + z
                * (
                    _QO_QS3_2
                    + z * (_QO_QS3_3 + z * (_QO_QS3_4 + z * _QO_QS3_5))
                )
            )
        )
    else:
        r = _QO_QR2_0 + z * (
            _QO_QR2_1
            + z
            * (_QO_QR2_2 + z * (_QO_QR2_3 + z * (_QO_QR2_4 + z * _QO_QR2_5)))
        )
        s = 1.0 + z * (
            _QO_QS2_0
            + z
            * (
                _QO_QS2_1
                + z
                * (
                    _QO_QS2_2
                    + z * (_QO_QS2_3 + z * (_QO_QS2_4 + z * _QO_QS2_5))
                )
            )
        )

    return (0.375 + r / s) / ix


def _chbevl[N: Int](x: Float64, array: InlineArray[Float64, N]) -> Float64:
    """Evaluate Chebyshev series via Clenshaw's recurrence.

    Coefficients stored in reverse order (highest degree first).
    Zero-order coefficient (last in array) is doubled.
    """
    var b0 = array[0]
    var b1 = 0.0
    var b2 = 0.0
    var i = 1
    while i < N:
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + array[i]
        i += 1
    return 0.5 * (b0 - b2)


# ===----------------------------------------------------------------------=== #
# Bessel functions of the first kind
# ===----------------------------------------------------------------------=== #


def j0(x: Float64) -> Float64:
    """Bessel function of the first kind of order 0.

    Uses the fdlibm algorithm (same as scipy.special.j0).

    Args:
        x: Input value.

    Returns:
        J₀(x).

    Examples:
        ```mojo
        from stamojo.special import j0
        from std.testing import assert_almost_equal

        def main() raises:
            assert_almost_equal(j0(1.0), 0.7651976865579666, atol=1e-12)
        ```
    """
    var ax = abs(x)

    # |x| >= 2.0: asymptotic expansion
    if ax >= 2.0:
        var s = sin(ax)
        var c = cos(ax)
        var ss = s - c
        var cc = s + c
        # Avoid cancellation: sin(x) +- cos(x) = -cos(2x) / (sin(x) -+ cos(x))
        if ax < 1e150:
            var z = -cos(ax + ax)
            if (s * c) < 0.0:
                cc = z / ss
            else:
                ss = z / cc
        if ax > 1e150:
            return _INV_SQRT_PI * cc / sqrt(ax)
        var u = _pzero(ax)
        var v = _qzero(ax)
        return _INV_SQRT_PI * (u * cc - v * ss) / sqrt(ax)

    # |x| < 2**-13: tiny
    if ax < 1.220703125e-04:
        if ax < 7.450580596923828125e-09:
            return 1.0
        return 1.0 - 0.25 * ax * ax

    # |x| < 2.0: rational approximation
    var z = x * x
    var r = z * (_J0_R02 + z * (_J0_R03 + z * (_J0_R04 + z * _J0_R05)))
    var s = 1.0 + z * (_J0_S01 + z * (_J0_S02 + z * (_J0_S03 + z * _J0_S04)))
    if ax < 1.0:
        return 1.0 + z * (-0.25 + r / s)
    else:
        var u = 0.5 * ax
        return (1.0 + u) * (1.0 - u) + z * (r / s)


def j1(x: Float64) -> Float64:
    """Bessel function of the first kind of order 1.

    Uses the fdlibm algorithm (same as scipy.special.j1).

    Args:
        x: Input value.

    Returns:
        J₁(x).

    Examples:
        ```mojo
        from stamojo.special import j1
        from std.testing import assert_almost_equal

        def main() raises:
            assert_almost_equal(j1(1.0), 0.44005058574493355, atol=1e-12)
        ```
    """
    var ax = abs(x)
    var sign: Float64 = 1.0 if x >= 0.0 else -1.0

    # |x| >= 2.0: asymptotic expansion
    if ax >= 2.0:
        var s = sin(ax)
        var c = cos(ax)
        var ss = -s - c
        var cc = s - c
        # Avoid cancellation
        if ax < 1e150:
            var z = cos(ax + ax)
            if (s * c) > 0.0:
                cc = z / ss
            else:
                ss = z / cc
        if ax > 1e150:
            return sign * _INV_SQRT_PI * cc / sqrt(ax)
        var u = _pone(ax)
        var v = _qone(ax)
        return sign * _INV_SQRT_PI * (u * cc - v * ss) / sqrt(ax)

    # |x| < 2**-27: tiny
    if ax < 7.450580596923828125e-09:
        return 0.5 * x

    # |x| < 2.0: rational approximation
    var z = x * x
    var r = z * (_J1_R00 + z * (_J1_R01 + z * (_J1_R02 + z * _J1_R03)))
    var s = 1.0 + z * (
        _J1_S01 + z * (_J1_S02 + z * (_J1_S03 + z * (_J1_S04 + z * _J1_S05)))
    )
    r *= x
    return x * 0.5 + r / s


def jn[n: Int](x: Float64) -> Float64:
    """Bessel function of the first kind of order *n*.

    Parameters:
        n: Order of the Bessel function (integer).

    Args:
        x: Input value.

    Returns:
        Jₙ(x).
    """

    comptime if n == 0:
        return j0(x)

    comptime if n == 1:
        return j1(x)

    comptime m = n if n >= 0 else -n
    comptime sign: Float64 = -1.0 if (n < 0 and (m % 2 == 1)) else 1.0

    var ax = abs(x)

    # For small n relative to x, use forward recurrence.
    if Float64(m) <= ax:
        var jm1 = j0(x)
        var jcur = j1(x)
        for k in range(1, m):
            var jnext = (2.0 * Float64(k) / x) * jcur - jm1
            jm1 = jcur
            jcur = jnext
        return sign * jcur

    # For large n relative to x, use power series.
    var fact: Float64 = 1.0
    for i in range(1, m + 1):
        fact *= Float64(i)
    var term: Float64 = 1.0
    for _ in range(m):
        term *= x * 0.5
    term /= fact

    var res = term
    var x2 = x * x * 0.25
    for k in range(1, 50):
        term *= -x2 / (Float64(k) * Float64(k + m))
        res += term
    return sign * res


# ===----------------------------------------------------------------------=== #
# Modified Bessel functions of the first kind and their scaled forms
# ===----------------------------------------------------------------------=== #


def i0(x: Float64) -> Float64:
    """Modified Bessel function of the first kind of order 0.

    Uses the Cephes algorithm (same as scipy.special.i0).

    Args:
        x: Input value.

    Returns:
        I₀(x).

    Examples:
        ```mojo
        from stamojo.special import i0
        from std.testing import assert_almost_equal

        def main() raises:
            assert_almost_equal(i0(1.0), 1.2660658777520082, atol=1e-12)
        ```
    """
    var ax = abs(x)

    if ax <= 8.0:
        var y = ax / 2.0 - 2.0
        return exp(ax) * _chbevl(y, _I0_A)
    else:
        return exp(ax) * _chbevl(32.0 / ax - 2.0, _I0_B) / sqrt(ax)


def i1(x: Float64) -> Float64:
    """Modified Bessel function of the first kind of order 1.

    Uses the Cephes algorithm (same as scipy.special.i1).

    Args:
        x: Input value.

    Returns:
        I₁(x).

    Examples:
        ```mojo
        from stamojo.special import i1
        from std.testing import assert_almost_equal

        def main() raises:
            assert_almost_equal(i1(1.0), 0.5651591039924851, atol=1e-12)
        ```
    """
    var ax = abs(x)
    var sign: Float64 = 1.0 if x >= 0.0 else -1.0

    if ax <= 8.0:
        var y = ax / 2.0 - 2.0
        return sign * _chbevl(y, _I1_A) * ax * exp(ax)
    else:
        return sign * exp(ax) * _chbevl(32.0 / ax - 2.0, _I1_B) / sqrt(ax)


def i0e(x: Float64) -> Float64:
    """Exponentially scaled modified Bessel function of the first kind
    of order 0: ``i0e(x) = exp(-|x|) * i0(x)``.

    Args:
        x: Input value.

    Returns:
        Value of exp(-|x|) * I₀(x).

    Examples:
        ```mojo
        from stamojo.special import i0e
        from std.testing import assert_almost_equal

        def main() raises:
            assert_almost_equal(i0e(1.0), 0.4657596075936405, atol=1e-12)
        ```
    """
    var ax = abs(x)

    if ax <= 8.0:
        var y = ax / 2.0 - 2.0
        return _chbevl(y, _I0_A)
    else:
        return _chbevl(32.0 / ax - 2.0, _I0_B) / sqrt(ax)


def i1e(x: Float64) -> Float64:
    """Exponentially scaled modified Bessel function of the first kind
    of order 1: ``i1e(x) = exp(-|x|) * i1(x)``.

    Args:
        x: Input value.

    Returns:
        Value of exp(-|x|) * I₁(x).

    Examples:
        ```mojo
        from stamojo.special import i1e
        from std.testing import assert_almost_equal

        def main() raises:
            assert_almost_equal(i1e(1.0), 0.2079104153497085, atol=1e-12)
        ```
    """
    var ax = abs(x)
    var sign: Float64 = 1.0 if x >= 0.0 else -1.0

    if ax <= 8.0:
        var y = ax / 2.0 - 2.0
        return sign * _chbevl(y, _I1_A) * ax
    else:
        return sign * _chbevl(32.0 / ax - 2.0, _I1_B) / sqrt(ax)


# ===----------------------------------------------------------------------=== #
# Bessel functions of the second kind
# ===----------------------------------------------------------------------=== #


def y0(x: Float64) -> Float64:
    """Bessel function of the second kind of order 0.

    Uses the fdlibm algorithm (same as scipy.special.y0).
    Defined for x > 0. Returns -∞ at x = 0 and NaN for x < 0.

    Args:
        x: Input value (must be positive).

    Returns:
        Y₀(x).

    Examples:
        ```mojo
        from stamojo.special import y0
        from std.testing import assert_almost_equal

        def main() raises:
            assert_almost_equal(y0(1.0), 0.08825696421567697, atol=1e-12)
        ```
    """
    if x == 0.0:
        return -inf[DType.float64]()
    if x < 0.0:
        return nan[DType.float64]()

    # |x| >= 2.0: asymptotic expansion
    if x >= 2.0:
        var s = sin(x)
        var c = cos(x)
        var ss = s - c
        var cc = s + c
        # Avoid cancellation
        if x < 1e150:
            var z = -cos(x + x)
            if (s * c) < 0.0:
                cc = z / ss
            else:
                ss = z / cc
        if x > 1e150:
            return _INV_SQRT_PI * ss / sqrt(x)
        var u = _pzero(x)
        var v = _qzero(x)
        return _INV_SQRT_PI * (u * ss + v * cc) / sqrt(x)

    # |x| < 2**-27: tiny
    if x < 7.450580596923828125e-09:
        return _Y0_U00 + _TWO_OVER_PI * log(x)

    # |x| < 2.0: rational approximation
    var z = x * x
    var u = _Y0_U00 + z * (
        _Y0_U01
        + z
        * (
            _Y0_U02
            + z * (_Y0_U03 + z * (_Y0_U04 + z * (_Y0_U05 + z * _Y0_U06)))
        )
    )
    var v = 1.0 + z * (_Y0_V01 + z * (_Y0_V02 + z * (_Y0_V03 + z * _Y0_V04)))
    return u / v + _TWO_OVER_PI * (j0(x) * log(x))


def y1(x: Float64) -> Float64:
    """Bessel function of the second kind of order 1.

    Uses the fdlibm algorithm (same as scipy.special.y1).
    Defined for x > 0. Returns -∞ at x = 0 and NaN for x < 0.

    Args:
        x: Input value (must be positive).

    Returns:
        Y₁(x).

    Examples:
        ```mojo
        from stamojo.special import y1
        from std.testing import assert_almost_equal

        def main() raises:
            assert_almost_equal(y1(1.0), -0.7812128213002887, atol=1e-12)
        ```
    """
    if x == 0.0:
        return -inf[DType.float64]()
    if x < 0.0:
        return nan[DType.float64]()

    # |x| >= 2.0: asymptotic expansion
    if x >= 2.0:
        var s = sin(x)
        var c = cos(x)
        var ss = -s - c
        var cc = s - c
        # Avoid cancellation
        if x < 1e150:
            var z = cos(x + x)
            if (s * c) > 0.0:
                cc = z / ss
            else:
                ss = z / cc
        if x > 1e150:
            return _INV_SQRT_PI * ss / sqrt(x)
        var u = _pone(x)
        var v = _qone(x)
        return _INV_SQRT_PI * (u * ss + v * cc) / sqrt(x)

    # |x| < 2**-54: tiny
    if x < 5.55111512312578270212e-17:
        return -_TWO_OVER_PI / x

    # |x| < 2.0: rational approximation
    var z = x * x
    var u = _Y1_U00 + z * (
        _Y1_U01 + z * (_Y1_U02 + z * (_Y1_U03 + z * _Y1_U04))
    )
    var v = 1.0 + z * (
        _Y1_V00 + z * (_Y1_V01 + z * (_Y1_V02 + z * (_Y1_V03 + z * _Y1_V04)))
    )
    return x * (u / v) + _TWO_OVER_PI * (j1(x) * log(x) - 1.0 / x)
