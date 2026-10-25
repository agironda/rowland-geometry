
"""
Seely et al., Applied Optics 58, 5225-5232 (2019).
DOI: 10.1364/AO.58.005225

Table 1: Internal planes of an asymmetric quartz crystal.

Each entry contains:
    (reflection, 2_d_nm, alpha_deg, bragg_deg)

Alpha is the angle between the reflection and the
quartz (101) crystal plane.

Bragg angles are calculated at 8048 eV.
"""

SEELY_CUT = (1, 0, 1)
SEELY_ENERGY = 8048.0

SEELY_QUARTZ = [
    ((1, 0, 0), 0.8510, 38.21, 10.43),
    ((1, 0, 1), 0.6687,  0.00, 13.32),
    ((1, 0, 2), 0.4563, 19.37, 19.73),
    ((2, 0, 0), 0.4255, 38.21, 21.23),
    ((2, 0, 1), 0.3959, 16.73, 22.90),
    ((2, 0, 2), 0.3343,  0.00, 27.44),
    ((2, 0, 3), 0.2750, 11.53, 34.07),
    ((3, 0, 1), 0.2744, 23.51, 34.16),
    ((3, 0, 2), 0.2512, 10.52, 37.83),
    ((3, 0, 3), 0.2229,  0.00, 43.72),
]
