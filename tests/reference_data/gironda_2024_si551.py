
"""
Gironda et al., Journal of Analytical Atomic Spectrometry
39, 1375-1387 (2024).
DOI: 10.1039/D3JA00437F

Table 1: Optimal asymmetric reflections for Si(551).

Each entry contains:
    (emission, energy_ev, reflection, bragg_deg, alpha_deg, theta_m_deg)

All configurations use the same Si(551) crystal cut.
theta_m = theta_B + alpha.

Overbars in Miller indices are represented by negative integers.
"""

GIRONDA_MATERIAL = "Si"
GIRONDA_CUT = (5, 5, 1)

GIRONDA_2024_ASYMMETRIC = [
    ("V Kβ",    5428, (3, 3, -1), 66.47, 21.31,  87.78),
    ("Mn Kα",   5900, (4, 2,  2), 71.44, 23.84,  95.28),
    ("Mn Kβ",   6492, (3, 3,  3), 66.04, 27.21,  93.25),
    ("Fe Kα",   6405, (3, 3,  3), 67.85, 27.21,  95.06),
    ("Fe Kβ",   7059, (5, 3,  1), 73.11, 13.97,  87.08),
    ("Ni Kα",   7480, (6, 2,  0), 74.87, 27.67, 102.54),
    ("Ni Kβ",   8267, (5, 5, -1), 80.49, 16.10,  96.59),
    ("Cu Kα",   8046, (4, 4,  4), 79.45, 27.21, 106.66),
    ("Cu Kβ",   8904, (7, 3,  1), 80.03, 21.61, 101.64),
    ("Zn Kα",   8637, (6, 4,  2), 81.57, 13.34,  94.91),
    ("Zn Kβ",   9570, (7, 3,  3), 77.56, 24.95, 102.51),
    ("Sr Kα",  14161, (8, 8,  4), 75.35, 11.42,  86.77),
    ("Sr Kβ",  15825, (12, 6, 2), 78.13, 18.25,  96.38),
]
