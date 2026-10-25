
"""
Orlat et al., Review of Scientific Instruments 97, 075203 (2026).
DOI: 10.1063/5.0312021

Table 1: Asymmetric crystal configurations at U and Pu L3 edges.

Each entry contains:
    (material, reflection, cut, alpha_deg, bragg_deg, theta_m_deg)

alpha_deg:
    Crystallographic asymmetry angle between the reflection
    and the crystal surface (cut).

bragg_deg:
    Bragg angle at the specified incident energy.

theta_m_deg:
    Meridional angle, theta_B + alpha.
"""

# Values at U L3 edge, 17166 eV
ORLAT_U_L3 = [
    ("Si", (12, 8, 0),   (5, 3, 1), 10.1, 73.6, 83.7),
    ("Si", (12, 6, 6),   (5, 3, 3),  5.1, 77.6, 82.8),
    ("Si", (10, 10, 4),  (3, 3, 1),  2.5, 77.8, 80.3),
    ("Si", (10, 10, 4),  (5, 5, 1),  7.7, 77.8, 85.5),
    ("Si", (12, 8, 4),   (6, 4, 2),  0.0, 84.5, 84.5),
    ("Ge", (9, 9, 9),    (1, 1, 1),  0.0, 84.3, 84.3),
]

# Values at Pu L3 edge, 18057 eV
ORLAT_PU_L3 = [
    ("Ge", (14, 6, 4),   (4, 2, 2),  9.9, 72.9, 82.8),
    ("Ge", (10, 10, 8),  (1, 1, 1),  5.8, 80.3, 86.1),
    ("Si", (14, 6, 0),   (5, 3, 1), 12.4, 74.4, 86.8),
    ("Si", (12, 10, 2),  (5, 5, 1),  5.2, 84.7, 89.8),
    ("Si", (9, 9, 9),    (1, 1, 1),  0.0, 80.2, 80.2),
]

ORLAT_TABLES = [
    ("U-L3",  17166.0, ORLAT_U_L3),
    ("Pu-L3", 18057.0, ORLAT_PU_L3),
]
