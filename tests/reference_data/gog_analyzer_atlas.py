
"""
Reference data from Gog et al., Analyzer Atlas,
Advanced Photon Source.

Each reflection entry contains:
    (material, hkl, e0_ev, bragg_deg)

e0_ev:
    Backscattering energy (theta_B = 90 degrees).

bragg_deg:
    Bragg angle at the incident energy specified
    for the corresponding table.
"""


# Values from Gog et. al, crystal atlas for Cr K at 5989 eV
GOG_CR_K = [
    ("Ge",          (3, 3, 3),  5694, 71.94),
    ("Ge",          (5, 1, 1),  5694, 71.94),
    ("Ge",          (1, 5, 1),  5694, 71.94),

    ("Si",          (3, 3, 3),  5931, 82.03),
    ("Si",          (5, 1, 1),  5931, 82.03),
    ("Si",          (1, 5, 1),  5931, 82.03),

    ("Sapphire",    (1, 3,-5),  5926, 81.68),
    ("Sapphire",    (3,-4,-5),  5926, 81.68),
    ("Sapphire",    (4,-1, 5),  5926, 81.68),
            
    ("Quartz",      (1,-1,-5),  5917, 81.12),
    ("Quartz",      (0, 1,-5),  5917, 81.12),
    ("Quartz",      (1, 0, 5),  5917, 81.12),

    ("LiNbO3",      (0,-4,-2),  5633, 70.16),
    ("LiNbO3",      (4, 0,-2),  5633, 70.16),
    ("LiNbO3",      (-4,4,-2),  5633, 70.16),
]

# Values from Gog et. al, crystal atlas for Mn K at 6539 eV
GOG_MN_K = [
    ("Ge",          (1, 3, 5),  6483, 82.48),
    ("Ge",          (0, 4, 4),  6199, 71.43),

    ("Si",          (0, 4, 4),  6457, 80.92),

    ("Sapphire",    (1, 2,-10), 6215, 71.88),
    ("Sapphire",    (3,-1, 10), 6215, 71.88),
    ("Sapphire",    (2,-3,-10), 6215, 71.88),

    ("Quartz",      (0, 3,-4),  6337, 75.72),
    ("Quartz",      (3, 0, 4),  6337, 75.72),
    ("Quartz",      (3,-3, 4),  6337, 75.72),

    ("LiNbO3",      (1, 4, 3),  6512, 84.76),
    ("LiNbO3",      (4, 1, 3),  6512, 84.76),
    ("LiNbO3",      (5,-1, 3),  6512, 84.76),
]

# Values from Gog et. al, crystal atlas for Ni K at 8333 eV
GOG_NI_K = [
    ("Ge",          (2, 4, 6),  8200, 79.75),

    ("Si",          (1, 1, 7),  8152, 78.02),
    ("Si",          (4, 4, 4),  7908, 71.63),

    ("Sapphire",    (0, 2,16),  8208, 80.06),

    ("Quartz",      (5, 1, 1),  8193, 79.48),

    ("LiNbO3",      (2,-6, 8),  8181, 79.05),
]

# Values from Gog et al., Analyzer Atlas for Se K at 12658 eV
GOG_SE_K = [
    ("Ge",          (1, 3, 11),   12542, 82.23),
    ("Ge",          (0, 8, 8),    12397, 78.35),
    ("Ge",          (2, 4, 10),   12004, 71.50),

    ("Si",          (2, 4, 10),   12504, 81.05),
    ("Si",          (3, 5, 9),    12241, 75.25),

    ("Sapphire",    (4, 4, 15),   12645, 87.40),
    ("Sapphire",    (2, 4,-20),   12430, 79.10),
    ("Sapphire",    (4, 5,-4),    11904, 70.12),

    ("Quartz",      (5, 5, 0),    12618, 85.44),
    ("Quartz",      (7, 1,-5),    12405, 78.53),
    ("Quartz",      (3, 3, 8),    11896, 70.02),

    ("LiNbO3",      (2, 4,-23),   12646, 87.50),
    ("LiNbO3",      (1, 8, 8),    12407, 78.58),
    ("LiNbO3",      (1, 8, 2),    11914, 70.26),
]

GOG_TABLES = [
    ("Cr-K",  5989.0, GOG_CR_K),
    ("Mn-K",  6539.0, GOG_MN_K),
    ("Ni-K",  8333.0, GOG_NI_K),
    ("Se-K", 12658.0, GOG_SE_K),
]