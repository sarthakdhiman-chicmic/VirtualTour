"""Hero kinds shared by the analysis (structure.py, needs librosa) and the planner (numpy only).

Within a tier, what a moment IS outranks how loud it is: the music changing (a drop, a dropout, an
entry, a new section) > a whoosh > a single hit (a hit's strength saturates easily)."""
KIND_RANK = {"drop": 3, "dropout": 2, "entry": 2, "section": 2, "breakdown": 2, "whoosh": 1,
             "bass_hit": 0, "big_hit": 0}
