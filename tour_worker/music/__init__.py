"""Music analysis for beat sync (the `music` extra: pip install "tour-worker[music]" - librosa,
scipy, soundfile). Not used by the render jobs: the backend sends the analysed plan; the track is
analysed once, offline (find_transitions.py CLI / analysis.py).
"""
