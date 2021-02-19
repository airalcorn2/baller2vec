import os

NORMALIZATION_COEF = 7
PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
INTERVAL = 10
DIFF = 6
X_MIN = 0
X_MAX = 100
Y_MIN = 0
Y_MAX = 50
COL_WIDTH = 0.3
SCALE = 1.65
FONTSIZE = 6
X_CENTER = X_MAX / 2 - DIFF / 1.5 + 0.10
Y_CENTER = Y_MAX - DIFF / 1.5 - 0.35
BALL_COLOR = "#ff8c00"
(COURT_WIDTH, COURT_LENGTH) = (50, 94)
TEAM_ID2PROPS = {
    1610612737: {"color": "#E13A3E", "abbreviation": "ATL"},
    1610612738: {"color": "#008348", "abbreviation": "BOS"},
    1610612751: {"color": "#061922", "abbreviation": "BKN"},
    1610612766: {"color": "#1D1160", "abbreviation": "CHA"},
    1610612741: {"color": "#CE1141", "abbreviation": "CHI"},
    1610612739: {"color": "#860038", "abbreviation": "CLE"},
    1610612742: {"color": "#007DC5", "abbreviation": "DAL"},
    1610612743: {"color": "#4D90CD", "abbreviation": "DEN"},
    1610612765: {"color": "#006BB6", "abbreviation": "DET"},
    1610612744: {"color": "#FDB927", "abbreviation": "GSW"},
    1610612745: {"color": "#CE1141", "abbreviation": "HOU"},
    1610612754: {"color": "#00275D", "abbreviation": "IND"},
    1610612746: {"color": "#ED174C", "abbreviation": "LAC"},
    1610612747: {"color": "#552582", "abbreviation": "LAL"},
    1610612763: {"color": "#0F586C", "abbreviation": "MEM"},
    1610612748: {"color": "#98002E", "abbreviation": "MIA"},
    1610612749: {"color": "#00471B", "abbreviation": "MIL"},
    1610612750: {"color": "#005083", "abbreviation": "MIN"},
    1610612740: {"color": "#002B5C", "abbreviation": "NOP"},
    1610612752: {"color": "#006BB6", "abbreviation": "NYK"},
    1610612760: {"color": "#007DC3", "abbreviation": "OKC"},
    1610612753: {"color": "#007DC5", "abbreviation": "ORL"},
    1610612755: {"color": "#006BB6", "abbreviation": "PHI"},
    1610612756: {"color": "#1D1160", "abbreviation": "PHX"},
    1610612757: {"color": "#E03A3E", "abbreviation": "POR"},
    1610612758: {"color": "#724C9F", "abbreviation": "SAC"},
    1610612759: {"color": "#BAC3C9", "abbreviation": "SAS"},
    1610612761: {"color": "#CE1141", "abbreviation": "TOR"},
    1610612762: {"color": "#00471B", "abbreviation": "UTA"},
    1610612764: {"color": "#002B5C", "abbreviation": "WAS"},
}
EVENTS_DIR = os.environ["EVENTS_DIR"]
TRACKING_DIR = os.environ["TRACKING_DIR"]
GAMES_DIR = os.environ["GAMES_DIR"]
DATA_DIR = os.environ["DATA_DIR"]
EXPERIMENTS_DIR = os.environ["EXPERIMENTS_DIR"]
