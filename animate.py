import os

from animator import Game
from settings import DATA_DIR, GAMES_DIR

gameid = "0021500622"
try:
    game = Game(DATA_DIR, GAMES_DIR, gameid)
except FileNotFoundError:
    home_dir = os.path.expanduser("~")
    DATA_DIR = f"{home_dir}/scratch"
    GAMES_DIR = f"{home_dir}/scratch"
    game = Game(DATA_DIR, GAMES_DIR, gameid)

# https://youtu.be/FRrh_WkyXko?t=109
start_period = 3
start_time = "1:55"
stop_period = 3
stop_time = "1:51"
game.show_seq(start_period, start_time, stop_period, stop_time)
