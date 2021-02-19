import os

from animator import Game

home_dir = os.path.expanduser("~")
DATA_DIR = f"{home_dir}/scratch"
GAMES_DIR = f"{home_dir}/scratch"
gameid = "0021500137"
fix_off = False
game = Game(DATA_DIR, GAMES_DIR, gameid)
start_period = 3
start_time = "10:45"
stop_period = 3
stop_time = "10:42"
game.show_seq(start_period, start_time, stop_period, stop_time)
