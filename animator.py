# Modified from: https://github.com/linouk23/NBA-Player-Movements.

import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib import animation
from settings import *


class Game:
    def __init__(self, data_dir, games_dir, gameid):
        baller2vec_config = pickle.load(
            open(f"{data_dir}/baller2vec_config.pydict", "rb")
        )
        self.player_idx2props = baller2vec_config["player_idx2props"]
        event2event_idx = baller2vec_config["event2event_idx"]
        self.event_idx2event = {
            event_idx: event for (event, event_idx) in event2event_idx.items()
        }

        X = np.load(f"{games_dir}/{gameid}_X.npy")
        self.y = np.load(f"{games_dir}/{gameid}_y.npy")

        self.periods = X[:, 3].astype(int)
        self.period_times = X[:, 1]
        self.shot_clocks = X[:, 2]

        self.left_scores = X[:, 4].astype(int)
        self.right_scores = X[:, 5].astype(int)

        self.player_idxs = X[:, 10:20].astype(int)

        self.ball_xs = X[:, 7]
        self.ball_ys = X[:, 8]
        self.ball_zs = X[:, 9]

        self.player_xs = X[:, 20:30]
        self.player_ys = X[:, 30:40]
        self.player_hoop_sides = X[:, 40:50]

    def update_radius(
        self,
        i,
        start,
        player_circles,
        ball_circle,
        annotations,
        clock_info,
        player_idx2circle_idx,
    ):
        time_step = start + i
        for (idx, player_idx) in enumerate(self.player_idxs[time_step]):
            circle_idx = player_idx2circle_idx[player_idx]
            player_circles[circle_idx].center = (
                self.player_xs[time_step, idx],
                Y_MAX - self.player_ys[time_step, idx],
            )
            annotations[circle_idx].set_position(player_circles[circle_idx].center)
            name = self.player_idx2props[player_idx]["name"].split()
            initials = name[0][0] + name[1][0]
            annotations[circle_idx].set_text(initials)
            if self.player_hoop_sides[time_step, idx]:
                player_circles[circle_idx].set_facecolor("white")
                player_circles[circle_idx].set_edgecolor("white")
            else:
                player_circles[circle_idx].set_facecolor("gray")
                player_circles[circle_idx].set_edgecolor("gray")

        clock_str = f"{self.periods[time_step]}/"
        period_time = int(720 - self.period_times[time_step])
        clock_str += f"{period_time // 60:02}:{period_time % 60:02}/"
        clock_str += f"{self.shot_clocks[time_step]:03.1f}/"
        clock_str += f"{self.left_scores[time_step]} - {self.right_scores[time_step]}/"
        clock_str += self.event_idx2event[self.y[time_step]]
        clock_info.set_text(clock_str)

        ball_circle.center = (self.ball_xs[time_step], Y_MAX - self.ball_ys[time_step])
        ball_circle.radius = self.ball_zs[time_step] / NORMALIZATION_COEF

        return (player_circles, ball_circle)

    def show_seq(
        self, start_period, start_time, stop_period, stop_time, save_gif=False
    ):
        # Leave some space for inbound passes.
        ax = plt.axes(xlim=(X_MIN, X_MAX), ylim=(Y_MIN, Y_MAX))
        ax.axis("off")
        fig = plt.gcf()
        # Remove grid.
        ax.grid(False)

        clock_info = ax.annotate(
            "",
            xy=[X_CENTER, Y_CENTER + 2],
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )

        (start_min, start_sec) = start_time.split(":")
        (start_min, start_sec) = (int(start_min), int(start_sec))
        period_time = 720 - (60 * start_min + start_sec)
        start = np.argwhere(
            (self.period_times > period_time) & (self.periods == start_period)
        ).min()

        (stop_min, stop_sec) = stop_time.split(":")
        (stop_min, stop_sec) = (int(stop_min), int(stop_sec))
        period_time = 720 - (60 * stop_min + stop_sec)
        stop = np.argwhere(
            (self.period_times > period_time) & (self.periods == stop_period)
        ).min()

        player_idxs = set(self.player_idxs[start])
        for time_step in range(start + 1, stop):
            # End sequence early at lineup change.
            if len(player_idxs & set(self.player_idxs[time_step])) != 10:
                stop = time_step
                break

        annotations = []
        player_idx2circle_idx = {}
        team_a_players = []
        team_b_players = []
        player_circles = []
        for (circle_idx, player_idx) in enumerate(self.player_idxs[start]):
            player_idx2circle_idx[player_idx] = circle_idx
            name = self.player_idx2props[player_idx]["name"].split()
            initials = name[0][0] + name[1][0]
            annotations.append(
                ax.annotate(
                    initials,
                    xy=[0, 0],
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontweight="bold",
                )
            )
            if self.player_hoop_sides[start, circle_idx]:
                team_a_players.append(
                    self.player_idx2props[player_idx]["name"] + f": {initials}"
                )
                player_circles.append(
                    plt.Circle((0, 0), PLAYER_CIRCLE_SIZE, color="white")
                )
            else:
                team_b_players.append(
                    self.player_idx2props[player_idx]["name"] + f": {initials}"
                )
                player_circles.append(
                    plt.Circle((0, 0), PLAYER_CIRCLE_SIZE, color="gray")
                )

        # Prepare table.
        column_labels = tuple(["Team A", "Team B"])
        players_data = list(zip(team_a_players, team_b_players))

        table = plt.table(
            cellText=players_data,
            colLabels=column_labels,
            colWidths=[COL_WIDTH, COL_WIDTH],
            loc="bottom",
            fontsize=FONTSIZE,
            cellLoc="center",
        )
        table.scale(1, SCALE)

        # Add animated objects.
        for circle in player_circles:
            ax.add_patch(circle)

        ball_circle = plt.Circle((0, 0), PLAYER_CIRCLE_SIZE, color=BALL_COLOR)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
            fig,
            self.update_radius,
            fargs=(
                start,
                player_circles,
                ball_circle,
                annotations,
                clock_info,
                player_idx2circle_idx,
            ),
            frames=stop - start,
            interval=INTERVAL,
        )
        court = plt.imread("court.png")
        plt.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
        if save_gif:
            anim.save("animation.mp4", writer="imagemagick", fps=25)

        plt.show()
