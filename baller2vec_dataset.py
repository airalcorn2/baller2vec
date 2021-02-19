import numpy as np
import torch

from settings import COURT_LENGTH, COURT_WIDTH, GAMES_DIR
from torch.utils.data import Dataset


class Baller2VecDataset(Dataset):
    def __init__(
        self,
        hz,
        secs,
        N,
        player_traj_n,
        max_player_move,
        ball_traj_n,
        max_ball_move,
        n_players,
        gameids,
        starts,
        mode,
        n_player_ids,
        filtered_player_idxs,
        next_score_change_time_max,
        n_time_to_next_score_change,
        n_ball_loc_x,
        n_ball_loc_y,
        ball_future_secs,
    ):
        # The raw data is recorded at 25 Hz.
        self.default_hz = 25
        self.hz = hz
        self.skip = self.default_hz // hz
        self.skip_secs = self.skip / self.default_hz
        self.secs = secs
        self.chunk_size = int(self.default_hz * secs)

        self.N = N
        self.n_players = n_players
        self.gameids = gameids
        self.starts = starts
        self.mode = mode
        self.n_player_ids = n_player_ids
        self.filtered_player_idxs = filtered_player_idxs

        self.player_traj_n = player_traj_n
        self.max_player_move = max_player_move
        self.player_traj_bins = np.linspace(
            -max_player_move, max_player_move, player_traj_n - 1
        )
        self.ball_traj_n = ball_traj_n
        self.ball_traj_bins = np.linspace(
            -max_ball_move, max_ball_move, ball_traj_n - 1
        )

        self.n_time_to_next_score_change = n_time_to_next_score_change
        self.time_to_next_score_change_bins = np.linspace(
            0, next_score_change_time_max, n_time_to_next_score_change - 1
        )
        max_score_change = 4
        self.n_score_change = 9
        self.next_score_change_bins = np.linspace(
            -max_score_change, max_score_change, self.n_score_change - 1
        )
        self.n_score_changes = n_time_to_next_score_change * self.n_score_change

        self.n_ball_loc_x = n_ball_loc_x
        self.n_ball_loc_y = n_ball_loc_y
        self.ball_loc_y_bins = np.linspace(0, COURT_WIDTH, n_ball_loc_y - 1)
        self.ball_loc_x_bins = np.linspace(0, COURT_LENGTH, n_ball_loc_x - 1)
        self.ball_loc_start = int(self.hz * ball_future_secs)

    def __len__(self):
        return self.N

    def get_sample(self, X, y, start):
        # Downsample.
        seq_data = X[start : start + self.chunk_size : self.skip]
        events = y[start : start + self.chunk_size : self.skip]

        # End sequence early if there is a position glitch. Often happens when there was
        # a break in the game, but glitches also happen for other reasons. See
        # glitch_example.py for an example.
        keep_players = np.random.choice(np.arange(10), self.n_players, False)
        player_xs = seq_data[:, 20:30][:, keep_players]
        player_ys = seq_data[:, 30:40][:, keep_players]
        player_x_diffs = np.diff(player_xs, axis=0)
        player_y_diffs = np.diff(player_ys, axis=0)
        try:
            glitch_x_break = np.where(
                np.abs(player_x_diffs) > 1.2 * self.max_player_move
            )[0].min()
        except ValueError:
            glitch_x_break = len(seq_data)

        try:
            glitch_y_break = np.where(
                np.abs(player_y_diffs) > 1.2 * self.max_player_move
            )[0].min()
        except ValueError:
            glitch_y_break = len(seq_data)

        glitch_break = min(glitch_x_break, glitch_y_break)
        seq_data = seq_data[:glitch_break]
        events = events[:glitch_break]

        periods = seq_data[:, 3].astype(int) - 1
        # Four overtimes (from this game --> https://www.nba.com/bulls/gameday/bulls-drop-4ot-thriller-pistons)
        # is the maximum in the dataset.
        one_hot_periods = np.identity(8)[periods]

        game_contexts = np.hstack([seq_data[:, :3], one_hot_periods, seq_data[:, 4:6]])
        ball_xs = seq_data[:, 7]
        ball_ys = seq_data[:, 8]
        ball_zs = seq_data[:, 9]
        player_idxs = seq_data[:, 10:20][:, keep_players].astype(int)
        player_xs = seq_data[:, 20:30][:, keep_players]
        player_ys = seq_data[:, 30:40][:, keep_players]
        player_hoop_sides = seq_data[:, 40:50][:, keep_players].astype(int)
        next_score_changes = seq_data[:, -2]

        # Randomly rotate the court because the hoop direction is arbitrary.
        if (self.mode == "train") and (np.random.random() < 0.5):
            player_xs = COURT_LENGTH - player_xs
            player_ys = COURT_WIDTH - player_ys
            player_hoop_sides = (player_hoop_sides + 1) % 2
            ball_xs = COURT_LENGTH - ball_xs
            ball_ys = COURT_WIDTH - ball_ys
            next_score_changes = -next_score_changes

        # Get player trajectories.
        player_x_diffs = np.diff(player_xs, axis=0)
        player_y_diffs = np.diff(player_ys, axis=0)

        player_traj_rows = np.digitize(player_y_diffs, self.player_traj_bins)
        player_traj_cols = np.digitize(player_x_diffs, self.player_traj_bins)
        player_trajs = player_traj_rows * self.player_traj_n + player_traj_cols

        # Get ball trajectories.
        ball_x_diffs = np.diff(ball_xs)
        ball_y_diffs = np.diff(ball_ys)
        ball_z_diffs = np.diff(ball_zs)

        ball_traj_rows = np.digitize(ball_z_diffs, self.ball_traj_bins)
        ball_traj_cols = np.digitize(ball_x_diffs, self.ball_traj_bins)
        ball_traj_deps = np.digitize(ball_y_diffs, self.ball_traj_bins)
        ball_trajs = (
            ball_traj_rows * self.ball_traj_n ** 2
            + ball_traj_cols * self.ball_traj_n
            + ball_traj_deps
        )

        # Substitute players that don't meet minimum playing time with generic player_idx.
        if (len(player_idxs) > 0) and (len(self.filtered_player_idxs) > 0):
            for player_idx in player_idxs[0]:
                if player_idx in self.filtered_player_idxs:
                    player_idxs[player_idxs == player_idx] = self.n_player_ids

        # Get score changes.
        time_to_next_score_changes = np.digitize(
            seq_data[:, -3], self.time_to_next_score_change_bins
        )
        next_score_changes = np.digitize(
            next_score_changes, self.next_score_change_bins
        )
        # # I think all of these are caused by gaps in the tracking data.
        # try:
        #     assert np.all(
        #         next_score_changes[
        #             time_to_next_score_changes < self.n_time_to_next_score_change
        #         ]
        #         < self.n_score_change
        #     )
        # except AssertionError:
        #     print(gameid, flush=True)
        #     print(set(seq_data[:, -2]), flush=True)
        #     print(set(next_score_changes), flush=True)
        #     raise AssertionError

        score_changes = (
            time_to_next_score_changes * self.n_score_change + next_score_changes
        )

        # Get ball position on court.
        ball_loc_rows = np.digitize(ball_ys, self.ball_loc_y_bins)
        ball_loc_cols = np.digitize(ball_xs, self.ball_loc_x_bins)
        ball_locs = ball_loc_rows * self.n_ball_loc_x + ball_loc_cols

        return {
            "player_idxs": torch.LongTensor(player_idxs[: glitch_break - 1]),
            "player_xs": torch.Tensor(player_xs[: glitch_break - 1]),
            "player_ys": torch.Tensor(player_ys[: glitch_break - 1]),
            "player_hoop_sides": torch.Tensor(player_hoop_sides[: glitch_break - 1]),
            "ball_xs": torch.Tensor(ball_xs[: glitch_break - 1]),
            "ball_ys": torch.Tensor(ball_ys[: glitch_break - 1]),
            "ball_zs": torch.Tensor(ball_zs[: glitch_break - 1]),
            "game_contexts": torch.Tensor(game_contexts[: glitch_break - 1]),
            "events": torch.LongTensor(events[: glitch_break - 1]),
            "player_trajs": torch.LongTensor(player_trajs),
            "ball_trajs": torch.LongTensor(ball_trajs),
            "score_changes": torch.LongTensor(score_changes[: glitch_break - 1]),
            "ball_locs": torch.LongTensor(ball_locs[self.ball_loc_start :]),
        }

    def __getitem__(self, idx):
        if self.mode == "train":
            gameid = np.random.choice(self.gameids)

        elif self.mode in {"valid", "test"}:
            gameid = self.gameids[idx]

        X = np.load(f"{GAMES_DIR}/{gameid}_X.npy")
        y = np.load(f"{GAMES_DIR}/{gameid}_y.npy")

        if self.mode == "train":
            start = np.random.randint(len(y) - self.chunk_size)

        elif self.mode in {"valid", "test"}:
            start = self.starts[idx]

        return self.get_sample(X, y, start)
