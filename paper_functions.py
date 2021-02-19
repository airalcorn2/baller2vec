import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pickle
import shutil
import torch
import yaml

from PIL import Image, ImageDraw
from settings import *
from sklearn.metrics import pairwise_distances
from torch import nn
from train_baller2vec import init_datasets, init_model

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def single_frame_loss():
    JOB = "20210124114528"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    device = torch.device("cuda:0")

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    ) = init_datasets(opts)
    model = init_model(opts, train_dataset)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss_one = 0.0
    test_loss_all = 0.0
    n_test = 0
    with torch.no_grad():
        for test_tensors in test_loader:
            # Skip bad sequences.
            if len(test_tensors["player_idxs"]) < model.seq_len:
                continue

            player_trajs = test_tensors["player_trajs"].flatten()
            n_player_trajs = len(player_trajs)
            labels = player_trajs.to(device)
            preds = model(test_tensors)["player"][:n_player_trajs]

            test_loss_one += criterion(preds[:10], labels[:10]).item()
            test_loss_all += criterion(preds, labels).item()

            n_test += 1

    print(test_loss_one / n_test)
    print(test_loss_all / n_test)


def naive_player_test_set_perplexity():
    JOB = "20210124114528"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    device = torch.device("cuda:0")

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    ) = init_datasets(opts)

    default_hz = 25
    hz = 5
    skip = default_hz // hz
    skip_secs = 1 / skip
    all_player_trajs = []
    for gameid in train_dataset.gameids:
        X = np.load(f"{GAMES_DIR}/{gameid}_X.npy")

        seq_data = X[::skip]

        player_xs = seq_data[:, 20:30]
        player_ys = seq_data[:, 30:40]
        player_x_diffs = np.diff(player_xs, axis=0)
        player_y_diffs = np.diff(player_ys, axis=0)

        wall_clock_diffs = np.diff(seq_data[:, -1]) / 1000

        keep_diffs = wall_clock_diffs <= 1.5 * skip_secs
        player_x_diffs = player_x_diffs[keep_diffs].flatten()
        player_y_diffs = player_y_diffs[keep_diffs].flatten()

        player_traj_rows = np.digitize(player_y_diffs, train_dataset.player_traj_bins)
        player_traj_cols = np.digitize(player_x_diffs, train_dataset.player_traj_bins)
        player_trajs = player_traj_rows * train_dataset.player_traj_n + player_traj_cols
        all_player_trajs.append(player_trajs)

    all_player_trajs = np.concatenate(all_player_trajs)
    (unique, counts) = np.unique(all_player_trajs, return_counts=True)
    d_counts = dict(zip(unique, counts))
    probs = np.zeros(train_dataset.ball_traj_n ** 3)
    for (traj, count) in d_counts.items():
        probs[traj] = count

    probs = probs / probs.sum()

    model = init_model(opts, train_dataset)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    n_test = 0
    with torch.no_grad():
        for test_tensors in test_loader:
            # Skip bad sequences.
            if len(test_tensors["player_idxs"]) < model.seq_len:
                continue

            labels = test_tensors["player_trajs"].flatten()
            test_loss -= np.log(probs[labels]).mean()
            n_test += 1

    print(test_loss)
    print(np.exp(test_loss / n_test))


def naive_ball_test_set_perplexity():
    JOB = "20210124114950"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    device = torch.device("cuda:0")

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    ) = init_datasets(opts)

    default_hz = 25
    hz = 5
    skip = default_hz // hz
    skip_secs = 1 / skip
    all_ball_trajs = []
    for gameid in train_dataset.gameids:
        X = np.load(f"{GAMES_DIR}/{gameid}_X.npy")

        seq_data = X[::skip]

        ball_xs = seq_data[:, 7]
        ball_ys = seq_data[:, 8]
        ball_zs = seq_data[:, 9]

        ball_x_diffs = np.diff(ball_xs)
        ball_y_diffs = np.diff(ball_ys)
        ball_z_diffs = np.diff(ball_zs)

        wall_clock_diffs = np.diff(seq_data[:, -1]) / 1000

        keep_diffs = wall_clock_diffs <= 1.5 * skip_secs
        ball_x_diffs = ball_x_diffs[keep_diffs]
        ball_y_diffs = ball_y_diffs[keep_diffs]
        ball_z_diffs = ball_z_diffs[keep_diffs]

        ball_traj_rows = np.digitize(ball_z_diffs, train_dataset.ball_traj_bins)
        ball_traj_cols = np.digitize(ball_x_diffs, train_dataset.ball_traj_bins)
        ball_traj_deps = np.digitize(ball_y_diffs, train_dataset.ball_traj_bins)
        ball_trajs = (
            ball_traj_rows * train_dataset.ball_traj_n ** 2
            + ball_traj_cols * train_dataset.ball_traj_n
            + ball_traj_deps
        )
        all_ball_trajs.append(ball_trajs)

    all_ball_trajs = np.concatenate(all_ball_trajs)
    (unique, counts) = np.unique(all_ball_trajs, return_counts=True)
    d_counts = dict(zip(unique, counts))
    probs = np.zeros(train_dataset.ball_traj_n ** 3)
    for (traj, count) in d_counts.items():
        probs[traj] = count

    probs += 1
    probs = probs / probs.sum()

    model = init_model(opts, train_dataset)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    n_test = 0
    with torch.no_grad():
        for test_tensors in test_loader:
            # Skip bad sequences.
            if len(test_tensors["player_idxs"]) < model.seq_len:
                continue

            labels = test_tensors["ball_trajs"]
            test_loss -= np.log(probs[labels]).mean()
            n_test += 1

    print(test_loss)
    print(np.exp(test_loss / n_test))


def plot_player_traj_dist():
    gameids = list(set([np_f.split("_")[0] for np_f in os.listdir(GAMES_DIR)]))

    default_hz = 25
    hz = 5
    skip = default_hz // hz
    skip_secs = 1 / skip
    all_player_x_diffs = []
    all_player_y_diffs = []
    for gameid in gameids:
        X = np.load(f"{GAMES_DIR}/{gameid}_X.npy")

        seq_data = X[::skip]

        player_xs = seq_data[:, 20:30]
        player_ys = seq_data[:, 30:40]
        player_x_diffs = np.diff(player_xs, axis=0)
        player_y_diffs = np.diff(player_ys, axis=0)

        game_clock_diffs = np.diff(seq_data[:, 0])

        keep_diffs = game_clock_diffs <= 1.2 * skip_secs
        all_player_x_diffs.append(player_x_diffs[keep_diffs].flatten())
        all_player_y_diffs.append(player_y_diffs[keep_diffs].flatten())

    all_player_x_diffs = np.concatenate(all_player_x_diffs)
    all_player_y_diffs = np.concatenate(all_player_y_diffs)

    max_player_move = 4.5 + 1
    player_traj_n = 11 + 2
    player_traj_bins = np.linspace(-max_player_move, max_player_move, player_traj_n - 1)
    (heatmap, xedges, yedges) = np.histogram2d(
        all_player_x_diffs, all_player_y_diffs, player_traj_bins
    )
    probs = (heatmap / heatmap.sum()).flatten()
    entropy = -(probs * np.log(probs)).sum()
    pp = np.exp(entropy)
    print(pp)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap += 1
    norm = matplotlib.colors.LogNorm(vmin=heatmap.min(), vmax=heatmap.max())

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/test", exist_ok=True)
    plt.imshow(heatmap.T, extent=extent, origin="lower", norm=norm)
    plt.savefig(f"{home_dir}/test/player_traj_heat_map.png")
    plt.clf()
    shutil.make_archive(f"{home_dir}/test", "zip", f"{home_dir}/test")
    shutil.rmtree(f"{home_dir}/test")


def get_player_idx(name, player_idx2props):
    for (player_idx, props) in player_idx2props.items():
        if props["name"] == name:
            return player_idx


def get_nearest_neighbors():
    JOB = "20210124114950"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    train_dataset = init_datasets(opts)[0]
    model = init_model(opts, train_dataset)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    player_embeddings = model.player_embedding.weight
    with torch.no_grad():
        if opts["model"]["sigmoid"] == "logistic":
            player_embeddings = torch.sigmoid(player_embeddings)
        elif opts["model"]["sigmoid"] == "tanh":
            player_embeddings = torch.tanh(player_embeddings)

    player_embeddings = player_embeddings.detach().numpy()
    all_player_dists = {
        "cosine": pairwise_distances(player_embeddings, metric="cosine"),
        "euclidean": pairwise_distances(player_embeddings, metric="euclidean"),
    }

    players = [
        "LeBron James",
        "Stephen Curry",
        "Russell Westbrook",
        "James Harden",
        "Pau Gasol",
        "Anthony Davis",
        "Serge Ibaka",
        "Giannis Antetokounmpo",
        "Kawhi Leonard",
        "Jimmy Butler",
    ]
    k = 20
    min_playing_time = 60000  # 0/13314/39917/60000 --> 100%/75%/50%/starters.
    baller2vec_config = pickle.load(open(f"{DATA_DIR}/baller2vec_config.pydict", "rb"))
    player_idx2props = baller2vec_config["player_idx2props"]
    for player in players:
        print(f"\n{player}")
        player_idx = get_player_idx(player, player_idx2props)
        for metric in ["cosine", "euclidean"]:
            print(f"\n{metric}\n")
            player_dists = all_player_dists[metric]
            sim_player_idxs = np.argsort(player_dists[player_idx])
            neighbors = 0
            for sim_player_idx in sim_player_idxs[1:]:
                if sim_player_idx in player_idx2props:
                    if "playing_time" not in player_idx2props[sim_player_idx]:
                        continue

                    if (
                        player_idx2props[sim_player_idx]["playing_time"]
                        < min_playing_time
                    ):
                        continue

                sim_player_score = player_dists[player_idx, sim_player_idx]
                name = player_idx2props.get(sim_player_idx, {"name": "Baller"})["name"]
                print(f"{name}: {sim_player_score:.4f}")
                neighbors += 1
                if neighbors == k:
                    break

    keep_embeddings = []
    player_names = []
    for (player_idx, player_embedding) in enumerate(player_embeddings):
        if player_idx != len(player_embeddings) - 1:
            if "playing_time" not in player_idx2props[player_idx]:
                continue

            if player_idx2props[player_idx]["playing_time"] < min_playing_time:
                continue

        keep_embeddings.append(player_embedding)
        if player_idx == len(player_embeddings) - 1:
            player_names.append("Baller")
        else:
            player_names.append(player_idx2props[player_idx]["name"])

    keep_embeddings = np.vstack(keep_embeddings)

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/test", exist_ok=True)
    np.savetxt(f"{home_dir}/test/embeddings.tsv", keep_embeddings, delimiter="\t")
    with open(f"{home_dir}/test/names.tsv", "w") as f:
        print("\n".join(player_names), file=f)

    shutil.make_archive(f"{home_dir}/test", "zip", f"{home_dir}/test")
    shutil.rmtree(f"{home_dir}/test")


def plot_umap():
    import matplotlib as mpl
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import umap

    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    embeddings = pd.read_csv("embeddings.tsv", delimiter="\t", header=None).iloc[:-1]
    names = pd.read_csv("names.tsv", delimiter="\t", header=None)[0].tolist()[:-1]

    keep_players = {
        "LeBron James",
        "Stephen Curry",
        "Russell Westbrook",
        "Anthony Davis",
        "Serge Ibaka",
        "Giannis Antetokounmpo",
        "Kawhi Leonard",
        "Dirk Nowitzki",
        "Chris Paul",
    }

    keep_colors = []
    for (player_idx, name) in enumerate(names):
        if name in keep_players:
            keep_colors.append("tomato")
        else:
            keep_colors.append("slategray")

    reducer = umap.UMAP(n_neighbors=4, metric="euclidean", n_epochs=500)
    umap_embeddings = reducer.fit_transform(embeddings)

    player_stats = pd.read_csv("player_stats.csv")
    player_stats["Player"] = player_stats["Player"].apply(lambda x: x.split("\\")[0])
    player_stats["Pos"] = player_stats["Pos"].apply(lambda x: x.split("-")[0])

    pos2marker = {"C": "^", "PF": "s", "SF": "o", "SG": "*", "PG": "P"}

    # Make sure all the player names match.
    print(set(names) - set(player_stats["Player"]))
    stat = "BLK"  # "3P%▼"
    name2minutes = dict(zip(player_stats["Player"], player_stats["MP"]))
    name2pos = dict(zip(player_stats["Player"], player_stats["Pos"]))
    name2stat = dict(zip(player_stats["Player"], player_stats[stat]))

    per_min = True
    stat_colors = []
    markers = []
    for (player_idx, name) in enumerate(names):
        player_stat = name2stat[name]
        if per_min:
            player_stat /= name2minutes[name]

        stat_colors.append(player_stat)
        markers.append(pos2marker[name2pos[name]])

    stat_colors = np.array(stat_colors)
    stat_colors = np.nan_to_num(stat_colors, 0.0)
    stat_colors = stat_colors ** (1 / 3)

    cmap = sns.cubehelix_palette(as_cmap=True)
    (fig, ax) = plt.subplots()
    (x, y) = (umap_embeddings[:, 0], umap_embeddings[:, 1])
    ax.axis("off")

    (vmin, vmax) = (stat_colors.min(), stat_colors.max())
    for (player_idx, name) in enumerate(names):
        s = ax.scatter(
            [x[player_idx]],
            [y[player_idx]],
            c=stat_colors[player_idx],
            marker=markers[player_idx],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if name in keep_players:
            ax.annotate(name, (x[player_idx], y[player_idx]), fontsize=13)

    cbar = plt.colorbar(mappable=s, ax=ax)
    cbar.set_label(r"$\sqrt[3]{\text{Blocks Per Minute}}$", rotation=270, labelpad=15)

    legend_labels = []
    abb2pos = {
        "C": "Center",
        "PF": "Power Forward",
        "SF": "Small Forward",
        "SG": "Shooting Guard",
        "PG": "Point Guard",
    }
    for (abb, pos) in abb2pos.items():
        legend_labels.append(
            mlines.Line2D(
                [],
                [],
                color="white",
                markeredgecolor="black",
                marker=pos2marker[abb],
                linestyle="None",
                markersize=10,
                label=pos,
            )
        )

    plt.legend(handles=legend_labels)
    plt.tight_layout()
    plt.show()


def add_grid(img, steps):
    # See: https://randomgeekery.org/post/2017/11/drawing-grids-with-python-and-pillow/.
    fill = (128, 128, 128)

    draw = ImageDraw.Draw(img)
    y_start = 0
    y_end = img.height
    step_size = int(img.width / steps)

    for x in range(0, img.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=fill)

    x_start = 0
    x_end = img.width

    for y in range(0, img.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=fill)

    mid_color = (0, 0, 255)
    half_steps = steps // 2
    (mid_start, mid_end) = (half_steps * step_size, half_steps * step_size + step_size)
    draw.line(((mid_start, mid_start), (mid_end, mid_start)), fill=mid_color)
    draw.line(((mid_start, mid_end), (mid_end, mid_end)), fill=mid_color)
    draw.line(((mid_start, mid_start), (mid_start, mid_end)), fill=mid_color)
    draw.line(((mid_end, mid_start), (mid_end, mid_end)), fill=mid_color)

    del draw


def plot_traj_preds():
    JOB = "20210124114528"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (train_dataset, _, _, _, test_dataset, _) = init_datasets(opts)
    model = init_model(opts, train_dataset)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    cand_test_idxs = []
    for test_idx in range(len(test_dataset.gameids)):
        tensors = test_dataset[test_idx]
        if len(tensors["player_idxs"]) == model.seq_len:
            cand_test_idxs.append(test_idx)

    court = plt.imread("court.png")
    width = 5
    height = width * court.shape[0] / float(court.shape[1])
    vert_buffer = 10
    pt_scale = 1.4
    scale = 8
    pred_player_idx = 5

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/test", exist_ok=True)
    (fig, ax) = plt.subplots(figsize=(width, height))

    saved = 0
    np.random.seed(2010)
    while saved < 20:
        test_idx = np.random.choice(cand_test_idxs)
        cand_test_idxs.remove(test_idx)
        print(test_idx, flush=True)

        tensors = test_dataset[test_idx]
        player_traj_n = test_dataset.player_traj_n
        with torch.no_grad():
            seq_len = len(tensors["player_idxs"])
            labels = tensors["player_trajs"].flatten()
            preds = model(tensors)["player"][: len(labels)]
            probs = torch.softmax(preds, dim=1)
            pred_player_probs = probs.view(seq_len, -1, player_traj_n ** 2)[
                :, pred_player_idx
            ]

        colors = []
        markers = []
        for p_idx in range(10):
            if p_idx == pred_player_idx:
                colors.append("r")
            elif tensors["player_hoop_sides"][0, p_idx]:
                colors.append("white")
            else:
                colors.append("gray")

            if tensors["player_hoop_sides"][0, p_idx]:
                markers.append("s")
            else:
                markers.append("^")

        for time_step in range(len(tensors["player_idxs"])):
            ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
            ax.axis("off")
            ax.grid(False)

            for p_idx in range(10):
                ax.scatter(
                    [tensors["player_xs"][time_step, p_idx]],
                    [tensors["player_ys"][time_step, p_idx]],
                    s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                    c=colors[p_idx],
                    marker=markers[p_idx],
                    edgecolors="black",
                )

            plt.xlim(auto=False)
            plt.ylim(auto=False)
            ax.plot(
                tensors["player_xs"][: time_step + 1, pred_player_idx],
                tensors["player_ys"][: time_step + 1, pred_player_idx],
                c="red",
            )
            ax.plot(
                tensors["player_xs"][time_step:, pred_player_idx],
                tensors["player_ys"][time_step:, pred_player_idx],
                linestyle=":",
                c="red",
            )
            ax.scatter(
                [tensors["ball_xs"][time_step]],
                [tensors["ball_ys"][time_step]],
                s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                c="#ff8c00",
                edgecolors="black",
            )

            plt.subplots_adjust(0, 0, 1, 1)
            fig.canvas.draw()
            court_img = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            plt.cla()

            traj_data = np.full((player_traj_n, player_traj_n, 3), 255, dtype=np.uint8)
            player_traj = tensors["player_trajs"][time_step, pred_player_idx]
            traj_data[player_traj // player_traj_n, player_traj % player_traj_n] = 0
            traj_img = Image.fromarray(traj_data).resize(
                (scale * player_traj_n, scale * player_traj_n), resample=0
            )
            add_grid(traj_img, player_traj_n)

            traj_pred = np.zeros((player_traj_n, player_traj_n, 3))
            traj_pred[:, :] = pred_player_probs[time_step].view(
                player_traj_n, player_traj_n, 1
            )
            pred_img = Image.fromarray(np.uint8(255 - 255 * traj_pred)).resize(
                (scale * player_traj_n, scale * player_traj_n), resample=0
            )
            add_grid(pred_img, player_traj_n)

            final_img = Image.new(
                "RGBA",
                (
                    court_img.size[0],
                    court_img.size[1] + scale * player_traj_n + vert_buffer,
                ),
            )
            final_img.paste(court_img, (0, 0))
            full_width = 3 * traj_img.size[0]
            horiz_buffer = court_img.size[0] // 2 - full_width // 2
            final_img.paste(traj_img, (horiz_buffer, court_img.size[1] + vert_buffer))
            final_img.paste(
                pred_img,
                (2 * traj_img.size[0] + horiz_buffer, court_img.size[1] + vert_buffer),
            )
            time_step = str(time_step).zfill(2)
            final_img.save(f"{home_dir}/test/{test_idx}_{time_step}.png")

        saved += 1

    shutil.make_archive(f"{home_dir}/test", "zip", f"{home_dir}/test")
    shutil.rmtree(f"{home_dir}/test")


def plot_generated_trajectories():
    JOB = "20210124114528"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (train_dataset, _, _, _, test_dataset, _) = init_datasets(opts)
    model = init_model(opts, train_dataset)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    cand_test_idxs = []
    for test_idx in range(len(test_dataset.gameids)):
        tensors = test_dataset[test_idx]
        if len(tensors["player_idxs"]) == model.seq_len:
            cand_test_idxs.append(test_idx)

    court = plt.imread("court.png")
    width = 5
    height = width * court.shape[0] / float(court.shape[1])
    pt_scale = 1.4
    player_traj_bins = np.array(list(test_dataset.player_traj_bins) + [5.5])

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/test", exist_ok=True)
    (fig, ax) = plt.subplots(figsize=(width, height))

    saved = 0
    np.random.seed(2010)
    while saved < 20:
        test_idx = np.random.choice(cand_test_idxs)
        cand_test_idxs.remove(test_idx)
        print(test_idx, flush=True)

        for start_time_step in [0, 5, 9, 19]:
            tensors = test_dataset[test_idx]
            (player_idxs, p_idxs) = tensors["player_idxs"][0].sort()
            player_traj_n = test_dataset.player_traj_n
            seq_len = len(tensors["player_idxs"])
            grid_gap = np.diff(test_dataset.player_traj_bins)[-1] / 2
            with torch.no_grad():
                for step in range(seq_len - start_time_step - 1):
                    preds_start = 10 * (start_time_step + step)
                    preds_stop = preds_start + 10
                    preds = model(tensors)["player"][preds_start:preds_stop]
                    probs = torch.softmax(preds, dim=1)
                    samp_trajs = torch.multinomial(probs, 1)
                    samp_rows = samp_trajs // player_traj_n
                    samp_cols = samp_trajs % player_traj_n
                    samp_xs = (
                        player_traj_bins[samp_cols]
                        - grid_gap
                        + np.random.uniform(-grid_gap, grid_gap)
                    )
                    samp_ys = (
                        player_traj_bins[samp_rows]
                        - grid_gap
                        + np.random.uniform(-grid_gap, grid_gap)
                    )
                    tensors["player_xs"][start_time_step + step + 1] = (
                        tensors["player_xs"][start_time_step + step] + samp_xs.flatten()
                    )
                    tensors["player_ys"][start_time_step + step + 1] = (
                        tensors["player_ys"][start_time_step + step] + samp_ys.flatten()
                    )

            markers = []
            for p_idx in range(10):
                if tensors["player_hoop_sides"][0, p_idx]:
                    markers.append("s")
                else:
                    markers.append("^")

            ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
            ax.axis("off")
            ax.grid(False)
            for (idx, p_idx) in enumerate(p_idxs):
                ax.scatter(
                    [tensors["player_xs"][0, p_idx]],
                    [tensors["player_ys"][0, p_idx]],
                    s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                    c=colors[idx],
                    marker=markers[p_idx],
                    edgecolors="black",
                )
                plt.xlim(auto=False)
                plt.ylim(auto=False)
                ax.plot(
                    tensors["player_xs"][: start_time_step + 1, p_idx],
                    tensors["player_ys"][: start_time_step + 1, p_idx],
                    c=colors[idx],
                )
                ax.plot(
                    tensors["player_xs"][start_time_step:, p_idx],
                    tensors["player_ys"][start_time_step:, p_idx],
                    c=colors[idx],
                    linestyle=":",
                )

            ax.plot(
                tensors["ball_xs"],
                tensors["ball_ys"],
                c="#ff8c00",
            )
            ax.scatter(
                [tensors["ball_xs"][0]],
                [tensors["ball_ys"][0]],
                s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                c="#ff8c00",
                edgecolors="black",
            )

            plt.subplots_adjust(0, 0, 1, 1)
            fig.canvas.draw()
            court_img = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            plt.cla()
            court_img.save(
                f"{home_dir}/test/trajectories_{test_idx}_{start_time_step}.png"
            )

        saved += 1

    shutil.make_archive(f"{home_dir}/test", "zip", f"{home_dir}/test")
    shutil.rmtree(f"{home_dir}/test")


def plot_attn():
    JOB = "20210124114950"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (train_dataset, _, _, _, test_dataset, _) = init_datasets(opts)
    model = init_model(opts, train_dataset)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    court = plt.imread("court.png")
    width = 5
    height = width * court.shape[0] / float(court.shape[1])
    pt_scale = 1.4
    time_step = 11

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/test", exist_ok=True)
    (fig, ax) = plt.subplots(figsize=(width, height))

    cand_test_idxs = []
    for test_idx in range(len(test_dataset.gameids)):
        tensors = test_dataset[test_idx]
        if len(tensors["player_idxs"]) == model.seq_len:
            cand_test_idxs.append(test_idx)

    saved = 0
    np.random.seed(2010)
    while saved < 20:
        print(test_idx, flush=True)
        save_output = SaveOutput()
        hook_handles = []
        for layer in model.transformer.layers:
            hook_handles.append(layer.self_attn.register_forward_hook(save_output))

        tensors = test_dataset[test_idx]
        with torch.no_grad():
            _ = model(tensors)

        colors = []
        markers = []
        for p_idx in range(10):
            if tensors["player_hoop_sides"][0, p_idx]:
                colors.append("white")
                markers.append("s")
            else:
                colors.append("gray")
                markers.append("^")

        ball_start = 10 * len(tensors["player_idxs"])
        ball_pos = ball_start + time_step

        ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
        ax.axis("off")
        ax.grid(False)
        for p_idx in range(10):
            ax.scatter(
                [tensors["player_xs"][time_step, p_idx]],
                [tensors["player_ys"][time_step, p_idx]],
                s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                c=colors[p_idx],
                marker=markers[p_idx],
                edgecolors="black",
            )

        ax.scatter(
            [tensors["ball_xs"][time_step]],
            [tensors["ball_ys"][time_step]],
            s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
            c="#ff8c00",
            edgecolors="black",
        )
        plt.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        court_img = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.cla()
        court_img.save(f"{home_dir}/test/court_{test_idx}.png")

        for attn_layer_idx in range(len(save_output.outputs)):
            attn_scores = (
                save_output.outputs[attn_layer_idx][1][0].detach().cpu().numpy()
            )
            for attn_head_idx in range(len(attn_scores)):
                attn_head_scores = attn_scores[attn_head_idx, ball_pos]
                player_scores = (
                    attn_head_scores[:ball_start].reshape(-1, 10).sum(axis=0)
                )
                vmax = player_scores.max()
                player_scores_normed = player_scores / vmax
                player_scores_normed = np.nan_to_num(player_scores_normed, 0)
                player_colors = [
                    str(player_score) for player_score in player_scores_normed
                ]
                ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
                ax.axis("off")
                ax.grid(False)

                for p_idx in range(10):
                    ax.scatter(
                        [tensors["player_xs"][time_step, p_idx]],
                        [tensors["player_ys"][time_step, p_idx]],
                        s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                        c=player_colors[p_idx],
                        marker=markers[p_idx],
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                        edgecolors="black",
                    )

                ball_score = attn_head_scores[ball_start:].sum()
                ax.scatter(
                    [tensors["ball_xs"][time_step]],
                    [tensors["ball_ys"][time_step]],
                    s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                    c="#ff8c00",
                    edgecolors="black",
                )

                plt.subplots_adjust(0, 0, 1, 1)
                fig.canvas.draw()
                court_img = Image.frombytes(
                    "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
                )
                plt.cla()

                ball_score = str(ball_score).split(".")[1][:2]
                img_id = f"{str(attn_layer_idx).zfill(2)}_{str(attn_head_idx).zfill(2)}"
                court_img.save(f"{home_dir}/test/{img_id}_{test_idx}_{ball_score}.png")

        saved += 1

    shutil.make_archive(f"{home_dir}/test", "zip", f"{home_dir}/test")
    shutil.rmtree(f"{home_dir}/test")


def plot_attn_through_time():
    JOB = "20210124114950"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (train_dataset, _, _, _, test_dataset, _) = init_datasets(opts)
    model = init_model(opts, train_dataset)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    court = plt.imread("court.png")
    width = 5
    height = width * court.shape[0] / float(court.shape[1])
    pt_scale = 1.4

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/test", exist_ok=True)
    (fig, ax) = plt.subplots(figsize=(width, height))

    save_output = SaveOutput()
    hook_handles = []
    for layer in model.transformer.layers:
        hook_handles.append(layer.self_attn.register_forward_hook(save_output))

    test_idx = 851
    tensors = test_dataset[test_idx]
    with torch.no_grad():
        _ = model(tensors)

    colors = []
    markers = []
    for p_idx in range(10):
        if tensors["player_hoop_sides"][0, p_idx]:
            colors.append("white")
            markers.append("s")
        else:
            colors.append("gray")
            markers.append("^")

    ball_start = 10 * len(tensors["player_idxs"])

    for time_step in range(19):
        print(time_step, flush=True)
        ball_pos = ball_start + time_step

        ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
        ax.axis("off")
        ax.grid(False)
        for p_idx in range(10):
            ax.scatter(
                [tensors["player_xs"][time_step, p_idx]],
                [tensors["player_ys"][time_step, p_idx]],
                s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                c=colors[p_idx],
                marker=markers[p_idx],
                edgecolors="black",
            )

        ax.scatter(
            [tensors["ball_xs"][time_step]],
            [tensors["ball_ys"][time_step]],
            s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
            c="#ff8c00",
            edgecolors="black",
        )
        plt.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        court_img = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.cla()
        time_step = str(time_step).zfill(2)
        court_img.save(f"{home_dir}/test/court_{time_step}.png")

        for attn_layer_idx in range(len(save_output.outputs)):
            attn_scores = (
                save_output.outputs[attn_layer_idx][1][0].detach().cpu().numpy()
            )
            for attn_head_idx in range(len(attn_scores)):
                attn_head_scores = attn_scores[attn_head_idx, ball_pos]
                player_scores = (
                    attn_head_scores[:ball_start].reshape(-1, 10).sum(axis=0)
                )
                vmax = player_scores.max()
                player_scores_normed = player_scores / vmax
                player_scores_normed = np.nan_to_num(player_scores_normed, 0)
                player_colors = [
                    str(player_score) for player_score in player_scores_normed
                ]
                ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
                ax.axis("off")
                ax.grid(False)

                for p_idx in range(10):
                    ax.scatter(
                        [tensors["player_xs"][time_step, p_idx]],
                        [tensors["player_ys"][time_step, p_idx]],
                        s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                        c=player_colors[p_idx],
                        marker=markers[p_idx],
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                        edgecolors="black",
                    )

                ball_score = attn_head_scores[ball_start:].sum()
                ax.scatter(
                    [tensors["ball_xs"][time_step]],
                    [tensors["ball_ys"][time_step]],
                    s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                    c="#ff8c00",
                    edgecolors="black",
                )

                plt.subplots_adjust(0, 0, 1, 1)
                fig.canvas.draw()
                court_img = Image.frombytes(
                    "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
                )
                plt.cla()

                ball_score = str(ball_score).split(".")[1][:2]
                img_id = f"{str(attn_layer_idx).zfill(2)}_{str(attn_head_idx).zfill(2)}"
                time_step = str(time_step).zfill(2)
                court_img.save(f"{home_dir}/test/{img_id}_{time_step}_{ball_score}.png")

    shutil.make_archive(f"{home_dir}/test", "zip", f"{home_dir}/test")
    shutil.rmtree(f"{home_dir}/test")


def plot_specific_attn_through_time():
    JOB = "20210124114950"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (train_dataset, _, _, _, test_dataset, _) = init_datasets(opts)
    model = init_model(opts, train_dataset)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    court = plt.imread("court.png")
    width = 5
    height = width * court.shape[0] / float(court.shape[1])
    pt_scale = 1.4
    ball_threshold = 1 / 3

    attn_layer_idx = 1
    attn_head_idx = 6

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/test", exist_ok=True)
    (fig, ax) = plt.subplots(figsize=(width, height))

    cand_test_idxs = []
    for test_idx in range(len(test_dataset.gameids)):
        tensors = test_dataset[test_idx]
        if len(tensors["player_idxs"]) == model.seq_len:
            cand_test_idxs.append(test_idx)

    saved = 0
    np.random.seed(2010)
    while saved < 20:
        test_idx = np.random.choice(cand_test_idxs)
        cand_test_idxs.remove(test_idx)
        print(test_idx, flush=True)
        save_output = SaveOutput()
        hook_handles = []
        for layer in model.transformer.layers:
            hook_handles.append(layer.self_attn.register_forward_hook(save_output))

        tensors = test_dataset[test_idx]
        if len(tensors["player_idxs"]) < model.seq_len:
            test_idx += 1
            continue

        with torch.no_grad():
            _ = model(tensors)

        colors = []
        markers = []
        for p_idx in range(10):
            if tensors["player_hoop_sides"][0, p_idx]:
                colors.append("white")
                markers.append("s")
            else:
                colors.append("gray")
                markers.append("^")

        ball_start = 10 * len(tensors["player_idxs"])

        for time_step in range(19):
            ball_pos = ball_start + time_step

            ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
            ax.axis("off")
            ax.grid(False)
            for p_idx in range(10):
                ax.scatter(
                    [tensors["player_xs"][time_step, p_idx]],
                    [tensors["player_ys"][time_step, p_idx]],
                    s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                    c=colors[p_idx],
                    marker=markers[p_idx],
                    edgecolors="black",
                )

            ax.scatter(
                [tensors["ball_xs"][time_step]],
                [tensors["ball_ys"][time_step]],
                s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                c="#ff8c00",
                edgecolors="black",
            )
            plt.subplots_adjust(0, 0, 1, 1)
            fig.canvas.draw()
            court_img = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            plt.cla()
            time_step_str = str(time_step).zfill(2)
            court_img.save(f"{home_dir}/test/court_{test_idx}_{time_step_str}.png")

            attn_scores = (
                save_output.outputs[attn_layer_idx][1][0].detach().cpu().numpy()
            )
            attn_head_scores = attn_scores[attn_head_idx, ball_pos]
            player_scores = attn_head_scores[:ball_start].reshape(-1, 10).sum(axis=0)
            vmax = player_scores.max()
            player_scores_normed = player_scores / vmax
            player_scores_normed = np.nan_to_num(player_scores_normed, 0)
            ball_score = attn_head_scores[ball_start:].sum()
            if ball_score > ball_threshold:
                player_scores_normed[:] = 0

            player_colors = [str(player_score) for player_score in player_scores_normed]
            ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
            ax.axis("off")
            ax.grid(False)

            for p_idx in range(10):
                ax.scatter(
                    [tensors["player_xs"][time_step, p_idx]],
                    [tensors["player_ys"][time_step, p_idx]],
                    s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                    c=player_colors[p_idx],
                    marker=markers[p_idx],
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    edgecolors="black",
                )

            ax.scatter(
                [tensors["ball_xs"][time_step]],
                [tensors["ball_ys"][time_step]],
                s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                c="#ff8c00",
                edgecolors="black",
            )

            plt.subplots_adjust(0, 0, 1, 1)
            fig.canvas.draw()
            court_img = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            plt.cla()

            ball_score = str(ball_score).split(".")[1][:2]
            img_id = f"{str(attn_layer_idx).zfill(2)}_{str(attn_head_idx).zfill(2)}"
            time_step = str(time_step).zfill(2)
            court_img.save(
                f"{home_dir}/test/{img_id}_{test_idx}_{time_step}_{ball_score}.png"
            )

        saved += 1

    shutil.make_archive(f"{home_dir}/test", "zip", f"{home_dir}/test")
    shutil.rmtree(f"{home_dir}/test")


def get_start_idx_from_time(X, period, period_time):
    periods = X[:, 3].astype(int)
    period_times = X[:, 1]
    (period_min, period_sec) = period_time.split(":")
    (period_min, start_sec) = (int(period_min), int(period_sec))
    period_time = 720 - (60 * period_min + start_sec)
    return np.argwhere((period_times > period_time) & (periods == period)).min()


def plot_ball_traj_preds_specific_game():
    JOB = "20210124114950"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (train_dataset, _, _, _, _, _) = init_datasets(opts)
    model = init_model(opts, train_dataset)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    court = plt.imread("court.png")
    width = 5
    height = width * court.shape[0] / float(court.shape[1])
    vert_buffer = 10
    pt_scale = 1.4
    scale = 4

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/test", exist_ok=True)
    (fig, ax) = plt.subplots(figsize=(width, height))

    save_output = SaveOutput()
    hook_handles = []
    for layer in model.transformer.layers:
        hook_handles.append(layer.self_attn.register_forward_hook(save_output))

    gameid = "0021500622"
    X = np.load(f"{GAMES_DIR}/{gameid}_X.npy")
    y = np.load(f"{GAMES_DIR}/{gameid}_y.npy")

    period = 3
    period_time = "1:55"
    start = get_start_idx_from_time(X, period, period_time)
    tensors = train_dataset.get_sample(X, y, start)
    sub_rubio = False
    if sub_rubio:
        baller2vec_config = pickle.load(
            open(f"{DATA_DIR}/baller2vec_config.pydict", "rb")
        )
        player_idx2props = baller2vec_config["player_idx2props"]
        lebron_idx = get_player_idx("LeBron James", player_idx2props)
        rubio_idx = get_player_idx("Ricky Rubio", player_idx2props)
        player_idxs = tensors["player_idxs"]
        player_idxs[player_idxs == lebron_idx] = rubio_idx

    with torch.no_grad():
        ball_trajs = model(tensors)["ball"][-len(tensors["player_idxs"]) :]
        ball_probs = torch.softmax(ball_trajs, dim=1)
        ball_traj_n = train_dataset.ball_traj_n
        ball_probs = ball_probs.reshape(
            len(ball_probs), ball_traj_n, ball_traj_n, ball_traj_n
        )
        ball_xy_probs = ball_probs.sum(dim=1).permute(0, 2, 1)

    colors = []
    markers = []
    for p_idx in range(10):
        if tensors["player_hoop_sides"][0, p_idx]:
            colors.append("white")
            markers.append("s")
        else:
            colors.append("gray")
            markers.append("^")

    ball_start = 10 * len(tensors["player_idxs"])

    for time_step in range(19):
        print(time_step, flush=True)

        ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
        ax.axis("off")
        ax.grid(False)
        for p_idx in range(10):
            ax.scatter(
                [tensors["player_xs"][time_step, p_idx]],
                [tensors["player_ys"][time_step, p_idx]],
                s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                c=colors[p_idx],
                marker=markers[p_idx],
                edgecolors="black",
            )

        ax.scatter(
            [tensors["ball_xs"][time_step]],
            [tensors["ball_ys"][time_step]],
            s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
            c="#ff8c00",
            edgecolors="black",
        )
        plt.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        court_img = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.cla()

        traj_data = np.full((ball_traj_n, ball_traj_n, 3), 255, dtype=np.uint8)
        ball_traj = tensors["ball_trajs"][time_step]
        (ball_row, ball_col_dep) = (
            ball_traj // ball_traj_n ** 2,
            ball_traj % ball_traj_n ** 2,
        )
        (ball_col, ball_dep) = (ball_col_dep // ball_traj_n, ball_col_dep % ball_traj_n)
        traj_data[ball_dep, ball_col] = 0
        traj_img = Image.fromarray(traj_data).resize(
            (scale * ball_traj_n, scale * ball_traj_n), resample=0
        )
        add_grid(traj_img, ball_traj_n)

        traj_pred = np.zeros((ball_traj_n, ball_traj_n, 3))
        traj_pred[:, :] = ball_xy_probs[time_step].view(ball_traj_n, ball_traj_n, 1)
        pred_img = Image.fromarray(np.uint8(255 - 255 * traj_pred)).resize(
            (scale * ball_traj_n, scale * ball_traj_n), resample=0
        )
        add_grid(pred_img, ball_traj_n)

        final_img = Image.new(
            "RGBA",
            (
                court_img.size[0],
                court_img.size[1] + scale * ball_traj_n + vert_buffer,
            ),
        )
        final_img.paste(court_img, (0, 0))
        full_width = 3 * traj_img.size[0]
        horiz_buffer = court_img.size[0] // 2 - full_width // 2
        final_img.paste(traj_img, (horiz_buffer, court_img.size[1] + vert_buffer))
        final_img.paste(
            pred_img,
            (2 * traj_img.size[0] + horiz_buffer, court_img.size[1] + vert_buffer),
        )
        true_prob = traj_pred[ball_dep, ball_col, 0]
        print(true_prob)
        true_prob = f"{true_prob:.2f}".split(".")[1]
        time_step = str(time_step).zfill(2)
        final_img.save(f"{home_dir}/test/court_{time_step}_{true_prob}.png")

    shutil.make_archive(f"{home_dir}/test", "zip", f"{home_dir}/test")
    shutil.rmtree(f"{home_dir}/test")
