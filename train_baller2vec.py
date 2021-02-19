import numpy as np
import pickle
import random
import sys
import torch
import yaml

from baller2vec import Baller2Vec, Baller2VecSeq2Seq
from baller2vec_dataset import Baller2VecDataset
from settings import *
from torch import nn, optim
from torch.utils.data import DataLoader


def worker_init_fn(worker_id):
    # See: https://pytorch.org/docs/stable/notes/faq.html#my-data-loader-workers-return-identical-random-numbers
    # and: https://pytorch.org/docs/stable/data.html#multi-process-data-loading
    # and: https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading.
    # NumPy seed takes a 32-bit unsigned integer.
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2 ** 32 - 1))


def init_datasets(opts):
    gameids = list(set([np_f.split("_")[0] for np_f in os.listdir(GAMES_DIR)]))
    gameids.sort()
    np.random.seed(2010)
    np.random.shuffle(gameids)
    n_train_valid = int(opts["train"]["train_valid_prop"] * len(gameids))
    train_valid_gameids = gameids[:n_train_valid]

    baller2vec_config = pickle.load(open(f"{DATA_DIR}/baller2vec_config.pydict", "rb"))
    n_player_ids = len(baller2vec_config["player_idx2props"])
    filtered_player_idxs = set()
    for (player_idx, player_props) in baller2vec_config["player_idx2props"].items():
        if "playing_time" not in player_props:
            continue

        if player_props["playing_time"] < opts["train"]["min_playing_time"]:
            filtered_player_idxs.add(player_idx)

    dataset_config = opts["dataset"]
    n_train = int(opts["train"]["train_prop"] * len(train_valid_gameids))
    dataset_config["gameids"] = train_valid_gameids[:n_train]
    dataset_config["N"] = opts["train"]["train_samples_per_epoch"]
    dataset_config["starts"] = []
    dataset_config["mode"] = "train"
    dataset_config["n_player_ids"] = n_player_ids
    dataset_config["filtered_player_idxs"] = filtered_player_idxs
    train_dataset = Baller2VecDataset(**dataset_config)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
        worker_init_fn=worker_init_fn,
    )

    valid_gameids = train_valid_gameids[n_train:]
    N = opts["train"]["valid_samples"]
    samps_per_gameid = N // len(valid_gameids)
    starts = []
    for gameid in valid_gameids:
        y = np.load(f"{GAMES_DIR}/{gameid}_y.npy")
        max_start = len(y) - train_dataset.chunk_size
        gaps = max_start // samps_per_gameid
        starts.append(gaps * np.arange(samps_per_gameid))

    dataset_config["gameids"] = np.repeat(valid_gameids, samps_per_gameid)
    dataset_config["N"] = len(dataset_config["gameids"])
    dataset_config["starts"] = np.concatenate(starts)
    dataset_config["mode"] = "valid"
    valid_dataset = Baller2VecDataset(**dataset_config)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    test_gamieds = gameids[n_train_valid:]
    samps_per_gameid = N // len(test_gamieds)
    starts = []
    for gameid in test_gamieds:
        y = np.load(f"{GAMES_DIR}/{gameid}_y.npy")
        max_start = len(y) - train_dataset.chunk_size
        gaps = max_start // samps_per_gameid
        starts.append(gaps * np.arange(samps_per_gameid))

    dataset_config["gameids"] = np.repeat(test_gamieds, samps_per_gameid)
    dataset_config["N"] = len(dataset_config["gameids"])
    dataset_config["starts"] = np.concatenate(starts)
    dataset_config["mode"] = "test"
    test_dataset = Baller2VecDataset(**dataset_config)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    if opts["train"]["task"] != "event":
        opts["n_seq_labels"] = train_dataset.n_score_changes + 1
    else:
        opts["n_seq_labels"] = len(baller2vec_config["event2event_idx"])

    return (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    )


def init_model(opts, train_dataset):
    model_config = opts["model"]
    # Add one for the generic player.
    model_config["n_player_ids"] = train_dataset.n_player_ids + 1
    model_config["seq_len"] = train_dataset.chunk_size // train_dataset.hz - 1
    model_config["n_player_labels"] = train_dataset.player_traj_n ** 2
    if opts["train"]["task"] == "seq2seq":
        model = Baller2VecSeq2Seq(**model_config)
    else:
        model_config["n_seq_labels"] = opts["n_seq_labels"]
        model_config["n_players"] = train_dataset.n_players
        if opts["train"]["task"] == "ball_loc":
            model_config["n_ball_labels"] = (
                train_dataset.n_ball_loc_y_bins * train_dataset.n_ball_loc_x_bins
            )
        else:
            model_config["n_ball_labels"] = train_dataset.ball_traj_n ** 3

        model = Baller2Vec(**model_config)

    return model


def get_preds_labels(tensors):
    if ("player" in task) or ("ball" in task):
        player_trajs = tensors["player_trajs"].flatten()
        n_player_trajs = len(player_trajs)
        if task == "player_traj":
            labels = player_trajs.to(device)
            preds = model(tensors)["player"][:n_player_trajs]
        else:
            if task == "ball_loc":
                labels = tensors["ball_locs"].flatten().to(device)
            else:
                labels = tensors["ball_trajs"].flatten().to(device)

            preds = model(tensors)["ball"][n_player_trajs:][: len(labels)]
    elif (task == "event") or (task == "score"):
        if task == "event":
            labels = tensors["events"].flatten().to(device)
        else:
            labels = tensors["score_changes"].flatten().to(device)

        preds = model(tensors)["seq_label"][-model.seq_len :]
    elif task == "seq2seq":
        # Randomly choose which team to encode.
        start_stops = {"enc": (0, 5), "dec": (5, 10)}
        if random.random() < 0.5:
            start_stops = {"enc": (5, 10), "dec": (0, 5)}

        (start, stop) = start_stops["dec"]
        labels = tensors["player_trajs"][:, start:stop].flatten().to(device)
        preds = model(tensors, start_stops)[: len(labels)]

    return (preds, labels)


def train_model(optimizer):
    best_train_loss = float("inf")
    best_valid_loss = float("inf")
    test_loss_best_valid = float("inf")
    total_train_loss = None
    no_improvement = 0
    for epoch in range(500):
        print(f"\nepoch: {epoch}", flush=True)

        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            n_valid = 0
            for valid_tensors in valid_loader:
                # Skip bad sequences.
                if len(valid_tensors["player_idxs"]) < model.seq_len:
                    continue

                (preds, labels) = get_preds_labels(valid_tensors)
                loss = criterion(preds, labels)
                total_valid_loss += loss.item()
                n_valid += 1

            probs = torch.softmax(preds, dim=1)
            print(probs.max(1), flush=True)
            print(labels, flush=True)

            total_valid_loss /= n_valid

        if total_valid_loss < best_valid_loss:
            best_valid_loss = total_valid_loss
            torch.save(optimizer.state_dict(), f"{JOB_DIR}/optimizer.pth")
            torch.save(model.state_dict(), f"{JOB_DIR}/best_params.pth")

            test_loss_best_valid = 0.0
            with torch.no_grad():
                n_test = 0
                for test_tensors in test_loader:
                    # Skip bad sequences.
                    if len(test_tensors["player_idxs"]) < model.seq_len:
                        continue

                    (preds, labels) = get_preds_labels(test_tensors)
                    loss = criterion(preds, labels)
                    test_loss_best_valid += loss.item()
                    n_test += 1

            test_loss_best_valid /= n_test

        elif ((task == "event") or (task == "score")) and (opts["train"]["prev_model"]):
            no_improvement += 1
            if no_improvement == opts["train"]["patience"]:
                train_params = [params for params in model.parameters()]
                optimizer = optim.Adam(train_params, lr=opts["train"]["learning_rate"])

        print(f"total_train_loss: {total_train_loss}")
        print(f"best_train_loss: {best_train_loss}")
        print(f"total_valid_loss: {total_valid_loss}")
        print(f"best_valid_loss: {best_valid_loss}")
        print(f"test_loss_best_valid: {test_loss_best_valid}")

        model.train()
        total_train_loss = 0.0
        n_train = 0
        for (train_idx, train_tensors) in enumerate(train_loader):
            if train_idx % 1000 == 0:
                print(train_idx, flush=True)

            # Skip bad sequences.
            if len(train_tensors["player_idxs"]) < model.seq_len:
                continue

            optimizer.zero_grad()
            (preds, labels) = get_preds_labels(train_tensors)
            loss = criterion(preds, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

        total_train_loss /= n_train
        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss


if __name__ == "__main__":
    JOB = sys.argv[1]
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    except IndexError:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    task = opts["train"]["task"]

    # Initialize datasets.
    (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    ) = init_datasets(opts)

    # Initialize model.
    device = torch.device("cuda:0")
    model = init_model(opts, train_dataset).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    # Initialize optimizer.
    if ((task == "event") or (task == "score")) and (opts["train"]["prev_model"]):
        old_job = opts["train"]["prev_model"]
        old_job_dir = f"{EXPERIMENTS_DIR}/{old_job}"
        model.load_state_dict(torch.load(f"{old_job_dir}/best_params.pth"))
        train_params = [params for params in model.event_classifier.parameters()]
    else:
        train_params = [params for params in model.parameters()]

    optimizer = optim.Adam(train_params, lr=opts["train"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Continue training on a prematurely terminated model.
    try:
        model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))

        try:
            optimizer.load_state_dict(torch.load(f"{JOB_DIR}/optimizer.pth"))
        except ValueError:
            print("Old optimizer doesn't match.")

    except FileNotFoundError:
        pass

    train_model(optimizer)
