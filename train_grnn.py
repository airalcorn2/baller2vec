import sys
import time
import torch
import yaml

from grnn import GRNN
from settings import *
from torch import nn, optim
from train_baller2vec import init_datasets

SEED = 2010
torch.manual_seed(SEED)
torch.set_printoptions(linewidth=160)


def init_model(opts, train_dataset):
    model_config = opts["model"]
    # Add one for the generic player.
    model_config["n_player_ids"] = train_dataset.n_player_ids + 1
    model_config["seq_len"] = train_dataset.chunk_size // train_dataset.hz - 1
    model_config["n_players"] = train_dataset.n_players
    model_config["n_player_labels"] = train_dataset.player_traj_n ** 2
    model = GRNN(**model_config)
    return model


def get_preds_labels(tensors):
    player_trajs = tensors["player_trajs"].flatten()
    n_player_trajs = len(player_trajs)
    labels = player_trajs.to(device)
    preds = model(tensors)["player"][:n_player_trajs]
    return (preds, labels)


def train_model():
    # Initialize optimizer.
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

    best_train_loss = float("inf")
    best_valid_loss = float("inf")
    test_loss_best_valid = float("inf")
    total_train_loss = None
    no_improvement = 0
    for epoch in range(175):
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
            print(probs.view(model.seq_len, model.n_players), flush=True)
            print(preds.view(model.seq_len, model.n_players), flush=True)
            print(labels.view(model.seq_len, model.n_players), flush=True)

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

        elif no_improvement < patience:
            no_improvement += 1
            if no_improvement == patience:
                print("Reducing learning rate.")
                optimizer = optim.Adam(
                    train_params, lr=0.1 * opts["train"]["learning_rate"]
                )

        print(f"total_train_loss: {total_train_loss}")
        print(f"best_train_loss: {best_train_loss}")
        print(f"total_valid_loss: {total_valid_loss}")
        print(f"best_valid_loss: {best_valid_loss}")
        print(f"test_loss_best_valid: {test_loss_best_valid}")

        model.train()
        total_train_loss = 0.0
        n_train = 0
        start_time = time.time()
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

        epoch_time = time.time() - start_time

        total_train_loss /= n_train
        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss

        print(f"epoch_time: {epoch_time:.2f}", flush=True)


if __name__ == "__main__":
    JOB = sys.argv[1]
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    except IndexError:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    patience = opts["train"]["patience"]

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

    train_model()
