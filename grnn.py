import math
import torch

from torch import nn


class GRNN(nn.Module):
    def __init__(
        self,
        n_player_ids,
        embedding_dim,
        seq_len,
        mlp_layers,
        n_players,
        n_player_labels,
        dim_feedforward,
        gnn_layers,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_players = n_players

        # Initialize players, ball, and CLS (if used) embeddings.
        initrange = 0.1
        self.player_embedding = nn.Embedding(n_player_ids, embedding_dim)
        self.player_embedding.weight.data.uniform_(-initrange, initrange)

        self.ball_embedding = nn.Parameter(torch.Tensor(embedding_dim))
        nn.init.uniform_(self.ball_embedding, -initrange, initrange)

        # Initialize preprocessing MLPs.
        player_mlp = nn.Sequential()
        ball_mlp = nn.Sequential()
        # Extra dimensions for (x, y) coordinates and hoop side (for players) or z
        # coordinate (for ball).
        in_feats = embedding_dim + 3
        for (layer_idx, out_feats) in enumerate(mlp_layers):
            player_mlp.add_module(f"layer{layer_idx}", nn.Linear(in_feats, out_feats))
            ball_mlp.add_module(f"layer{layer_idx}", nn.Linear(in_feats, out_feats))
            if layer_idx < len(mlp_layers) - 1:
                player_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                ball_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

            in_feats = out_feats

        self.player_mlp = player_mlp
        self.ball_mlp = ball_mlp

        # Initialize GRU components.
        d_model = mlp_layers[-1]
        self.d_model = d_model
        self.h_0s = nn.Parameter(torch.Tensor(n_players + 1, d_model))
        nn.init.normal_(self.h_0s)
        gru_comps = {}
        gru_biases = {}
        gru_norms = {}
        for z_r_h in ["z", "r", "h"]:
            gru_biases[z_r_h] = nn.Parameter(torch.Tensor(1, d_model))
            nn.init.normal_(gru_biases[z_r_h])
            for W_U in ["W", "U"]:
                comp = f"{W_U}_{z_r_h}"
                gru_comps[comp] = nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, d_model),
                )
                gru_norms[comp] = nn.LayerNorm(d_model)

        self.gru_comps = nn.ModuleDict(gru_comps)
        self.gru_biases = nn.ParameterDict(gru_biases)
        self.gru_norms = nn.ModuleDict(gru_norms)

        # Initialize GNN components.
        self.edge_project = nn.Linear(2 * d_model, d_model)
        edge_layers = []
        edge_norms = []
        node_layers = []
        node_norms = []
        for layer_idx in range(gnn_layers):
            edge_layers.append(
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, d_model),
                )
            )
            edge_norms.append(nn.LayerNorm(d_model))
            node_layers.append(
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, d_model),
                )
            )
            node_norms.append(nn.LayerNorm(d_model))

        self.edge_layers = nn.ModuleList(edge_layers)
        self.edge_norms = nn.ModuleList(edge_norms)
        self.node_layers = nn.ModuleList(node_layers)
        self.node_norms = nn.ModuleList(node_norms)

        # Initialize classification layer.
        self.player_classifier = nn.Linear(d_model, n_player_labels)
        self.player_classifier.weight.data.uniform_(-initrange, initrange)
        self.player_classifier.bias.data.zero_()

    def forward(self, tensors):
        device = list(self.player_mlp.parameters())[0].device

        # Get player position/time features.
        player_embeddings = self.player_embedding(
            tensors["player_idxs"].flatten().to(device)
        )
        player_xs = tensors["player_xs"].flatten().unsqueeze(1).to(device)
        player_ys = tensors["player_ys"].flatten().unsqueeze(1).to(device)
        player_hoop_sides = (
            tensors["player_hoop_sides"].flatten().unsqueeze(1).to(device)
        )
        player_pos = torch.cat(
            [
                player_embeddings,
                player_xs,
                player_ys,
                player_hoop_sides,
            ],
            dim=1,
        )
        player_pos_feats = self.player_mlp(player_pos)

        # Get ball position/time features.
        ball_embeddings = self.ball_embedding.repeat(self.seq_len, 1)
        ball_xs = tensors["ball_xs"].unsqueeze(1).to(device)
        ball_ys = tensors["ball_ys"].unsqueeze(1).to(device)
        ball_zs = tensors["ball_zs"].unsqueeze(1).to(device)
        ball_pos = torch.cat(
            [
                ball_embeddings,
                ball_xs,
                ball_ys,
                ball_zs,
            ],
            dim=1,
        )
        ball_pos_feats = self.ball_mlp(ball_pos) * math.sqrt(self.d_model)

        # Run GRNN.
        h_ts = self.h_0s
        b_z = self.gru_biases["z"]
        b_r = self.gru_biases["r"]
        b_h = self.gru_biases["h"]
        preds = []
        for time_step in range(self.seq_len):
            start = time_step * self.n_players
            end = start + self.n_players
            time_step_players = player_pos_feats[start:end]
            time_step_ball = ball_pos_feats[time_step : time_step + 1]
            x_ts = torch.cat([time_step_players, time_step_ball])

            # GRU pieces. See: https://en.wikipedia.org/wiki/Gated_recurrent_unit.
            # z_ts.
            W_z_ts = x_ts.clone()
            W_z_t_2s = self.gru_comps["W_z"](W_z_ts)
            W_z_ts = self.gru_norms["W_z"](W_z_ts + W_z_t_2s)
            U_z_ts = h_ts.clone()
            U_z_t_2s = self.gru_comps["U_z"](U_z_ts)
            U_z_ts = self.gru_norms["U_z"](U_z_ts + U_z_t_2s)
            z_ts = torch.sigmoid(W_z_ts + U_z_ts + b_z)

            # r_ts.
            W_r_ts = x_ts.clone()
            W_r_t_2s = self.gru_comps["W_r"](W_r_ts)
            W_r_ts = self.gru_norms["W_r"](W_r_ts + W_r_t_2s)
            U_r_ts = h_ts.clone()
            U_r_t_2s = self.gru_comps["U_r"](U_r_ts)
            U_r_ts = self.gru_norms["U_r"](U_r_ts + U_r_t_2s)
            r_ts = torch.sigmoid(W_r_ts + U_r_ts + b_r)

            # h_t_hats.
            W_h_ts = x_ts.clone()
            W_h_t_2s = self.gru_comps["W_h"](W_h_ts)
            W_h_ts = self.gru_norms["W_h"](W_h_ts + W_h_t_2s)
            U_h_ts = h_ts.clone()
            U_h_t_2s = self.gru_comps["U_h"](r_ts * U_h_ts)
            U_h_ts = self.gru_norms["U_h"](U_h_ts + U_h_t_2s)
            h_t_hats = torch.tanh(W_h_ts + U_h_ts + b_h)

            # h_ts.
            h_ts = (1 - z_ts) * h_ts + z_ts * h_t_hats

            # GNN pieces. See equations (2) and (3) in Yeh et al. (2019).
            v_is = h_ts.repeat_interleave(self.n_players + 1, 0)
            v_js = h_ts.repeat(self.n_players + 1, 1)
            e_t_i_js = self.edge_project(torch.cat([v_is, v_js], dim=1))
            for (edge_layer_idx, edge_layer) in enumerate(self.edge_layers):
                e_t_i_j_2s = edge_layer(e_t_i_js)
                e_t_i_js = self.edge_norms[edge_layer_idx](e_t_i_js + e_t_i_j_2s)

            e_t_i_js = e_t_i_js.view(
                self.n_players + 1, self.n_players + 1, self.d_model
            )
            o_is = e_t_i_js.sum(dim=1)
            for (node_layer_idx, node_layer) in enumerate(self.node_layers):
                o_i_2s = node_layer(o_is)
                o_is = self.node_norms[node_layer_idx](o_is + o_i_2s)

            h_ts = o_is
            preds.append(self.player_classifier(h_ts[: self.n_players]))

        preds = {"player": torch.cat(preds)}
        return preds
