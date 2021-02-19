# Adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import torch

from torch import nn


class TimeEncoder(nn.Module):
    def __init__(self, seq_len, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.time_embeddings = nn.Parameter(torch.Tensor(seq_len, d_model))
        nn.init.normal_(self.time_embeddings)

    def forward(self, x, repeat):
        repeated = self.time_embeddings.repeat(repeat, 1)
        x = x + repeated
        return self.dropout(x)


class Baller2Vec(nn.Module):
    def __init__(
        self,
        n_player_ids,
        embedding_dim,
        sigmoid,
        seq_len,
        mlp_layers,
        n_players,
        n_player_labels,
        n_ball_labels,
        n_seq_labels,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
        use_cls,
        embed_before_mlp,
    ):
        super().__init__()
        self.sigmoid = sigmoid
        self.seq_len = seq_len
        self.use_cls = use_cls
        self.n_players = n_players
        self.embed_before_mlp = embed_before_mlp

        # Initialize players, ball, and CLS (if used) embeddings.
        initrange = 0.1
        self.player_embedding = nn.Embedding(n_player_ids, embedding_dim)
        self.player_embedding.weight.data.uniform_(-initrange, initrange)

        self.ball_embedding = nn.Parameter(torch.Tensor(embedding_dim))
        nn.init.uniform_(self.ball_embedding, -initrange, initrange)
        if use_cls:
            self.cls_embedding = nn.Parameter(torch.Tensor(mlp_layers[-1]))
            nn.init.uniform_(self.cls_embedding, -initrange, initrange)

        # Initialize preprocessing MLPs.
        player_mlp = nn.Sequential()
        ball_mlp = nn.Sequential()
        # Extra dimensions for (x, y) coordinates and hoop side (for players) or z
        # coordinate (for ball).
        in_feats = embedding_dim + 3 if embed_before_mlp else 3
        for (layer_idx, out_feats) in enumerate(mlp_layers):
            if (not embed_before_mlp) and (layer_idx == len(mlp_layers) - 1):
                out_feats = out_feats - embedding_dim

            player_mlp.add_module(f"layer{layer_idx}", nn.Linear(in_feats, out_feats))
            ball_mlp.add_module(f"layer{layer_idx}", nn.Linear(in_feats, out_feats))

            if layer_idx < len(mlp_layers) - 1:
                player_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                ball_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

            in_feats = out_feats

        self.player_mlp = player_mlp
        self.ball_mlp = ball_mlp

        # Initialize time encoders.
        d_model = mlp_layers[-1]
        self.d_model = d_model
        self.player_time_encoder = TimeEncoder(seq_len, d_model, dropout)
        self.ball_time_encoder = TimeEncoder(seq_len, d_model, dropout)
        if use_cls:
            self.cls_time_encoder = TimeEncoder(seq_len, d_model, dropout)

        # Initialize Transformer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Initialize classification layers.
        self.player_classifier = nn.Linear(d_model, n_player_labels)
        self.player_classifier.weight.data.uniform_(-initrange, initrange)
        self.player_classifier.bias.data.zero_()

        self.ball_classifier = nn.Linear(d_model, n_ball_labels)
        self.ball_classifier.weight.data.uniform_(-initrange, initrange)
        self.ball_classifier.bias.data.zero_()

        if use_cls:
            self.event_classifier = nn.Linear(d_model, n_seq_labels)
            self.event_classifier.weight.data.uniform_(-initrange, initrange)
            self.event_classifier.bias.data.zero_()

        # Initialize mask.
        self.register_buffer("mask", self.generate_square_subsequent_mask())

    def generate_square_subsequent_mask(self):
        # n players plus the ball and the CLS entity (if used).
        if self.use_cls:
            sz = (self.n_players + 2) * self.seq_len
        else:
            sz = (self.n_players + 1) * self.seq_len

        mask = torch.zeros(sz, sz)
        ball_start = self.n_players * self.seq_len
        if self.use_cls:
            cls_start = ball_start + self.seq_len

        for step in range(self.seq_len):
            start = self.n_players * step
            stop = start + self.n_players
            ball_stop = ball_start + step + 1

            # The players can look at the players.
            mask[start:stop, :stop] = 1
            # The players can look at the ball.
            mask[start:stop, ball_start:ball_stop] = 1
            # The ball can look at the players.
            mask[ball_start + step, :stop] = 1
            # The ball can look at the ball.
            mask[ball_start + step, ball_start:ball_stop] = 1
            if self.use_cls:
                cls_stop = cls_start + step + 1
                # The players can look at the CLS.
                mask[start:stop, cls_start:cls_stop] = 1
                # The ball can look at the CLS.
                mask[ball_start + step, cls_start:cls_stop] = 1
                # The CLS can look at the players.
                mask[cls_start + step, :stop] = 1
                # The CLS can look at the ball.
                mask[cls_start + step, ball_start:ball_stop] = 1
                # The CLS can look at the CLS.
                mask[cls_start + step, cls_start:cls_stop] = 1

        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tensors):
        device = list(self.player_mlp.parameters())[0].device

        # Get player position/time features.
        player_embeddings = self.player_embedding(
            tensors["player_idxs"].flatten().to(device)
        )
        if self.sigmoid == "logistic":
            player_embeddings = torch.sigmoid(player_embeddings)
        elif self.sigmoid == "tanh":
            player_embeddings = torch.tanh(player_embeddings)

        player_xs = tensors["player_xs"].flatten().unsqueeze(1).to(device)
        player_ys = tensors["player_ys"].flatten().unsqueeze(1).to(device)
        player_hoop_sides = (
            tensors["player_hoop_sides"].flatten().unsqueeze(1).to(device)
        )
        if self.embed_before_mlp:
            player_pos = torch.cat(
                [
                    player_embeddings,
                    player_xs,
                    player_ys,
                    player_hoop_sides,
                ],
                dim=1,
            )
            player_pos_feats = self.player_mlp(player_pos) * math.sqrt(self.d_model)
        else:
            player_pos = torch.cat(
                [
                    player_xs,
                    player_ys,
                    player_hoop_sides,
                ],
                dim=1,
            )
            player_pos_feats = self.player_mlp(player_pos) * math.sqrt(self.d_model)
            player_pos_feats = torch.cat([player_embeddings, player_pos_feats], dim=1)

        player_pos_time_feats = self.player_time_encoder(
            player_pos_feats, self.n_players
        )

        # Get ball position/time features.
        ball_embeddings = self.ball_embedding.repeat(self.seq_len, 1)
        ball_xs = tensors["ball_xs"].unsqueeze(1).to(device)
        ball_ys = tensors["ball_ys"].unsqueeze(1).to(device)
        ball_zs = tensors["ball_zs"].unsqueeze(1).to(device)
        if self.embed_before_mlp:
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
        else:
            ball_pos = torch.cat(
                [
                    ball_xs,
                    ball_ys,
                    ball_zs,
                ],
                dim=1,
            )
            ball_pos_feats = self.player_mlp(ball_pos) * math.sqrt(self.d_model)
            ball_pos_feats = torch.cat([ball_embeddings, ball_pos_feats], dim=1)

        ball_pos_time_feats = self.ball_time_encoder(ball_pos_feats, 1)

        # Combine players and ball features.
        combined = torch.cat([player_pos_time_feats, ball_pos_time_feats], dim=0)

        if self.use_cls:
            # Get CLS time features.
            cls_feats = self.cls_embedding.repeat(self.seq_len, 1)
            cls_time_feats = self.cls_time_encoder(cls_feats, 1)

            # Combine with CLS features.
            combined = torch.cat([combined, cls_time_feats], dim=0)

        output = self.transformer(combined.unsqueeze(1), self.mask)
        preds = {
            "player": self.player_classifier(output).squeeze(1),
            "ball": self.ball_classifier(output).squeeze(1),
        }
        if self.use_cls:
            preds["seq_label"] = self.event_classifier(output).squeeze(1)

        return preds


class Baller2VecSeq2Seq(nn.Module):
    def __init__(
        self,
        n_player_ids,
        embedding_dim,
        sigmoid,
        seq_len,
        mlp_layers,
        n_player_labels,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.sigmoid = sigmoid
        self.seq_len = seq_len

        # Initialize players and ball embeddings.
        initrange = 0.1
        self.player_embedding = nn.Embedding(n_player_ids, embedding_dim)
        self.player_embedding.weight.data.uniform_(-initrange, initrange)
        self.ball_embedding = nn.Parameter(torch.Tensor(embedding_dim))
        nn.init.uniform_(self.ball_embedding, -initrange, initrange)

        d_model = mlp_layers[-1]
        self.d_model = d_model

        model = {}
        for enc_dec in ["enc", "dec"]:
            # Initialize preprocessing MLPs.
            player_mlp = nn.Sequential()
            ball_mlp = nn.Sequential()
            # Extra dimensions for (x, y) coordinates and hoop side (for players) or z
            # coordinate (for ball).
            in_feats = embedding_dim + 3
            for (layer_idx, out_feats) in enumerate(mlp_layers):
                player_mlp.add_module(
                    f"layer{layer_idx}", nn.Linear(in_feats, out_feats)
                )
                ball_mlp.add_module(f"layer{layer_idx}", nn.Linear(in_feats, out_feats))
                if layer_idx < len(mlp_layers) - 1:
                    player_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                    ball_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

                in_feats = out_feats

            # Initialize time encoders.
            player_time_encoder = TimeEncoder(seq_len, d_model, dropout)
            ball_time_encoder = TimeEncoder(seq_len, d_model, dropout)

            # Initialize Transformer.
            if enc_dec == "enc":
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout
                )
                transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            else:
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model, nhead, dim_feedforward, dropout
                )
                transformer = nn.TransformerDecoder(decoder_layer, num_layers)

            model[enc_dec] = nn.ModuleDict(
                {
                    "player_mlp": player_mlp,
                    "ball_mlp": ball_mlp,
                    "player_time_encoder": player_time_encoder,
                    "ball_time_encoder": ball_time_encoder,
                    "transformer": transformer,
                }
            )

        self.model = nn.ModuleDict(model)

        # Initialize classification layer.
        self.player_classifier = nn.Linear(d_model, n_player_labels)
        self.player_classifier.weight.data.uniform_(-initrange, initrange)
        self.player_classifier.bias.data.zero_()

        # Initialize mask.
        self.register_buffer("mask", self.generate_square_subsequent_mask())

    def generate_square_subsequent_mask(self):
        # Five players plus the ball.
        sz = 6 * self.seq_len
        mask = torch.zeros(sz, sz)
        ball_start = 5 * self.seq_len
        for step in range(self.seq_len):
            start = 5 * step
            stop = start + 5
            ball_stop = ball_start + step + 1

            # The players can look at the players.
            mask[start:stop, :stop] = 1
            # The players can look at the ball.
            mask[start:stop, ball_start:ball_stop] = 1
            # The ball can look at the players.
            mask[ball_start + step, :stop] = 1
            # The ball can look at the ball.
            mask[ball_start + step, ball_start:ball_stop] = 1

        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tensors, start_stops):
        device = list(self.player_embedding.parameters())[0].device

        for enc_dec in ["enc", "dec"]:
            (start, stop) = start_stops[enc_dec]

            # Get player position/time features.
            player_embeddings = self.player_embedding(
                tensors["player_idxs"][:, start:stop].flatten().to(device)
            )
            if self.sigmoid == "logistic":
                player_embeddings = torch.sigmoid(player_embeddings)
            elif self.sigmoid == "tanh":
                player_embeddings = torch.tanh(player_embeddings)

            player_xs = (
                tensors["player_xs"][:, start:stop].flatten().unsqueeze(1).to(device)
            )
            player_ys = (
                tensors["player_ys"][:, start:stop].flatten().unsqueeze(1).to(device)
            )
            player_hoop_sides = (
                tensors["player_hoop_sides"][:, start:stop]
                .flatten()
                .unsqueeze(1)
                .to(device)
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
            player_pos_feats = self.model[enc_dec]["player_mlp"](
                player_pos
            ) * math.sqrt(self.d_model)
            player_pos_time_feats = self.model[enc_dec]["player_time_encoder"](
                player_pos_feats, 5
            )

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
            ball_pos_feats = self.model[enc_dec]["ball_mlp"](ball_pos) * math.sqrt(
                self.d_model
            )
            ball_pos_time_feats = self.model[enc_dec]["ball_time_encoder"](
                ball_pos_feats, 1
            )

            # Combine players and ball features.
            combined = torch.cat(
                [player_pos_time_feats, ball_pos_time_feats], dim=0
            ).unsqueeze(1)

            if enc_dec == "enc":
                output = self.model[enc_dec]["transformer"](combined)
            else:
                output = self.model[enc_dec]["transformer"](combined, output, self.mask)

        output = self.player_classifier(output).squeeze(1)
        return output
