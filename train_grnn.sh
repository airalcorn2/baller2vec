#!/usr/bin/env bash

JOB=$(date +%Y%m%d%H%M%S)

echo "train:" >> ${JOB}.yaml
echo "  task: player_traj" >> ${JOB}.yaml
echo "  min_playing_time: 0" >> ${JOB}.yaml
echo "  train_valid_prop: 0.95" >> ${JOB}.yaml
echo "  train_prop: 0.95" >> ${JOB}.yaml
echo "  train_samples_per_epoch: 20000" >> ${JOB}.yaml
echo "  valid_samples: 1000" >> ${JOB}.yaml
echo "  workers: 10" >> ${JOB}.yaml
echo "  learning_rate: 1.0e-5" >> ${JOB}.yaml
echo "  patience: 20" >> ${JOB}.yaml

echo "dataset:" >> ${JOB}.yaml
echo "  hz: 5" >> ${JOB}.yaml
echo "  secs: 4" >> ${JOB}.yaml
echo "  player_traj_n: 11" >> ${JOB}.yaml
echo "  max_player_move: 4.5" >> ${JOB}.yaml
echo "  ball_traj_n: 19" >> ${JOB}.yaml
echo "  max_ball_move: 8.5" >> ${JOB}.yaml
echo "  n_players: 10" >> ${JOB}.yaml
echo "  next_score_change_time_max: 35" >> ${JOB}.yaml
echo "  n_time_to_next_score_change: 36" >> ${JOB}.yaml
echo "  n_ball_loc_x: 95" >> ${JOB}.yaml
echo "  n_ball_loc_y: 51" >> ${JOB}.yaml
echo "  ball_future_secs: 2" >> ${JOB}.yaml

echo "model:" >> ${JOB}.yaml
echo "  embedding_dim: 20" >> ${JOB}.yaml
echo "  mlp_layers: [128, 256, 512]" >> ${JOB}.yaml
echo "  dim_feedforward: 2048" >> ${JOB}.yaml
echo "  gnn_layers: 1" >> ${JOB}.yaml

# Save experiment settings.
mkdir -p ${EXPERIMENTS_DIR}/${JOB}
mv ${JOB}.yaml ${EXPERIMENTS_DIR}/${JOB}/

gpu=0
cd ${PROJECT_DIR}
nohup python3 train_grnn.py ${JOB} ${gpu} > ${EXPERIMENTS_DIR}/${JOB}/train.log &