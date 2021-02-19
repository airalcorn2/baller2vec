import multiprocessing
import numpy as np
import pandas as pd
import pickle
import py7zr
import shutil

from settings import *

HALF_COURT_LENGTH = COURT_LENGTH // 2
THRESHOLD = 1.0


def game_name2gameid_worker(game_7zs, queue):
    game_names = []
    gameids = []
    for game_7z in game_7zs:
        game_name = game_7z.split(".7z")[0]
        game_names.append(game_name)
        try:
            archive = py7zr.SevenZipFile(f"{TRACKING_DIR}/{game_7z}", mode="r")
            archive.extractall(path=f"{TRACKING_DIR}/{game_name}")
            archive.close()
        except AttributeError:
            shutil.rmtree(f"{TRACKING_DIR}/{game_name}")
            gameids.append("N/A")
            continue

        try:
            gameids.append(os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0])
        except IndexError:
            gameids.append("N/A")

        shutil.rmtree(f"{TRACKING_DIR}/{game_name}")

    queue.put((game_names, gameids))


def get_game_name2gameid_map():
    q = multiprocessing.Queue()

    dir_fs = os.listdir(TRACKING_DIR)
    all_game_7zs = [dir_f for dir_f in dir_fs if dir_f.endswith(".7z")]
    processes = multiprocessing.cpu_count()
    game_7zs_per_process = int(np.ceil(len(all_game_7zs) / processes))
    jobs = []
    for i in range(processes):
        start = i * game_7zs_per_process
        end = start + game_7zs_per_process
        game_7zs = all_game_7zs[start:end]
        p = multiprocessing.Process(target=game_name2gameid_worker, args=(game_7zs, q))
        jobs.append(p)
        p.start()

    all_game_names = []
    all_gameids = []
    for _ in jobs:
        (game_names, gameids) = q.get()
        all_game_names.extend(game_names)
        all_gameids.extend(gameids)

    for p in jobs:
        p.join()

    df = pd.DataFrame.from_dict({"game_name": all_game_names, "gameid": all_gameids})
    home_dir = os.path.expanduser("~")
    df.to_csv(f"{home_dir}/test.csv", index=False)


def playerid2player_idx_map_worker(game_7zs, queue):
    playerid2props = {}
    for game_7z in game_7zs:
        game_name = game_7z.split(".7z")[0]
        try:
            archive = py7zr.SevenZipFile(f"{TRACKING_DIR}/{game_7z}", mode="r")
            archive.extractall(path=f"{TRACKING_DIR}/{game_name}")
            archive.close()
        except AttributeError:
            print(f"{game_name}\nBusted.", flush=True)
            shutil.rmtree(f"{TRACKING_DIR}/{game_name}")
            continue

        try:
            gameid = os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0]
        except IndexError:
            print(f"No tracking data for {game_name}.", flush=True)
            shutil.rmtree(f"{TRACKING_DIR}/{game_name}")
            continue

        df_tracking = pd.read_json(f"{TRACKING_DIR}/{game_name}/{gameid}.json")
        event = df_tracking["events"].iloc[0]
        players = event["home"]["players"] + event["visitor"]["players"]
        for player in players:
            playerid = player["playerid"]
            playerid2props[playerid] = {
                "name": " ".join([player["firstname"], player["lastname"]]),
            }

    queue.put(playerid2props)


def get_playerid2player_idx_map():
    q = multiprocessing.Queue()

    dir_fs = os.listdir(TRACKING_DIR)
    all_game_7zs = [dir_f for dir_f in dir_fs if dir_f.endswith(".7z")]
    processes = multiprocessing.cpu_count()
    game_7zs_per_process = int(np.ceil(len(all_game_7zs) / processes))
    jobs = []
    for i in range(processes):
        start = i * game_7zs_per_process
        end = start + game_7zs_per_process
        game_7zs = all_game_7zs[start:end]
        p = multiprocessing.Process(
            target=playerid2player_idx_map_worker, args=(game_7zs, q)
        )
        jobs.append(p)
        p.start()

    playerid2props = {}
    for _ in jobs:
        playerid2props.update(q.get())

    for p in jobs:
        p.join()

    playerid2player_idx = {}
    player_idx2props = {}
    for (player_idx, playerid) in enumerate(playerid2props):
        playerid2player_idx[playerid] = player_idx
        player_idx2props[player_idx] = playerid2props[playerid]
        player_idx2props[player_idx]["playerid"] = playerid

    return (playerid2player_idx, player_idx2props)


def get_game_time(game_clock_secs, period):
    period_secs = 720 if period <= 4 else 300
    period_time = period_secs - game_clock_secs
    if period <= 4:
        return (period - 1) * 720 + period_time
    else:
        return 4 * 720 + (period - 5) * 300 + period_time


def get_shot_times_worker(game_7zs, queue):
    shot_times = {}
    for game_7z in game_7zs:
        game_name = game_7z.split(".7z")[0]
        try:
            gameid = os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0]
        except FileNotFoundError:
            continue

        df_events = pd.read_csv(f"{EVENTS_DIR}/{gameid}.csv")
        game_shot_times = {}
        for (row_idx, row) in df_events.iterrows():
            period = row["PERIOD"]
            game_clock = row["PCTIMESTRING"].split(":")
            game_clock_secs = 60 * int(game_clock[0]) + int(game_clock[1])
            game_time = get_game_time(game_clock_secs, period)

            if (row["EVENTMSGTYPE"] == 1) or (row["EVENTMSGTYPE"] == 2):
                game_shot_times[game_time] = row["PLAYER1_TEAM_ABBREVIATION"]

        shot_times[gameid] = game_shot_times

    queue.put(shot_times)


def get_shot_times():
    q = multiprocessing.Queue()

    dir_fs = os.listdir(TRACKING_DIR)
    all_game_7zs = [dir_f for dir_f in dir_fs if dir_f.endswith(".7z")]
    processes = multiprocessing.cpu_count()
    game_7zs_per_process = int(np.ceil(len(all_game_7zs) / processes))
    jobs = []
    for i in range(processes):
        start = i * game_7zs_per_process
        end = start + game_7zs_per_process
        game_7zs = all_game_7zs[start:end]
        p = multiprocessing.Process(target=get_shot_times_worker, args=(game_7zs, q))
        jobs.append(p)
        p.start()

    shot_times = {}
    for _ in jobs:
        shot_times.update(q.get())

    for p in jobs:
        p.join()

    return shot_times


def fill_in_periods(game_hoop_sides):
    for p1 in range(4):
        if p1 not in game_hoop_sides:
            continue

        adder = 1 if p1 <= 2 else 3
        p2 = (p1 % 2) + adder
        if p2 not in game_hoop_sides:
            game_hoop_sides[p2] = game_hoop_sides[p1].copy()

        period_hoop_sides = list(game_hoop_sides[p1].items())
        swapped = {
            period_hoop_sides[0][0]: period_hoop_sides[1][1],
            period_hoop_sides[1][0]: period_hoop_sides[0][1],
        }

        p2s = [3, 4] if p1 <= 2 else [1, 2]
        for p2 in p2s:
            if p2 not in game_hoop_sides:
                game_hoop_sides[p2] = swapped.copy()

    return game_hoop_sides


def check_periods(game_hoop_sides, team, game):
    for period in range(4):
        if period in {2, 4}:
            if period - 1 in game_hoop_sides:
                assert (
                    game_hoop_sides[period - 1][team] == game_hoop_sides[period][team]
                ), f"{team} has different sides in periods {period} and {period - 1} of {game}."

        if period in {3, 4}:
            for first_half in [1, 2]:
                if first_half in game_hoop_sides:
                    assert (
                        game_hoop_sides[first_half][team]
                        != game_hoop_sides[period][team]
                    ), f"{team} has same side in periods {first_half} and {period} of {game}."


def get_game_hoop_sides(teams, hoop_side_counts, game):
    [team_a, team_b] = list(teams)
    game_hoop_sides = {period: {} for period in hoop_side_counts}
    do_check = False
    periods = list(hoop_side_counts)
    periods.sort()
    for period in periods:
        if len(hoop_side_counts[period]) == 0:
            print(f"No shooting data for {period} of {game}.")
            continue

        for team in teams:
            if team not in hoop_side_counts[period]:
                print(f"Missing {team} for {period} of {game}.")
                hoop_side_counts[period][team] = {0: 0, COURT_LENGTH: 0}

            l_count = hoop_side_counts[period][team][0]
            r_count = hoop_side_counts[period][team][COURT_LENGTH]
            if l_count > r_count:
                game_hoop_sides[period][team] = 0
            elif l_count < r_count:
                game_hoop_sides[period][team] = COURT_LENGTH
            else:
                do_check = True

        if do_check:
            team_in = team_a if team_a in game_hoop_sides[period] else team_b
            team_out = team_a if team_a not in game_hoop_sides[period] else team_b
            hoop_side = game_hoop_sides[period][team_in]
            if hoop_side == 0:
                game_hoop_sides[period][team_out] = 0
            else:
                game_hoop_sides[period][team_out] = COURT_LENGTH

            do_check = False

        assert (
            game_hoop_sides[period][team_a] != game_hoop_sides[period][team_b]
        ), f"{team_a} and {team_b} have same side in {period} of {game}."

    game_hoop_sides = fill_in_periods(game_hoop_sides)

    for team in teams:
        check_periods(game_hoop_sides, team, game)

    return game_hoop_sides


def get_hoop_sides_worker(game_7zs, queue):
    hoop_sides = {}
    for game_7z in game_7zs:
        game_name = game_7z.split(".7z")[0]

        try:
            gameid = os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0]
        except FileNotFoundError:
            continue

        df_tracking = pd.read_json(f"{TRACKING_DIR}/{game_name}/{gameid}.json")
        hoop_side_counts = {}
        used_game_times = set()
        teams = set()
        for tracking_event in df_tracking["events"]:
            for moment in tracking_event["moments"]:
                period = moment[0]
                game_clock = moment[2]
                game_time = int(get_game_time(game_clock, period))
                if (game_time in shot_times[gameid]) and (
                    game_time not in used_game_times
                ):
                    ball_x = moment[5][0][2]

                    if ball_x < HALF_COURT_LENGTH:
                        hoop_side = 0
                    else:
                        hoop_side = COURT_LENGTH

                    if period not in hoop_side_counts:
                        hoop_side_counts[period] = {}

                    shooting_team = shot_times[gameid][game_time]
                    if shooting_team not in hoop_side_counts[period]:
                        hoop_side_counts[period][shooting_team] = {
                            0: 0,
                            COURT_LENGTH: 0,
                        }

                    hoop_side_counts[period][shooting_team][hoop_side] += 1
                    used_game_times.add(game_time)
                    teams.add(shooting_team)

        if len(teams) == 0:
            print(f"The moments in the {game_name} JSON are empty.", flush=True)
            continue

        hoop_sides[gameid] = get_game_hoop_sides(teams, hoop_side_counts, game_name)

    queue.put(hoop_sides)


def get_team_hoop_sides():
    q = multiprocessing.Queue()

    dir_fs = os.listdir(TRACKING_DIR)
    all_game_7zs = [dir_f for dir_f in dir_fs if dir_f.endswith(".7z")]
    processes = multiprocessing.cpu_count()
    game_7zs_per_process = int(np.ceil(len(all_game_7zs) / processes))
    jobs = []
    for i in range(processes):
        start = i * game_7zs_per_process
        end = start + game_7zs_per_process
        game_7zs = all_game_7zs[start:end]
        p = multiprocessing.Process(target=get_hoop_sides_worker, args=(game_7zs, q))
        jobs.append(p)
        p.start()

    hoop_sides = {}
    for _ in jobs:
        hoop_sides.update(q.get())

    for p in jobs:
        p.join()

    return hoop_sides


def get_event_stream(gameid):
    df_events = pd.read_csv(f"{EVENTS_DIR}/{gameid}.csv")
    df_events = df_events.fillna("")
    df_events["DESCRIPTION"] = (
        df_events["HOMEDESCRIPTION"]
        + " "
        + df_events["NEUTRALDESCRIPTION"]
        + " "
        + df_events["VISITORDESCRIPTION"]
    )

    # Posession times.
    # EVENTMSGTYPE descriptions can be found at: https://github.com/rd11490/NBA_Tutorials/tree/master/analyze_play_by_play.
    event_col = "EVENTMSGTYPE"
    description_col = "DESCRIPTION"
    player1_team_col = "PLAYER1_TEAM_ABBREVIATION"

    teams = list(df_events[player1_team_col].unique())
    teams.sort()
    teams = teams[1:] if len(teams) > 2 else teams

    event = None
    score = "0 - 0"
    # pos_team is the team that had possession prior to the event.
    (pos_team, pos_team_idx) = (None, None)
    jump_ball_team_idx = None
    # I think most of these are technical fouls.
    skip_fouls = {10, 11, 16, 19}

    events = set()
    pos_stream = []
    event_stream = []
    for (row_idx, row) in df_events.iterrows():
        period = row["PERIOD"]
        game_clock = row["PCTIMESTRING"].split(":")
        game_clock_secs = 60 * int(game_clock[0]) + int(game_clock[1])
        game_time = get_game_time(game_clock_secs, period)

        description = row[description_col].lower().strip()

        eventmsgtype = row[event_col]

        # Don't know.
        if eventmsgtype == 18:
            continue

        # Blank line.
        elif eventmsgtype == 14:
            continue

        # End of a period.
        elif eventmsgtype == 13:
            if period == 4:
                jump_ball_team_idx = None

            continue

        # Start of a period.
        elif eventmsgtype == 12:
            if 2 <= period <= 4:
                if period == 4:
                    pos_team_idx = jump_ball_team_idx
                else:
                    pos_team_idx = (jump_ball_team_idx + 1) % 2

            elif 6 <= period:
                pos_team_idx = (jump_ball_team_idx + (period - 5)) % 2

            continue

        # Ejection.
        elif eventmsgtype == 11:
            continue

        # Jump ball.
        elif eventmsgtype == 10:
            if int(row["PLAYER3_ID"]) in TEAM_ID2PROPS:
                pos_team = TEAM_ID2PROPS[int(row["PLAYER3_ID"])]["abbreviation"]
            else:
                pos_team = row["PLAYER3_TEAM_ABBREVIATION"]

            if pos_team == teams[0]:
                pos_team_idx = 0
            else:
                pos_team_idx = 1

            if (period in {1, 5}) and (jump_ball_team_idx is None):
                jump_ball_team_idx = pos_team_idx

            continue

        # Timeout.
        elif eventmsgtype == 9:
            # TV timeout?
            if description == "":
                continue

            event = "timeout"

        # Substitution.
        elif eventmsgtype == 8:
            continue

        # Violation.
        elif eventmsgtype == 7:
            # With 35 seconds left in the fourth period, there was a kicked ball
            # violation attributed to Wayne Ellington of the Brooklyn Nets, but the
            # following event is a shot by the Nets, which means possession never changed.
            if (gameid == "0021500414") and (row_idx == 427):
                continue

            # Goaltending is considered a made shot, so the following event is always the
            # made shot event.
            if "goaltending" in description:
                score = row["SCORE"] if row["SCORE"] else score
                continue
            # Jump ball violations have weird possession rules.
            elif "jump ball" in description:
                if row["PLAYER1_TEAM_ABBREVIATION"] == teams[0]:
                    pos_team_idx = 1
                else:
                    pos_team_idx = 0

                pos_team = teams[pos_team_idx]
                if (period == 1) and (game_time == 0):
                    jump_ball_team_idx = pos_team_idx

                continue

            else:
                if row[player1_team_col] == teams[pos_team_idx]:
                    event = "violation_offense"
                    pos_team_idx = (pos_team_idx + 1) % 2
                else:
                    event = "violation_defense"

        # Foul.
        elif eventmsgtype == 6:
            # Skip weird fouls.
            if row["EVENTMSGACTIONTYPE"] in skip_fouls:
                score = row["SCORE"] if row["SCORE"] else score
                continue
            else:
                if row[player1_team_col] == teams[pos_team_idx]:
                    event = "offensive_foul"
                    pos_team_idx = (pos_team_idx + 1) % 2
                else:
                    event = "defensive_foul"

        # Turnover.
        elif eventmsgtype == 5:
            if "steal" in description:
                event = "steal"
            elif "goaltending" in description:
                event = "goaltending_offense"
            elif (
                ("violation" in description)
                or ("dribble" in description)
                or ("traveling" in description)
            ):
                event = "violation_offense"
            else:
                event = "turnover"

            # Team turnover.
            if row[player1_team_col] == "":
                team_id = int(row["PLAYER1_ID"])
                team_abb = TEAM_ID2PROPS[team_id]["abbreviation"]
            else:
                team_abb = row[player1_team_col]

            pos_team_idx = 1 if team_abb == teams[0] else 0

        # Rebound.
        elif eventmsgtype == 4:
            # With 17 seconds left in the first period, Spencer Hawes missed a tip in,
            # which was rebounded by DeAndre Jordan. The tip in is recorded as a rebound
            # and a missed shot for Hawes. All three events have the same timestamp,
            # which seems to have caused the order of the events to be slightly shuffled
            # with the Jordan rebound occurring before the tip in.
            if (gameid == "0021500550") and (row_idx == 97):
                continue

            # Team rebound.
            if row[player1_team_col] == "":
                team_id = int(row["PLAYER1_ID"])
                team_abb = TEAM_ID2PROPS[team_id]["abbreviation"]
                if team_abb == teams[pos_team_idx]:
                    event = "rebound_offense"
                else:
                    event = "rebound_defense"
                    pos_team_idx = (pos_team_idx + 1) % 2

            elif row[player1_team_col] == teams[pos_team_idx]:
                event = "rebound_offense"
            else:
                event = "rebound_defense"
                pos_team_idx = (pos_team_idx + 1) % 2

        # Free throw.
        elif eventmsgtype == 3:
            # See rules for technical fouls: https://official.nba.com/rule-no-12-fouls-and-penalties/.
            # Possession only changes for too many players, which is extremely rare.
            if "technical" not in description:
                pos_team_idx = 0 if row[player1_team_col] == teams[0] else 1
                if (
                    ("Clear Path" not in row[description_col])
                    and ("Flagrant" not in row[description_col])
                    and ("MISS" not in row[description_col])
                    and (
                        ("1 of 1" in description)
                        or ("2 of 2" in description)
                        or ("3 of 3" in description)
                    )
                ):
                    # Hack to handle foul shots for away from play fouls.
                    if ((gameid == "0021500274") and (row_idx == 519)) or (
                        (gameid == "0021500572") and (row_idx == 428)
                    ):
                        pass
                    # This event is a made foul shot by Thaddeus Young of the Brooklyn
                    # Nets following an and-one foul, so possession should have changed
                    # to the Milwaukee Bucks. However, the next event is a made shot by
                    # Brook Lopez (also of the Brooklyn Nets) with no event indicating a
                    # change of possession occurring before it.
                    elif (gameid == "0021500047") and (row_idx == 64):
                        pass
                    else:
                        pos_team_idx = (pos_team_idx + 1) % 2

                pos_team = teams[pos_team_idx]

            score = row["SCORE"] if row["SCORE"] else score

            continue

        # Missed shot.
        elif eventmsgtype == 2:
            if "dunk" in description:
                shot_type = "dunk"
            elif "layup" in description:
                shot_type = "layup"
            else:
                shot_type = "shot"

            if "BLOCK" in row[description_col]:
                miss_type = "block"
            else:
                miss_type = "miss"

            event = f"{shot_type}_{miss_type}"

            if row[player1_team_col] != teams[pos_team_idx]:
                print(pos_stream[-5:])
                raise ValueError(f"Incorrect possession team in row {str(row_idx)}.")

        # Made shot.
        elif eventmsgtype == 1:
            if "dunk" in description:
                shot_type = "dunk"
            elif "layup" in description:
                shot_type = "layup"
            else:
                shot_type = "shot"

            event = f"{shot_type}_made"

            if row[player1_team_col] != teams[pos_team_idx]:
                print(pos_stream[-5:])
                raise ValueError(f"Incorrect possession team in row {str(row_idx)}.")

            pos_team_idx = (pos_team_idx + 1) % 2

        events.add(event)
        pos_stream.append(pos_team_idx)

        if row[player1_team_col] == "":
            team_id = int(row["PLAYER1_ID"])
            event_team = TEAM_ID2PROPS[team_id]["abbreviation"]
        else:
            event_team = row[player1_team_col]

        event_stream.append(
            {
                "game_time": game_time - 1,
                "pos_team": pos_team,
                "event": event,
                "description": description,
                "event_team": event_team,
                "score": score,
            }
        )

        # With 17 seconds left in the first period, Spencer Hawes missed a tip in,
        # which was rebounded by DeAndre Jordan. The tip in is recorded as a rebound
        # and a missed shot for Hawes. All three events have the same timestamp,
        # which seems to have caused the order of the events to be slightly shuffled
        # with the Jordan rebound occurring before the tip in.
        if (gameid == "0021500550") and (row_idx == 98):
            event_stream.append(
                {
                    "game_time": game_time - 1,
                    "pos_team": pos_team,
                    "event": "rebound_defense",
                    "description": "jordan rebound (off:2 def:5)",
                    "event_team": "LAC",
                    "score": score,
                }
            )
            pos_team_idx = (pos_team_idx + 1) % 2

        # This event is a missed shot by Kawhi Leonard of the San Antonio Spurs. The next
        # event is a missed shot by Bradley Beal of the Washington Wizards with no
        # event indicating a change of possession occurring before it.
        if (gameid == "0021500061") and (row_idx == 240):
            pos_team_idx = (pos_team_idx + 1) % 2

        pos_team = teams[pos_team_idx]
        score = row["SCORE"] if row["SCORE"] else score

    return event_stream


def get_event_streams_worker(game_names, queue):
    gameid2event_stream = {}
    for (game_idx, game_name) in enumerate(game_names):
        try:
            gameid = os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0]
            gameid2event_stream[gameid] = get_event_stream(gameid)
        except IndexError:
            continue

    queue.put(gameid2event_stream)


def get_event_streams():
    q = multiprocessing.Queue()

    dir_fs = os.listdir(TRACKING_DIR)
    all_game_names = [dir_f for dir_f in dir_fs if not dir_f.endswith(".7z")]

    processes = multiprocessing.cpu_count()
    game_names_per_process = int(np.ceil(len(all_game_names) / processes))
    jobs = []
    for i in range(processes):
        start = i * game_names_per_process
        end = start + game_names_per_process
        game_names = all_game_names[start:end]
        p = multiprocessing.Process(
            target=get_event_streams_worker, args=(game_names, q)
        )
        jobs.append(p)
        p.start()

    gameid2event_stream = {}
    for _ in jobs:
        gameid2event_stream.update(q.get())

    for p in jobs:
        p.join()

    event2event_idx = {}
    for event_stream in gameid2event_stream.values():
        for event in event_stream:
            event2event_idx.setdefault(event["event"], len(event2event_idx))

    return (event2event_idx, gameid2event_stream)


def add_score_changes(X):
    score_diff_idx = 6
    period_idx = 3
    wall_clock_idx = -1

    score_change_idxs = np.where(np.diff(X[:, score_diff_idx]) != 0)[0] + 1
    score_changes = (
        X[score_change_idxs, score_diff_idx] - X[score_change_idxs - 1, score_diff_idx]
    )

    # Score changes at the half are the result of the teams changing sides.
    half_idx = -1
    if (X[:, period_idx].min() <= 2) and (X[:, period_idx].max() >= 3):
        period_change_idxs = np.where(np.diff(X[:, 3]) != 0)[0] + 1
        for period_change_idx in period_change_idxs:
            before_period = X[period_change_idx - 1, period_idx]
            after_period = X[period_change_idx, period_idx]
            if (before_period <= 2) and (after_period > 2):
                half_idx = period_change_idx
                break

    else:
        half_idx = -1

    cur_score_change_idx = 0
    times_to_next_score_change = []
    next_score_changes = []
    end_time = X[-1, wall_clock_idx]
    for (idx, row) in enumerate(X):
        try:
            if idx == score_change_idxs[cur_score_change_idx]:
                cur_score_change_idx += 1

            score_change_idx = score_change_idxs[cur_score_change_idx]
            next_score_time = X[score_change_idx, wall_clock_idx]
            if score_change_idx == half_idx:
                next_score_change = 0
            else:
                next_score_change = score_changes[cur_score_change_idx]

        except IndexError:
            next_score_time = end_time
            next_score_change = 0

        cur_time = X[idx, wall_clock_idx]
        time_to_next_score_change = (next_score_time - cur_time) / 1000
        times_to_next_score_change.append(time_to_next_score_change)
        next_score_changes.append(next_score_change)

    times_to_next_score_change = np.array(times_to_next_score_change)[None].T
    next_score_changes = np.array(next_score_changes)[None].T

    X = np.hstack(
        [
            X[:, :wall_clock_idx],
            times_to_next_score_change,
            next_score_changes,
            X[:, wall_clock_idx:],
        ]
    )
    return X


def save_game_numpy_arrays(game_name):
    try:
        gameid = os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0]
    except IndexError:
        shutil.rmtree(f"{TRACKING_DIR}/{game_name}")
        return

    if gameid not in gameid2event_stream:
        print(f"Missing gameid: {gameid}", flush=True)
        return

    df_tracking = pd.read_json(f"{TRACKING_DIR}/{game_name}/{gameid}.json")
    home_team = None
    cur_time = -1
    event_idx = 0
    game_over = False
    X = []
    y = []
    event_stream = gameid2event_stream[gameid]
    for tracking_event in df_tracking["events"]:
        event_id = tracking_event["eventId"]
        if home_team is None:
            home_team_id = tracking_event["home"]["teamid"]
            home_team = TEAM_ID2PROPS[home_team_id]["abbreviation"]

        moments = tracking_event["moments"]
        for moment in moments:
            period = moment[0]
            # Milliseconds.
            wall_clock = moment[1]
            game_clock = moment[2]
            shot_clock = moment[3]
            shot_clock = shot_clock if shot_clock else game_clock

            period_time = 720 - game_clock if period <= 4 else 300 - game_clock
            game_time = get_game_time(game_clock, period)

            # Moments can overlap temporally, so previously processed time points are
            # skipped along with clock stoppages.
            if game_time <= cur_time:
                continue

            while game_time > event_stream[event_idx]["game_time"]:
                event_idx += 1
                if event_idx >= len(event_stream):
                    game_over = True
                    break

            if game_over:
                break

            event = event_stream[event_idx]
            score = event["score"]
            (away_score, home_score) = (int(s) for s in score.split(" - "))
            home_hoop_side = hoop_sides[gameid][period][home_team]
            if home_hoop_side == 0:
                (left_score, right_score) = (home_score, away_score)
            else:
                (right_score, left_score) = (home_score, away_score)

            (ball_x, ball_y, ball_z) = moment[5][0][2:5]
            data = [
                game_time,
                period_time,
                shot_clock,
                period,
                left_score,  # off_score,
                right_score,  # def_score,
                left_score - right_score,
                ball_x,
                ball_y,
                ball_z,
            ]

            if len(moment[5][1:]) != 10:
                continue

            player_idxs = []
            player_xs = []
            player_ys = []
            player_hoop_sides = []

            try:
                for player in moment[5][1:]:
                    player_idxs.append(playerid2player_idx[player[1]])
                    player_xs.append(player[2])
                    player_ys.append(player[3])
                    hoop_side = hoop_sides[gameid][period][
                        TEAM_ID2PROPS[player[0]]["abbreviation"]
                    ]
                    player_hoop_sides.append(int(hoop_side == COURT_LENGTH))

            except KeyError:
                if player[1] == 0:
                    print(
                        f"Bad player in event {event_id} for {game_name}.", flush=True
                    )
                    continue

                else:
                    raise KeyError

            order = np.argsort(player_idxs)
            for idx in order:
                data.append(player_idxs[idx])

            for idx in order:
                data.append(player_xs[idx])

            for idx in order:
                data.append(player_ys[idx])

            for idx in order:
                data.append(player_hoop_sides[idx])

            data.append(event_idx)
            data.append(wall_clock)

            if len(data) != 52:
                raise ValueError

            X.append(np.array(data))
            y.append(event2event_idx.setdefault(event["event"], len(event2event_idx)))
            cur_time = game_time

        if game_over:
            break

    X = np.stack(X)
    y = np.array(y)

    X = add_score_changes(X)

    np.save(f"{GAMES_DIR}/{gameid}_X.npy", X)
    np.save(f"{GAMES_DIR}/{gameid}_y.npy", y)


def save_numpy_arrays_worker(game_names):
    for game_name in game_names:
        try:
            save_game_numpy_arrays(game_name)
        except ValueError:
            pass

        shutil.rmtree(f"{TRACKING_DIR}/{game_name}")


def save_numpy_arrays():
    dir_fs = os.listdir(TRACKING_DIR)
    all_game_names = [dir_f for dir_f in dir_fs if not dir_f.endswith(".7z")]

    processes = multiprocessing.cpu_count()
    game_names_per_process = int(np.ceil(len(all_game_names) / processes))
    jobs = []
    for i in range(processes):
        start = i * game_names_per_process
        end = start + game_names_per_process
        game_names = all_game_names[start:end]
        p = multiprocessing.Process(target=save_numpy_arrays_worker, args=(game_names,))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()


def player_idx2playing_time_map_worker(gameids, queue):
    player_idx2playing_time = {}
    for gameid in gameids:
        X = np.load(f"{GAMES_DIR}/{gameid}_X.npy")
        wall_clock_diffs = np.diff(X[:, -1]) / 1000
        all_player_idxs = X[:, 10:20].astype(int)
        prev_players = set(all_player_idxs[0])
        for (row_idx, player_idxs) in enumerate(all_player_idxs[1:]):
            current_players = set(player_idxs)
            if len(prev_players & current_players) == 10:
                wall_clock_diff = wall_clock_diffs[row_idx]
                if wall_clock_diff < THRESHOLD:
                    for player_idx in current_players:
                        player_idx2playing_time[player_idx] = (
                            player_idx2playing_time.get(player_idx, 0) + wall_clock_diff
                        )

            prev_players = current_players

    queue.put(player_idx2playing_time)


def get_player_idx2playing_time_map():
    q = multiprocessing.Queue()

    all_gameids = list(set([np_f.split("_")[0] for np_f in os.listdir(GAMES_DIR)]))
    processes = multiprocessing.cpu_count()
    gameids_per_process = int(np.ceil(len(all_gameids) / processes))
    jobs = []
    for i in range(processes):
        start = i * gameids_per_process
        end = start + gameids_per_process
        gameids = all_gameids[start:end]
        p = multiprocessing.Process(
            target=player_idx2playing_time_map_worker, args=(gameids, q)
        )
        jobs.append(p)
        p.start()

    player_idx2playing_time = {}
    for _ in jobs:
        game_player_idx2_playing_time = q.get()
        for (player_idx, playing_time) in game_player_idx2_playing_time.items():
            player_idx2playing_time[player_idx] = (
                player_idx2playing_time.get(player_idx, 0) + playing_time
            )

    for p in jobs:
        p.join()

    playing_times = list(player_idx2playing_time.values())
    print(np.quantile(playing_times, [0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9]))
    # [ 3364.6814     10270.2988     13314.768      39917.09399999
    #  59131.73249999 63400.76839999 72048.72879999]
    return player_idx2playing_time


if __name__ == "__main__":
    os.makedirs(GAMES_DIR, exist_ok=True)

    (playerid2player_idx, player_idx2props) = get_playerid2player_idx_map()

    try:
        baller2vec_config = pickle.load(
            open(f"{DATA_DIR}/baller2vec_config.pydict", "rb")
        )
        player_idx2props = baller2vec_config["player_idx2props"]
        event2event_idx = baller2vec_config["event2event_idx"]
        playerid2player_idx = {}
        for (player_idx, props) in player_idx2props.items():
            playerid2player_idx[props["playerid"]] = player_idx

    except FileNotFoundError:
        baller2vec_config = False

    shot_times = get_shot_times()
    hoop_sides = get_team_hoop_sides()
    (event2event_idx, gameid2event_stream) = get_event_streams()
    if baller2vec_config:
        event2event_idx = baller2vec_config["event2event_idx"]

    save_numpy_arrays()

    player_idx2playing_time = get_player_idx2playing_time_map()
    for (player_idx, playing_time) in player_idx2playing_time.items():
        player_idx2props[player_idx]["playing_time"] = playing_time

    if not baller2vec_config:
        baller2vec_config = {
            "player_idx2props": player_idx2props,
            "event2event_idx": event2event_idx,
        }
        pickle.dump(
            baller2vec_config, open(f"{DATA_DIR}/baller2vec_config.pydict", "wb")
        )
