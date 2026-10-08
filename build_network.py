from pathlib import Path

import pandas as pd
import networkx as nx

def load_events(file_path):
    events = pd.read_csv(file_path, dtype = str, keep_default_na = False)
    events.columns = (
        events.columns.str.strip()
        .str.lstrip("\ufeff")
        .str.replace("_", " ", regex = False)
    )

    required = [
        "Home Team", "Away Team", "Team", "Period", "Clock", 
        "Event", "Home Team Skaters", "Away Team Skaters",
    ]
    for column in required:
        events[column] = events[column].str.strip()

    for column in ["Home Team Skaters", "Away Team Skaters"]:
        counts = pd.to_numeric(events[column], errors = "coerce")
        events[column] = counts.astype(int)

    events["source_csv_line"] = events.index + 2
    return events

def isolate_powerplay_events(events, team):
    events = events.copy()

    events["clock_seconds"] = pd.to_timedelta(
        "00:" + events["Clock"]
    ).dt.total_seconds()

    events["event_order"] = (
        events["Event"] != "Penalty Taken"
    ).astype(int)

    events = events.sort_values(
        ["Period", "clock_seconds", "event_order", "source_csv_line"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    if team == events["Home Team"].iloc[0]:
        our_skaters = events["Home Team Skaters"]
        their_skaters = events["Away Team Skaters"]

    elif team == events["Away Team"].iloc[0]:
        our_skaters = events["Away Team Skaters"]
        their_skaters = events["Home Team Skaters"]

    on_powerplay = (our_skaters == 5) & (their_skaters == 4)
    new_powerplay = on_powerplay & (~on_powerplay.shift(fill_value = False) | events["Period"].ne(events["Period"].shift()))

    events["powerplay_id"] = new_powerplay.cumsum()

    return events.loc[on_powerplay].copy()

def isolate_offensive_zone_events(events, attack_directions):
    events = events.copy()

    periods = pd.to_numeric(events["Period"], errors = "raise")
    x_coordinates = pd.to_numeric(events["X Coordinate"], errors = "coerce")

    directions = periods.map(attack_directions)
    if directions.isna().any():
        raise ValueError("Missing attacking direction for a period")

    events["Attacking X"] = x_coordinates * directions
    offensive_zone_events = events.loc[events["Attacking X"] >= 25].copy()

    return offensive_zone_events

def build_player_graph(events, use_positions, player_positions):
    graph = nx.DiGraph()

    graph.add_node("Zone Entry", node_type="action")
    graph.add_node("Shot", node_type="action")

    actions = events.loc[
        events["Event"].isin(["Play", "Zone Entry", "Shot", "Goal"])
    ].copy()

    if use_positions:
        actions["Actor"] = (
            actions["Player Id"].str.strip().map(player_positions)
        )
        actions["Receiver"] = (
            actions["Player Id 2"].str.strip().map(player_positions)
        )

        missing_role = (
            actions["Actor"].isna()
            | (
                actions["Event"].eq("Play")
                & actions["Receiver"].isna()
            )
        )

        if missing_role.any():
            raise ValueError("Add a position for every player involved in these actions")

    else:
        actions["Actor"] = actions["Player Id"].str.strip()
        actions["Receiver"] = actions["Player Id 2"].str.strip()

        missing_id = (
            actions["Actor"].eq("")
            | (
                actions["Event"].eq("Play")
                & actions["Receiver"].eq("")
            )
        )

        if missing_id.any():
            raise ValueError("An action has a missing player ID")

    node_type = "position" if use_positions else "player"

    for event, actor, receiver in actions[
        ["Event", "Actor", "Receiver"]
    ].itertuples(index=False, name = None):

        graph.add_node(actor, node_type = node_type)

        if event == "Play":
            graph.add_node(receiver, node_type=node_type)
            source, target = actor, receiver

        elif event == "Zone Entry":
            source, target = "Zone Entry", actor

        else:
            source, target = actor, "Shot"

        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1
        else:
            graph.add_edge(source, target, weight=1)

    for node in graph.nodes():
        graph.nodes[node]["label"] = str(node)

    for source, target, data in graph.edges(data=True):
        data["label"] = str(data["weight"])

    return graph

def main():
    events_file = (Path.cwd() / "data" /  "A_D_Oct_11" / "2025-10-11.Team.A.@.Team.D.Events.csv")
    output_path = Path.cwd() / "output"

    team = "Team A"
    attack_directions = {1: -1, 2: 1, 3: -1}
    selected_powerplay_id = 1
    use_positions = True
    player_positions = {  # placeholders!
        "59": "Point",
        "88": "Left Flank",
        "9": "Bumper",
        "14": "Net-Front",
        "5": "Point",
        "92": "Left Flank",
        "29": "Right Flank",
        "39": "Bumper",
        "46": "Net-Front",
        "63": "Left Flank",
    }

    events = load_events(events_file)
    powerplay_events = isolate_powerplay_events(events, team)

    print(
        powerplay_events.groupby("powerplay_id").agg(
            period = ("Period", "first"),
            first_clock = ("Clock", "first"),
            last_clock = ("Clock", "last"),
        )
    )

    single_powerplay = powerplay_events.loc[
        (powerplay_events["powerplay_id"] == selected_powerplay_id)
        & (powerplay_events["Team"] == team)
    ].copy()

    attacker_pp_ozone_events = isolate_offensive_zone_events(
        single_powerplay, attack_directions
    )

    graph_events = attacker_pp_ozone_events.copy()

    graph = build_player_graph(
        graph_events,
        use_positions,
        player_positions,
    )

    if use_positions:
        positions = {
            "Point": (0, 0),
            "Left Flank": (-60, 50),
            "Right Flank": (60, 50),
            "Bumper": (0, 80),
            "Net-Front": (0, 120),
            "Zone Entry": (0, -80),
            "Shot": (0, 190),
        }

        layout_scale = 3.0
        for node in graph.nodes():
            if node in positions:
                x, y = positions[node]

                graph.nodes[node]["viz"] = {
                    "position": {
                        "x": float(x * layout_scale),
                        "y": float(y * layout_scale),
                        "z": 0.0,
                    }
                }

    output_path.mkdir(parents = True, exist_ok = True)
    graph_type = "positions" if use_positions else "players"
    nx.write_gexf(
        graph,
        output_path
        / f"powerplay_{selected_powerplay_id}_{graph_type}.gexf",
    )

    print("")
    preview_columns = ["Period", "Clock", "Team", "Event", "Player Id", "X Coordinate", "Y Coordinate", "Detail 1", "Detail 2"]
    print(attacker_pp_ozone_events[preview_columns].head(20).to_string(index = False))

if __name__ == "__main__":
    main()