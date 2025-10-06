"""
A script that animates tracking data, given gameId and playId. 
Players can be identified by mousing over the individuals dots. 
The play description is also displayed at the bottom of the plot, 
together with play information at the top of the plot. 

Data should be stored in a dir named data, in the same dir as this script. 

Original Source: https://www.kaggle.com/code/huntingdata11/animated-and-interactive-nfl-plays-in-plotly/notebook
"""

import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from custom_assets import colors

pio.renderers.default = (
    "browser"  # modify this to plot on something else besides browser
)


script_dir = Path(__file__).parent  
data_path = (script_dir / "../../data").resolve()


# Modify the variables below to plot your desired play
input_data_loc = "train/input_2023_w01.csv"
output_data_loc = "train/output_2023_w01.csv"

game_id = 2023090700
play_id = 101
supplementary_file = "supplementary_data.csv"

input_data_file = data_path / input_data_loc
output_data_file = data_path / output_data_loc
info_file = data_path / supplementary_file

df_input = pd.read_csv(input_data_file)
df_output = pd.read_csv(output_data_file)
df_plays = pd.read_csv(info_file)


df_input_merged = df_input.merge(df_plays, on=["game_id", "play_id"])
df_output_merged = df_output.merge(df_plays, on=["game_id", "play_id"])


drop_list = ['penalty_yards',
       'pre_penalty_yards_gained', 'yards_gained', 'expected_points',
       'expected_points_added', 'pre_snap_home_team_win_probability',
       'pre_snap_visitor_team_win_probability',
       'home_team_win_probability_added', 'visitor_team_win_probility_added', 'play_nullified_by_penalty']

df_input_final = df_input_merged.drop(drop_list, axis=1)
df_output_merged = df_output_merged.drop(drop_list, axis=1)

# Create a mapping of player roles from pre-pass data
# Each player has one role per play
role_mapping = df_input_final[['game_id', 'play_id', 'nfl_id', 'player_role']].drop_duplicates()
df_output_final = df_output_merged.merge(
    role_mapping,
    on=['game_id', 'play_id', 'nfl_id'],
    how='left'
)

df_output_final['player_to_predict'] = True


df_input_focused = df_input_final[
    (df_input_final["play_id"] == play_id) & (df_input_final["game_id"] == game_id)
]

df_output_focused = df_output_final[
    (df_output_final["play_id"] == play_id) & (df_output_final["game_id"] == game_id)
]


# Concat the two dataframes - merge the two segments of the play into one
pass_frame_id = df_input_focused['frame_id'].max() + 1
df_output_focused['frame_id'] = df_output_focused['frame_id'] + df_input_focused['frame_id'].max()
df_focused = pd.concat([df_input_focused, df_output_focused], ignore_index=True)


# Get General Play Information
absolute_yd_line = df_focused.absolute_yardline_number.values[0]
play_going_right = (
    df_focused.play_direction.values[0] == "right"
)  # 0 if left, 1 if right

line_of_scrimmage = absolute_yd_line
ball_land_x = df_focused.ball_land_x.values[0]
ball_land_y = df_focused.ball_land_y.values[0]


print(f'Play going {"right" if play_going_right else "left"} starting on the {absolute_yd_line} yd line')

# place LOS depending on play direction and absolute_yd_line. 110 because absolute_yd_line includes endzone width

first_down_marker = (
    (line_of_scrimmage + df_focused.yards_to_go.values[0])
    if play_going_right
    else (line_of_scrimmage - df_focused.yards_to_go.values[0])
)  # Calculate 1st down marker


down = df_focused.down.values[0]
quarter = df_focused.quarter.values[0]
game_clock = df_focused.game_clock.values[0]
play_description = df_focused.play_description.values[0]


# Handle case where we have a really long Play Description and want to split it into two lines
if len(play_description.split(" ")) > 15 and len(play_description) > 115:
    play_description = (
        " ".join(play_description.split(" ")[0:16])
        + "<br>"
        + " ".join(play_description.split(" ")[16:])
    )

print(
    f"Line of Scrimmage: {line_of_scrimmage}, First Down Marker: {first_down_marker}, Down: {down}, Quarter: {quarter}, Game Clock: {game_clock}, Play Description: {play_description}"
)



# initialize plotly play and pause buttons for animation
updatemenus_dict = [
    {
        "buttons": [
            {
                "args": [
                    None,
                    {
                        "frame": {"duration": 100, "redraw": False},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    },
                ],
                "label": "Play",
                "method": "animate",
            },
            {
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": "Pause",
                "method": "animate",
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top",
    }
]



# initialize plotly slider to show frame position in animation
sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Frame:",
        "visible": True,
        "xanchor": "right",
    },
    "transition": {"duration": 300, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": [],
}

# Frame Info
sorted_frame_list = df_focused.frame_id.unique()
sorted_frame_list.sort()

frames = []
data_history = []
for frameId in sorted_frame_list:
    data = []
    # Add Yardline Numbers to Field
    data.append(
        go.Scatter(
            x=np.arange(20, 110, 10),
            y=[5] * len(np.arange(20, 110, 10)),
            mode="text",
            text=list(
                map(str, list(np.arange(20, 61, 10) - 10) + list(np.arange(40, 9, -10)))
            ),
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo="none",
        )
    )
    data.append(
        go.Scatter(
            x=np.arange(20, 110, 10),
            y=[53.5 - 5] * len(np.arange(20, 110, 10)),
            mode="text",
            text=list(
                map(str, list(np.arange(20, 61, 10) - 10) + list(np.arange(40, 9, -10)))
            ),
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo="none",
        )
    )
    # Add line of scrimage
    data.append(
        go.Scatter(
            x=[line_of_scrimmage, line_of_scrimmage],
            y=[0, 53.5],
            line_dash="dash",
            line_color="blue",
            showlegend=False,
            hoverinfo="none",
        )
    )
    # Add First down line
    data.append(
        go.Scatter(
            x=[first_down_marker, first_down_marker],
            y=[0, 53.5],
            line_dash="dash",
            line_color="yellow",
            showlegend=False,
            hoverinfo="none",
        )
    )




    
    # Plot Players
    for role in df_focused.player_role.unique():

        plot_df = df_focused[
            (df_focused.player_role == role) & (df_focused.frame_id == frameId)
        ].copy()

    
        hover_text_array = []
        for nflId in plot_df.nfl_id:
            selected_player_df = plot_df[plot_df.nfl_id == nflId]
            hover_text_array.append(
                f"nflId:{selected_player_df['nfl_id'].values[0]}<br>displayName:{selected_player_df['player_name'].values[0]}"
            )
        data.append(
            go.Scatter(
                x=plot_df["x"],
                y=plot_df["y"],
                mode="markers",
                marker_color=colors[role],
                marker_size=10,
                name=role,
                hovertext=hover_text_array,
                hoverinfo="text",
            )
        )


        # Add orientation lines
        line_length = 1.25

        line_x = []
        line_y = []
        
        for i in range(len(plot_df)):
            
            # Only add line if this specific player has orientation data
            if pd.notna(plot_df["o"].iloc[i]):
                x_end_val = plot_df["x"].iloc[i] + line_length * np.sin(np.radians(plot_df["o"].iloc[i]))
                y_end_val = plot_df["y"].iloc[i] + line_length * np.cos(np.radians(plot_df["o"].iloc[i]))
                
                line_x.extend([plot_df["x"].iloc[i], x_end_val, None])
                line_y.extend([plot_df["y"].iloc[i], y_end_val, None])
        

        data.append(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color=colors[role], width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )


    # plot ball landing
    data.append(
            go.Scatter(
                x=np.array(ball_land_x),
                y=np.array(ball_land_y),
                mode="markers",
                marker_color='#895129',
                marker_size=10,
                name='Ball Land',
                hovertext=['Ball!'],
                hoverinfo="text",
            )
    )

    if frameId == pass_frame_id:
        data_history.append(data)
        
    # add frame to slider
    slider_step = {
        "args": [
            [frameId],
            {
                "frame": {"duration": 100, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ],
        "label": str(frameId),
        "method": "animate",
    }
    sliders_dict["steps"].append(slider_step)
    frames.append(go.Frame(data=data, name=str(frameId)))

    
scale = 10
layout = go.Layout(
    autosize=False,
    width=120 * scale,
    height=60 * scale,
    xaxis=dict(
        range=[0, 120],
        autorange=False,
        tickmode="array",
        tickvals=np.arange(10, 111, 5).tolist(),
        showticklabels=False,
    ),
    yaxis=dict(range=[0, 53.3], autorange=False, showgrid=False, showticklabels=False),
    plot_bgcolor="#00B140",
    # Create title and add play description at the bottom of the chart for better visual appeal
    title=f"GameId: {game_id}, PlayId: {play_id}<br>{game_clock} {quarter}Q, Pass at Frame {pass_frame_id}"
    + "<br>" * 19
    + f"{play_description}",
    updatemenus=updatemenus_dict,
    sliders=[sliders_dict],
)

fig = go.Figure(data=frames[0]["data"], layout=layout, frames=frames[1:])
fig.update_layout(
    updatemenus=[{
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 100, "redraw": True},
                               "fromcurrent": True, 
                               "transition": {"duration": 0}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                 "mode": "immediate",
                                 "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }]
)

# Create First Down Markers
for y_val in [0, 53]:
    fig.add_annotation(
        x=first_down_marker,
        y=y_val,
        text=str(down),
        showarrow=False,
        font=dict(family="Courier New, monospace", size=16, color="black"),
        align="center",
        bordercolor="black",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=1,
    )


    fig.show()