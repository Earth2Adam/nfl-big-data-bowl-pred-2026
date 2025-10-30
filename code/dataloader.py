"""
Dataset file for the NFL's tracking data. 
"""
import os
import pathlib
import numpy as np
import glob
import math
import random
import torch
from torch.utils.data import Dataset
import argparse
import pandas as pd




class NFLTrackingDataset(Dataset):
    """
    Dataset for player trajectory prediction with context from nearby players.
    
    Each sample predicts one target player's future trajectory given:
    - Their past trajectory
    - Ball position/trajectory
    - Context from nearby players
    """
    
    def __init__(self, plays_data, input_seq_length=20, output_seq_length=10, 
                 max_nearby_players=5):
        """
        Parameters:
        -----------
        plays_data : list of dicts
            Each dict represents one play with structure:
            {
                'players': DataFrame with columns [frame, player_id, x, y, velocity_x, 
                                                   velocity_y, orientation, team, ...],
                'ball': DataFrame with columns [frame, x, y]
            }
        input_seq_length : int
            Number of frames to use as input (e.g., 20 frames = 2 seconds)
        output_seq_length : int
            Number of frames to predict (e.g., 10 frames = 1 second)
        max_nearby_players : int
            Maximum number of nearby players to include in context
        """
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.max_nearby_players = max_nearby_players
        
        # Create training samples
        self.samples = []
        self._create_samples(plays_data)
        
        # Fit scaler on all data
        self._fit_scaler()
    
    def _create_samples(self, plays_data):
        """Extract individual player trajectory samples from plays."""
        
        for play in plays_data:
            players_df = play['players']
            ball_df = play['ball']
            
            # Get unique players in this play
            player_ids = players_df['player_id'].unique()
            
            # Get total frames
            max_frame = players_df['frame'].max()
            total_seq_length = self.input_seq_length + self.output_seq_length
            
            # Create samples with sliding window
            for start_frame in range(0, max_frame - total_seq_length + 1, 5):  # stride of 5
                end_input_frame = start_frame + self.input_seq_length
                end_output_frame = start_frame + total_seq_length
                
                # For each player, create a training sample
                for target_player_id in player_ids:
                    sample = self._create_player_sample(
                        players_df, ball_df, target_player_id,
                        start_frame, end_input_frame, end_output_frame
                    )
                    
                    if sample is not None:
                        self.samples.append(sample)
        
        print(f"Created {len(self.samples)} training samples")
    
    def _create_player_sample(self, players_df, ball_df, target_player_id,
                             start_frame, end_input_frame, end_output_frame):
        """Create one training sample for a target player."""
        
        # Get target player data
        target_data = players_df[
            (players_df['player_id'] == target_player_id) &
            (players_df['frame'] >= start_frame) &
            (players_df['frame'] < end_output_frame)
        ].sort_values('frame')
        
        # Need complete sequence
        if len(target_data) != (end_output_frame - start_frame):
            return None
        
        # Split into input and output sequences
        input_data = target_data[target_data['frame'] < end_input_frame]
        output_data = target_data[target_data['frame'] >= end_input_frame]
        
        # Get ball data for input sequence
        ball_input = ball_df[
            (ball_df['frame'] >= start_frame) &
            (ball_df['frame'] < end_input_frame)
        ].sort_values('frame')
        
        # Build input features for each frame
        input_features = []
        for idx, row in input_data.iterrows():
            frame = row['frame']
            
            # Get other players at this frame
            other_players = players_df[
                (players_df['frame'] == frame) &
                (players_df['player_id'] != target_player_id)
            ]
            
            # Get ball at this frame
            ball_row = ball_input[ball_input['frame'] == frame].iloc[0]
            
            # Combine target player features with context
            features = self._get_frame_features(row, other_players, ball_row)
            input_features.append(features)
        
        # Output is just future x, y positions
        output_positions = output_data[['x', 'y']].values
        
        return {
            'input': np.array(input_features),  # (seq_len, n_features)
            'output': output_positions,  # (output_seq_len, 2)
            'target_player_id': target_player_id,
            'start_frame': start_frame
        }
    
    def _get_frame_features(self, target_player, other_players, ball):
        """Extract features for one frame."""
        
        features = []
        
        # === TARGET PLAYER FEATURES ===
        features.extend([
            target_player['x'],
            target_player['y'],
            target_player['velocity_x'],
            target_player['velocity_y'],
            target_player['orientation'],
            target_player.get('speed', 0),
            target_player.get('acceleration', 0),
        ])
        
        # === BALL FEATURES ===
        ball_dx = ball['x'] - target_player['x']
        ball_dy = ball['y'] - target_player['y']
        ball_dist = np.sqrt(ball_dx**2 + ball_dy**2)
        ball_angle = np.arctan2(ball_dy, ball_dx)
        
        features.extend([
            ball['x'],
            ball['y'],
            ball_dx,
            ball_dy,
            ball_dist,
            ball_angle
        ])
        
        # === NEARBY PLAYERS CONTEXT ===
        if len(other_players) > 0:
            # Calculate distances to all other players
            distances = []
            for _, other in other_players.iterrows():
                dx = other['x'] - target_player['x']
                dy = other['y'] - target_player['y']
                dist = np.sqrt(dx**2 + dy**2)
                distances.append((other, dist, dx, dy))
            
            # Sort by distance and take nearest N
            distances.sort(key=lambda x: x[1])
            nearest = distances[:self.max_nearby_players]
            
            # Features for each nearby player
            for other, dist, dx, dy in nearest:
                angle = np.arctan2(dy, dx)
                rel_vel_x = other['velocity_x'] - target_player['velocity_x']
                rel_vel_y = other['velocity_y'] - target_player['velocity_y']
                same_team = int(other.get('team', -1) == target_player.get('team', -1))
                
                features.extend([
                    dx,  # relative position
                    dy,
                    dist,
                    angle,
                    rel_vel_x,  # relative velocity
                    rel_vel_y,
                    same_team
                ])
            
            # Pad if fewer than max_nearby_players
            for _ in range(self.max_nearby_players - len(nearest)):
                features.extend([0, 0, 0, 0, 0, 0, 0])  # 7 features per player
            
            # Aggregate features
            features.extend([
                np.mean([p['x'] for _, p in other_players.iterrows()]),  # center of mass
                np.mean([p['y'] for _, p in other_players.iterrows()]),
                len(other_players),  # number of nearby players
            ])
        else:
            # No other players - fill with zeros
            features.extend([0] * (self.max_nearby_players * 7 + 3))
        
        return features
    
    def _fit_scaler(self):
        """Fit scaler on all input features."""
        all_inputs = np.vstack([s['input'] for s in self.samples])
        self.input_scaler = StandardScaler()
        self.input_scaler.fit(all_inputs)
        
        all_outputs = np.vstack([s['output'] for s in self.samples])
        self.output_scaler = StandardScaler()
        self.output_scaler.fit(all_outputs)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Normalize
        input_norm = self.input_scaler.transform(sample['input'])
        output_norm = self.output_scaler.transform(sample['output'])
        
        return {
            'input': torch.FloatTensor(input_norm),
            'output': torch.FloatTensor(output_norm)
        }
    
    def get_feature_dim(self):
        """Return the dimension of input features."""
        return self.samples[0]['input']