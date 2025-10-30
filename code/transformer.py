import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Standard positional encoding for temporal information."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class SpatialAttention(nn.Module):
    """
    Spatial attention layer: models interactions between players at each timestep.
    Uses multi-head self-attention across the player dimension.
    """
    
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super(SpatialAttention, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, num_players, d_model)
            mask: (batch, num_players) - True for real players, False for padding
        
        Returns:
            (batch, num_players, d_model)
        """
        # Self-attention across players
        # Convert mask format if provided
        attn_mask = None
        if mask is not None:
            # mask shape: (batch, num_players)
            # attention mask needs shape: (batch * num_heads, num_players, num_players)
            # For simplicity, we'll use key_padding_mask instead
            key_padding_mask = ~mask  # Invert: True means ignore
        else:
            key_padding_mask = None
        
        # Attention
        attn_out, _ = self.attention(
            x, x, x,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class TemporalAttention(nn.Module):
    """
    Temporal attention layer: models evolution of each player over time.
    Uses multi-head self-attention across the temporal dimension.
    """
    
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super(TemporalAttention, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            (batch, seq_len, d_model)
        """
        # Self-attention across time
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class SpatioTemporalEncoderLayer(nn.Module):
    """
    Combined spatio-temporal encoding layer.
    First applies spatial attention, then temporal attention.
    """
    
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super(SpatioTemporalEncoderLayer, self).__init__()
        
        self.spatial_attention = SpatialAttention(d_model, num_heads, dropout)
        self.temporal_attention = TemporalAttention(d_model, num_heads, dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, num_players, d_model)
            mask: (batch, num_players) - mask for padding
        
        Returns:
            (batch, seq_len, num_players, d_model)
        """
        batch_size, seq_len, num_players, d_model = x.shape
        
        # Spatial attention: for each timestep, attend across players
        spatial_out = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch, num_players, d_model)
            x_t = self.spatial_attention(x_t, mask)
            spatial_out.append(x_t)
        
        x = torch.stack(spatial_out, dim=1)  # (batch, seq_len, num_players, d_model)
        
        # Temporal attention: for each player, attend across time
        temporal_out = []
        for p in range(num_players):
            x_p = x[:, :, p, :]  # (batch, seq_len, d_model)
            x_p = self.temporal_attention(x_p)
            temporal_out.append(x_p)
        
        x = torch.stack(temporal_out, dim=2)  # (batch, seq_len, num_players, d_model)
        
        return x


class SpatioTemporalTransformer(nn.Module):
    """
    Full Spatio-Temporal Transformer for trajectory prediction.
    
    Architecture:
    1. Input embedding
    2. Positional encoding
    3. Multiple spatio-temporal encoder layers
    4. Decoder to predict future trajectories
    """
    
    def __init__(self, 
                 input_dim,
                 d_model=128,
                 num_heads=4,
                 num_layers=3,
                 dropout=0.1,
                 max_players=11,
                 output_seq_length=10):
        """
        Args:
            input_dim: Dimension of input features per player per timestep
            d_model: Dimension of model (embedding size)
            num_heads: Number of attention heads
            num_layers: Number of spatio-temporal encoder layers
            dropout: Dropout probability
            max_players: Maximum number of players (for padding)
            output_seq_length: Number of future timesteps to predict
        """
        super(SpatioTemporalTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_players = max_players
        self.output_seq_length = output_seq_length
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for temporal dimension
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Spatio-temporal encoder layers
        self.encoder_layers = nn.ModuleList([
            SpatioTemporalEncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Decoder: predict future trajectories
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_seq_length * 2)  # x, y for each future frame
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input features (batch, seq_len, num_players, input_dim)
            mask: Player mask (batch, num_players) - True for real players
        
        Returns:
            predictions: (batch, num_players, output_seq_length, 2)
        """
        batch_size, seq_len, num_players, input_dim = x.shape
        
        # Embed inputs
        # Reshape to apply embedding: (batch * seq_len * num_players, input_dim)
        x_flat = x.reshape(-1, input_dim)
        x_embed = self.input_embedding(x_flat)
        x_embed = x_embed.reshape(batch_size, seq_len, num_players, self.d_model)
        
        # Apply positional encoding to temporal dimension
        # For each player, add positional encoding across time
        x_pos = []
        for p in range(num_players):
            x_p = x_embed[:, :, p, :]  # (batch, seq_len, d_model)
            x_p = self.pos_encoder(x_p)
            x_pos.append(x_p)
        x_embed = torch.stack(x_pos, dim=2)  # (batch, seq_len, num_players, d_model)
        
        # Apply spatio-temporal encoder layers
        encoded = x_embed
        for layer in self.encoder_layers:
            encoded = layer(encoded, mask)
        
        # Use final timestep for prediction
        final_state = encoded[:, -1, :, :]  # (batch, num_players, d_model)
        
        # Decode to future trajectories
        predictions = self.decoder(final_state)  # (batch, num_players, output_seq_length * 2)
        predictions = predictions.reshape(batch_size, num_players, self.output_seq_length, 2)
        
        return predictions


class SpatioTemporalDataset(torch.utils.data.Dataset):
    """
    Dataset for spatio-temporal transformer.
    Returns all players at once (not individually).
    Handles partial ground truth - only some players have target trajectories.
    """
    
    def __init__(self, plays_data, input_seq_length=20, output_seq_length=10, 
                 max_players=11, pad_output=True):
        """
        Args:
            pad_output: If True, pad output sequences to output_seq_length.
                       If False, skip samples that don't have exact output_seq_length frames.
        """
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.max_players = max_players
        self.pad_output = pad_output
        
        self.samples = []
        self._create_samples(plays_data)
    
    def _create_samples(self, plays_data):
        """Create samples where each sample contains all players."""
        
        for play in plays_data:
            players_df = play['players']
            ball_df = play['ball']
            
            max_frame = players_df['frame'].max()
            total_seq_length = self.input_seq_length + self.output_seq_length
            
            # Sliding window through play
            for start_frame in range(0, max_frame - total_seq_length + 1, 5):
                end_input_frame = start_frame + self.input_seq_length
                end_output_frame = start_frame + total_seq_length
                
                sample = self._create_play_sample(
                    players_df, ball_df, start_frame, end_input_frame, end_output_frame
                )
                
                if sample is not None:
                    self.samples.append(sample)
        
        print(f"Created {len(self.samples)} training samples")
    
    def _create_play_sample(self, players_df, ball_df, start_frame, end_input_frame, 
                           end_output_frame, actual_output_length):
        """
        Create one sample with all players.
        Only players marked as 'needs_prediction' will have ground truth in output.
        Handles variable output lengths with padding/masking.
        """
        
        player_ids = players_df['player_id'].unique()
        
        # Input features for all players
        input_features = np.zeros((self.input_seq_length, self.max_players, 7))
        output_positions = np.zeros((self.output_seq_length, self.max_players, 2))
        input_mask = np.zeros(self.max_players, dtype=bool)
        output_mask = np.zeros(self.max_players, dtype=bool)
        temporal_mask = np.zeros(self.output_seq_length, dtype=bool)  # NEW: mask for valid timesteps
        
        # Mark which output timesteps are valid (not padding)
        temporal_mask[:actual_output_length] = True
        
        for p_idx, player_id in enumerate(player_ids):
            if p_idx >= self.max_players:
                break
            
            player_data = players_df[
                (players_df['player_id'] == player_id) &
                (players_df['frame'] >= start_frame) &
                (players_df['frame'] < end_output_frame)
            ].sort_values('frame')
            
            # Need complete input sequence
            if len(player_data[player_data['frame'] < end_input_frame]) != self.input_seq_length:
                continue
            
            # Input sequence - all players used as context
            input_data = player_data[player_data['frame'] < end_input_frame]
            for t, (_, row) in enumerate(input_data.iterrows()):
                input_features[t, p_idx, :] = [
                    row['x'], row['y'],
                    row['velocity_x'], row['velocity_y'],
                    row['orientation'],
                    row.get('speed', 0),
                    row.get('acceleration', 0)
                ]
            input_mask[p_idx] = True
            
            # Output sequence - only if player needs prediction
            needs_pred = player_data.iloc[0].get('needs_prediction', False)
            
            if needs_pred:
                output_data = player_data[player_data['frame'] >= end_input_frame]
                for t, (_, row) in enumerate(output_data.iterrows()):
                    if t < self.output_seq_length:  # Don't exceed max output length
                        output_positions[t, p_idx, :] = [row['x'], row['y']]
                
                # Pad remaining timesteps with last position if needed
                if actual_output_length < self.output_seq_length and len(output_data) > 0:
                    last_pos = output_positions[actual_output_length - 1, p_idx, :]
                    for t in range(actual_output_length, self.output_seq_length):
                        output_positions[t, p_idx, :] = last_pos
                
                output_mask[p_idx] = True
        
        if not input_mask.any():
            return None
        
        return {
            'input': input_features,
            'output': output_positions,
            'input_mask': input_mask,
            'output_mask': output_mask,
            'temporal_mask': temporal_mask  # NEW: which output timesteps are valid
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input': torch.FloatTensor(sample['input']),
            'output': torch.FloatTensor(sample['output']),
            'input_mask': torch.BoolTensor(sample['input_mask']),
            'output_mask': torch.BoolTensor(sample['output_mask']),
            'temporal_mask': torch.BoolTensor(sample['temporal_mask'])
        }


def masked_trajectory_loss(predictions, targets, output_mask, temporal_mask, criterion=nn.MSELoss(reduction='none')):
    """
    Compute loss only for players that need prediction and valid timesteps.
    
    Args:
        predictions: [batch, output_steps, num_players, 2]
        targets: [batch, output_steps, num_players, 2]
        output_mask: [batch, num_players] - True for players to predict
        temporal_mask: [batch, output_steps] - True for valid timesteps (not padding)
        criterion: Loss function
    
    Returns:
        loss: Scalar loss value
    """
    batch_size, output_steps, num_players, _ = predictions.shape
    
    # Compute loss for all predictions
    # Shape: [batch, output_steps, num_players, 2]
    losses = criterion(predictions, targets)
    
    # Average over (x, y) coordinates
    # Shape: [batch, output_steps, num_players]
    losses = losses.mean(dim=3)
    
    # Create combined mask: [batch, output_steps, num_players]
    # True only if BOTH player needs prediction AND timestep is valid
    output_mask_expanded = output_mask.unsqueeze(1)  # [batch, 1, num_players]
    temporal_mask_expanded = temporal_mask.unsqueeze(2)  # [batch, output_steps, 1]
    combined_mask = output_mask_expanded & temporal_mask_expanded  # [batch, output_steps, num_players]
    
    # Apply mask
    masked_losses = losses * combined_mask.float()
    
    # Count how many predictions we're actually computing loss for
    num_valid = combined_mask.sum(dim=(1, 2), keepdim=False).float()  # [batch]
    num_valid = torch.clamp(num_valid, min=1.0)  # Avoid division by zero
    
    # Average loss per sample
    avg_loss_per_sample = masked_losses.sum(dim=(1, 2)) / num_valid
    
    # Average over batch
    total_loss = avg_loss_per_sample.mean()
    
    return total_loss


def train_spatiotemporal_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.0001):
    """
    Training loop for spatio-temporal transformer with masked loss.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            input_mask = batch['input_mask'].to(device)
            output_mask = batch['output_mask'].to(device)
            temporal_mask = batch['temporal_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(inputs, input_mask)
            
            # Compute loss only for players that need prediction AND valid timesteps
            loss = masked_trajectory_loss(predictions, targets, output_mask, temporal_mask)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['output'].to(device)
                input_mask = batch['input_mask'].to(device)
                output_mask = batch['output_mask'].to(device)
                temporal_mask = batch['temporal_mask'].to(device)
                
                predictions = model(inputs, input_mask)
                loss = masked_trajectory_loss(predictions, targets, output_mask, temporal_mask)
                
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_spatiotemporal_model.pth')
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_spatiotemporal_model.pth'))
    return model


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    import pandas as pd
    
    # Create dummy data
    plays_data = []
    for play_idx in range(50):
        n_frames = 50
        n_players = 6
        
        players_list = []
        for player_id in range(n_players):
            # Only predict half the players (e.g., receivers only)
            needs_pred = player_id < 3  # Only first 3 players need prediction
            
            for frame in range(n_frames):
                t = frame * 0.1
                players_list.append({
                    'frame': frame,
                    'player_id': player_id,
                    'x': 10 + player_id * 5 + 3 * t + np.random.normal(0, 0.2),
                    'y': 20 + 2 * t + player_id * 2 + np.random.normal(0, 0.2),
                    'velocity_x': 3 + np.random.normal(0, 0.5),
                    'velocity_y': 2 + np.random.normal(0, 0.5),
                    'orientation': np.pi/4,
                    'speed': 3.6,
                    'acceleration': 0.1,
                    'needs_prediction': needs_pred  # Flag: do we need to predict this player?
                })
        
        ball_list = []
        for frame in range(n_frames):
            ball_list.append({
                'frame': frame,
                'x': 50,
                'y': 50
            })
        
        plays_data.append({
            'players': pd.DataFrame(players_list),
            'ball': pd.DataFrame(ball_list)
        })
    
    # Create dataset
    dataset = SpatioTemporalDataset(
        plays_data,
        input_seq_length=20,
        output_seq_length=10,
        max_players=11
    )
    
    # Create model
    model = SpatioTemporalTransformer(
        input_dim=7,  # x, y, vel_x, vel_y, orientation, speed, accel
        d_model=64,   # Smaller for limited data
        num_heads=4,
        num_layers=2,  # Start with just 2 layers
        dropout=0.3,
        max_players=11,
        output_seq_length=10
    )
    
    print(f"Model created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Input: {sample['input'].shape} = [steps, players, features]")
    print(f"  Output: {sample['output'].shape} = [steps, players, 2]")
    print(f"  Input mask: {sample['input_mask'].shape} = [players] - {sample['input_mask'].sum()} players exist")
    print(f"  Output mask: {sample['output_mask'].shape} = [players] - {sample['output_mask'].sum()} players need prediction")
    
    with torch.no_grad():
        pred = model(
            sample['input'].unsqueeze(0),  # Add batch dimension: [1, steps, players, features]
            sample['input_mask'].unsqueeze(0)
        )
        print(f"\nPrediction shape: {pred.shape}")
        print(f"  Expected: (1, {dataset.output_seq_length}, {dataset.max_players}, 2)")
        print(f"  Interpretation: [batch=1, output_steps={dataset.output_seq_length}, players={dataset.max_players}, (x,y)=2]")
    
    print("\n✓ Spatio-Temporal Transformer ready to train!")
    print("\nKey points:")
    print("  • Input has 20 steps (frames), output has 10 steps")
    print("  • All players used as context in input")
    print("  • Loss only computed for players with 'needs_prediction' flag")
    print("  • Model still predicts all players (but we ignore some in loss)")