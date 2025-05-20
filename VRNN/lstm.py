import torch
import torch.nn as nn

"""LSTM modules."""

class LSTMLayer(nn.Module):
    
    def __init__(self, input_size, hidden_size, n_lstm_layers, use_orthogonal):
        super(LSTMLayer, self).__init__()
        self.n_lstm_layers = n_lstm_layers
        self._use_orthogonal = use_orthogonal

        # lstm. note that batch_first=False (default)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=n_lstm_layers)
        # layer norm
        self.norm = nn.LayerNorm(hidden_size)

        # initialisation
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        
    def forward(self, x, hxs, cxs, masks):
        """
        Process one timestep input with correct dimension handling
        Args:
            x: Input tensor (batch_size, input_size)
            hxs: Hidden state (n_layers, batch_size, hidden_size)
            cxs: Cell state (n_layers, batch_size, hidden_size)
            masks: Binary mask for active sequences (batch_size,)
        """
        batch_size = x.size(0)
    
        # Reshape mask for broadcasting (n_layers, batch_size, 1)
        mask_expanded = masks.view(1, -1, 1).expand(self.n_lstm_layers, batch_size, 1)
    
        # Apply mask to states (zeros out states for completed sequences)
        hxs_masked = hxs * mask_expanded
        cxs_masked = cxs * mask_expanded
    
        # Add sequence dimension for LSTM input
        x_sequence = x.unsqueeze(0)
    
        # LSTM forward pass
        output, (hxs_new, cxs_new) = self.lstm(x_sequence, (hxs_masked, cxs_masked))
    
        # Remove sequence dimension
        output = output.squeeze(0)
    
        # Apply layer normalization
        output = self.norm(output)
    
        return output, (hxs_new, cxs_new)