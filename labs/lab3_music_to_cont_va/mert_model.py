import torch
import torch.nn as nn
from transformers import AutoModel

class MERTContinuousModel(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=2, unfreeze_last_layer=True):
        super().__init__()
        # Load MERT
        self.mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        
        # Freeze MERT by default, then optionally fine-tune only the last encoder layer.
        for param in self.mert.parameters():
            param.requires_grad = False

        if unfreeze_last_layer:
            for param in self.mert.encoder.layers[-1].parameters():
                param.requires_grad = True

        # The rest is the same...
        self.adapter = nn.Linear(768 * 2, hidden_dim) 
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3 # --- OPTIMIZATION 2: ADD DROPOUT ---
        )
        self.regressor = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_values, debug=False):
        if debug:
            print(f"{'='*10} FORWARD PASS SHAPES {'='*10}")
            print(f"1. input_values:      {input_values.shape}")
            
        outputs = self.mert(input_values, output_hidden_states=True)
        
        layer_5 = outputs.hidden_states[5] 
        layer_6 = outputs.hidden_states[6]
        
        if debug:
            print(f"2. layer_5:           {layer_5.shape}")
            print(f"3. layer_6:           {layer_6.shape}")
            
        combined_features = torch.cat([layer_5, layer_6], dim=-1)
        
        if debug:
            print(f"4. combined_features: {combined_features.shape}")
            
        x = torch.relu(self.adapter(combined_features))
        
        if debug:
            print(f"5. x (adapter out):   {x.shape}")
            
        lstm_out, _ = self.lstm(x)
        
        if debug:
            print(f"6. lstm_out:          {lstm_out.shape}")
            
        predictions = self.regressor(lstm_out)
        
        if debug:
            print(f"7. predictions:       {predictions.shape}")
            print(f"{'='*39}")
            
        return predictions

if __name__ == "__main__":
    print("Running dummy test...")
    print("Loading MERT model (this might take a moment)...")
    model = MERTContinuousModel()
    
    # Create dummy audio data (Batch Size: 2, Audio Length: 1 second at 16kHz)
    print("Creating dummy audio tensor...")
    dummy_audio = torch.randn(2, 16000) 
    
    print("Running forward pass with debug=True...")
    predictions = model(dummy_audio, debug=True)