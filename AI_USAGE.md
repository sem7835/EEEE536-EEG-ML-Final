Below is fully listed every use of AI throughout this process:

Tool: Google "AI Mode" (Gemini)
Prompt: "using pytorch, make a simple neural net that can discriminate between two normal distributions"
Response: 
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Data Generation
n_samples = 1000
# Distribution A: mean -2, std 1
x0 = torch.randn(n_samples, 1) - 2.0
y0 = torch.zeros(n_samples, 1)

# Distribution B: mean 2, std 1
x1 = torch.randn(n_samples, 1) + 2.0
y1 = torch.ones(n_samples, 1)

x = torch.cat([x0, x1], dim=0)
y = torch.cat([y0, y1], dim=0)

# 2. Simple Neural Network Architecture
model = nn.Sequential(
    nn.Linear(1, 10),    # Input layer
    nn.ReLU(),           # Activation
    nn.Linear(10, 1),    # Hidden to Output
    nn.Sigmoid()         # Probability output
)

# 3. Training Setup
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training Loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# 5. Test with new values
test_val = torch.tensor([[2.5], [-2.5]])
with torch.no_grad():
    preds = model(test_val)
    print(f"Prediction for 2.5: {preds[0].item():.4f} (Class 1)")


This request was made to help figure out the specifics of implementing a more complex neural network, as I was having extreme hold ups with regards to the specifics
I modified this code to have more than one input channel

Prompt: "I need code that can take in two different individuals EEG readings - the EEG signals have 22 channels each, and vary over time with matched timestamps - the code needs to be able to learn the differences between the two individuals and output a prediction of which of the two a new signal belongs to"

import torch
import torch.nn as nn
import torch.optim as optim

# 1. Setup Hyperparameters
n_channels = 22
time_steps = 500  # Example: 2 seconds of data at 250Hz
n_samples = 1000

# 2. Synthetic EEG Data Generation
# Input shape: [Batch, Channels, Time]
x_person_a = torch.randn(n_samples, n_channels, time_steps) + 0.1 # Individual A
x_person_b = torch.randn(n_samples, n_channels, time_steps) - 0.1 # Individual B

x = torch.cat([x_person_a, x_person_b], dim=0)
y = torch.cat([torch.zeros(n_samples, 1), torch.ones(n_samples, 1)], dim=0)

# 3. Define Conv1D Architecture
class EEGNet(nn.Module):
    def __init__(self, n_channels, time_steps):
        super().__init__()
        self.features = nn.Sequential(
            # Temporal convolution: slides along the time axis
            nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4), # Downsample time dimension
            
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Collapse time to a single value per feature map
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# 4. Initialize and Train
model = EEGNet(n_channels, time_steps)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Simple training loop
for epoch in range(20):
    optimizer.zero_grad()
    preds = model(x)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. Prediction Function
def identify_person(signal):
    # signal shape must be [1, 22, time_steps]
    model.eval()
    with torch.no_grad():
        prob = model(signal).item()
        return "Individual B" if prob > 0.5 else "Individual A"

This was asked as I was at a point where my own implementation was failing universally, and I had decided to start over
I did not use the training or prediction methods provided here, but i did adapt the model itself for use in my project
This specific architecture suited my needs adequately and allowed for me to be able to program even when I lacked the specific knowledge to do so

After this, I changed tools to use Antrhopic's "Claude" (Sonnet 4.6):
Prompt: (Provided model code for a prototype gesture identifier) "I am using the above "GestNet" code to create a neural network using pytorch for the purposes of motor imagery (MI) classification from EEG signal.

This model should take in a number of samples corresponding to MI gestures, learn from them, and then be able to tell, during evaluation, which gesture a random sample best corresponds to"
✅ Improved Version
pythonimport torch
import torch.nn as nn

class GestNet(nn.Module):
    def __init__(self, n_channels=22, time_steps=1000, n_gest=5, lstm_hidden=128, lstm_layers=2, dropout=0.5):
        super().__init__()

        # --- Spatial filter (across EEG channels, like a depthwise op) ---
        self.input_norm = nn.BatchNorm1d(n_channels)  # Normalize raw EEG per channel

        # --- Temporal CNN: progressive pooling instead of one aggressive pool ---
        self.cnn = nn.Sequential(
            # Block 1: broad temporal patterns
            nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2),          # 1000 → 500
            nn.Dropout(0.25),

            # Block 2: finer patterns
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),          # 500 → 250
            nn.Dropout(0.25),

            # Block 3: compact representation
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),          # 250 → 125
        )

        # --- LSTM: captures temporal dynamics in the CNN feature sequence ---
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,  # inter-layer dropout
            bidirectional=False,  # set True for offline/non-causal analysis
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)  # stabilizes LSTM output

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_gest),
        )

    def forward(self, x):
        # x: (batch, n_channels, time_steps)
        x = self.input_norm(x)          # normalize EEG channels
        x = self.cnn(x)                 # (batch, 128, T//8)
        x = x.permute(0, 2, 1)         # (batch, T//8, 128) for LSTM
        lstm_out, _ = self.lstm(x)
        x = self.lstm_norm(lstm_out[:, -1, :])  # take last timestep + normalize
        return self.classifier(x)

I prompted Claude for assistance in creating the gesture net identifier program, as it was repeatedly failing when I tried my own implementations
I co opted the code from this output, with very few changes, and attempted to run it
Ultimately, this line of questioning resulted in failure, as I was unable to turn this model into a functional classifier
Perhaps it was an issue with my data? I am still unclear as to why, and will eventually do my own personal investigation to establish a reason, though for the sake of this project, I simply ran out of time to do so :/

LEARNING AND REFLECTION:
I am in a very conflicted spot with regards to the large language models used for assistance:
On the one hand, they provided me with the ability to program when I otherwise would not have been able to, and solved a number of issues that I would have been stuck on
Overall, the use of these AI tools allowed me to complete much more of this project than I think I would have been able to otherwise, with regards to programming
I have never and WILL never use AI to write for me, though, I only wish to use it to make up for my lack of programming knowledge
That said, I am a very prideful person, and am almost willing to believe that if I could not come up with these solutions on my own, then I did not "deserve" to complete the project in this way
Essentially, since the code is in many ways not my own, I'm unsure whether I wish to take credit for the work done here or not
I digress
And all of this doesn't even address the other issues present with large AIs including pertinent things such as corporatism, environmentalism, or the potentially negative impact on job security in my future

