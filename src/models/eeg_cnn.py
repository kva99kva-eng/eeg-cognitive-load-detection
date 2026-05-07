import torch
import torch.nn as nn


class EEGSimpleCNN(nn.Module):
    """
    Simple CNN for EEG cognitive load classification.

    Input shape:
        x: (batch_size, n_channels=14, n_times=256)

    Output:
        logits: (batch_size,)
    """

    def __init__(self, n_channels: int = 14, n_times: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # Input after unsqueeze: (B, 1, 14, 256)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 15),
                padding=(0, 7),
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Spatial convolution across EEG channels.
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(n_channels, 1),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AvgPool2d(kernel_size=(1, 4)),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 7),
                padding=(0, 3),
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        x = x.unsqueeze(1)  # (B, 1, 14, 256)
        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)
