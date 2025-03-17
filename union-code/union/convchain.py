import math
import torch
from torch import nn

from union.modules import ConvNeXtStage, SineCosinePosEmb


class ConvChain(nn.Module):
    """
    A backbone model consisting of ConvNext blocks arranged in a U-Net-like fashion.
    Depending on 'frame_conditioned', it may also incorporate a positional embedding
    for the frame difference. The skip connections link the first half of the blocks
    to the second half, concatenating feature maps similarly to a U-Net.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        depth=16,
        filters_per_layer=64,
        frame_conditioned=False
    ):
        """
        Args:
            in_channels (int):    Number of channels in the input image.
            out_channels (int):   Number of channels in the output image.
            depth (int):          Total number of ConvNext blocks in the network.
            filters_per_layer (int or list[int]):
                                  Either a fixed number of channels for each block
                                  or a list specifying channel counts per block.
            frame_conditioned (bool):
                                  If True, an additional embedding is computed from
                                  'frame_diff' and merged with the time embedding.
        """
        super().__init__()

        # Determine the per-layer channel dimensions (dims)
        if isinstance(filters_per_layer, (list, tuple)):
            ch_dims = filters_per_layer
        else:
            ch_dims = [filters_per_layer] * depth

        # The dimensionality for the time (and possibly frame) embeddings
        self.depth = depth
        time_dim = ch_dims[0]
        embed_dim = time_dim * 2 if frame_conditioned else time_dim

        # Create a module list to store the ConvNext blocks
        self.blocks = nn.ModuleList()

        # First block: from in_channels to ch_dims[0], no normalization
        self.blocks.append(
            ConvNeXtStage(in_channels, ch_dims[0], emb_dim=embed_dim, norm=False)
        )

        # Middle blocks (down path)
        half_depth = math.ceil(self.depth / 2)
        for idx in range(1, half_depth):
            self.blocks.append(
                ConvNeXtStage(ch_dims[idx - 1], ch_dims[idx], emb_dim=embed_dim, norm=True)
            )

        # Middle-up blocks (up path) �� each block concatenates skip features
        for idx in range(half_depth, depth):
            self.blocks.append(
                ConvNeXtStage(ch_dims[idx - 1] * 2, ch_dims[idx], emb_dim=embed_dim, norm=True)
            )

        # Final 1��1 convolution to map to the desired output channels
        self.final_conv = nn.Conv2d(ch_dims[-1], out_channels, kernel_size=1)

        # Time embedding (positional) encoder
        self.time_encoder = nn.Sequential(
            SineCosinePosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Optional frame embedding if 'frame_conditioned' is True
        if frame_conditioned:
            self.frame_encoder = nn.Sequential(
                SineCosinePosEmb(time_dim),
                nn.Linear(time_dim, time_dim * 4),
                nn.GELU(),
                nn.Linear(time_dim * 4, time_dim)
            )

        self.frame_conditioned = frame_conditioned

    def forward(self, x, t, frame_diff=None):
        """
        Forward pass through the ConvChain.

        Args:
            x (tensor):          Input image of shape (N, in_channels, H, W).
            t (tensor):          1D tensor representing timesteps for positional embedding.
            frame_diff (tensor): Optional 1D tensor representing frame differences
                                 for additional embedding if frame_conditioned=True.

        Returns:
            A tensor of shape (N, out_channels, H, W), after applying all ConvNext blocks
            and the final 1��1 convolution.
        """
        # Encode time
        time_emb = self.time_encoder(t)

        # If frame_diff is provided and we're frame_conditioned, concatenate embeddings
        if self.frame_conditioned and frame_diff is not None:
            frame_emb = self.frame_encoder(frame_diff)
            combined_emb = torch.cat([time_emb, frame_emb], dim=1)
        else:
            combined_emb = time_emb

        # Split blocks into ��down�� and ��up�� segments
        half_depth = math.ceil(self.depth / 2)
        skip_connections = []

        # Down path: apply blocks in sequence, store intermediate outputs
        for idx in range(half_depth):
            x = self.blocks[idx](x, combined_emb)
            skip_connections.append(x)

        # Up path: each block concatenates the skip connection before forward
        for idx in range(half_depth, self.depth):
            skipped = skip_connections.pop()  # get last skip
            x = torch.cat((x, skipped), dim=1)
            x = self.blocks[idx](x, combined_emb)

        # Final 1x1 convolution to get desired output channels
        return self.final_conv(x)
