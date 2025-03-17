
from utils.resize_right import resize
from union.modules import *


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True
    ):
        super().__init__()
        self.channels = channels

        # Construct a list of feature dimensions from the base 'dim'
        # multiplied by the factors in 'dim_mults'
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # If with_time_emb is True, create a time embedding MLP
        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SineCosinePosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # -----------------------------
        # Down-sampling blocks
        # -----------------------------
        for idx, (dim_in, dim_out_) in enumerate(in_out):
            is_last = idx >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNeXtStage(in_dim=dim_in, out_dim=dim_out_, emb_dim=time_dim, use_norm=(idx != 0)),
                ConvNeXtStage(in_dim=dim_out_, out_dim=dim_out_, emb_dim=time_dim, use_norm=True),
                ResidualAdd(PreNormalization(dim_out_, FastAttention(dim_out_))),
                Down2x(dim_out_) if not is_last else nn.Identity()
            ]))

        # -----------------------------
        # Middle block
        # -----------------------------
        mid_dim = dims[-1]
        self.mid_block1 = ConvNeXtStage(in_dim=mid_dim, out_dim=mid_dim, emb_dim=time_dim, use_norm=True)
        self.mid_attn = ResidualAdd(PreNormalization(mid_dim, FullAttention(mid_dim)))
        self.mid_block2 = ConvNeXtStage(in_dim=mid_dim, out_dim=mid_dim, emb_dim=time_dim, use_norm=True)

        # -----------------------------
        # Up-sampling blocks
        # -----------------------------
        for idx, (dim_in, dim_out_) in enumerate(reversed(in_out[1:])):
            is_last = idx >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ConvNeXtStage(in_dim=(dim_out_ * 2), out_dim=dim_in, emb_dim=time_dim, use_norm=True),
                ConvNeXtStage(in_dim=dim_in, out_dim=dim_in, emb_dim=time_dim, use_norm=True),
                ResidualAdd(PreNormalization(dim_in, FastAttention(dim_in))),
                Up2x(dim_in) if not is_last else nn.Identity()
            ]))

        # By default, the final output dimension = input channels (unless specified)
        out_dim = fallback(out_dim, channels)

        # Final 1��1 convolution after a concluding ConvNeXtStage
        self.final_conv = nn.Sequential(
            ConvNeXtStage(in_dim=dim, out_dim=dim, emb_dim=None, use_norm=True),
            nn.Conv2d(dim, out_dim, kernel_size=1)
        )

    def forward(self, x, time):
        # Compute time embedding if applicable
        t = self.time_mlp(time) if is_not_none(self.time_mlp) else None
        initial_shape = x.shape[-2:]  # keep track of original spatial dimensions
        h = []  # list for skip connections

        # -----------------------------
        # Down-sampling
        # -----------------------------
        for stage1, stage2, attn, down_op in self.downs:
            x = stage1(x, t)
            x = stage2(x, t)
            x = attn(x)
            h.append(x)
            x = down_op(x)

        # -----------------------------
        # Middle
        # -----------------------------
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # -----------------------------
        # Up-sampling
        # -----------------------------
        for stage1, stage2, attn, up_op in self.ups:
            # Pop last skip-connection feature map
            skip_feat = h.pop()
            # Resize current x to match skip feature��s resolution for concatenation
            x_resized = resize(x, out_shape=skip_feat.shape[-2:])
            x = torch.cat((x_resized, skip_feat), dim=1)

            x = stage1(x, t)
            x = stage2(x, t)
            x = attn(x)
            x = up_op(x)

        # Resize once more to original input resolution before final convolution
        x = resize(x, out_shape=initial_shape)
        return self.final_conv(x)
