import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Architecture ablation modes used in Group E experiments.
# "standard" is the default mode used in all confirmatory Group A runs.
# W_addon is instantiated in every mode so the parameter budget is always
# 3,951,951; it is active only in arch_mode="additive" (E3) and remains at
# its zero initialisation in all other modes.
# ---------------------------------------------------------------------------
VALID_ARCH_MODES = frozenset({
    "standard",    # full model (default)
    "meta_only",   # E1: ECG backbone zeroed before fusion (metadata-only path)
    "concat",      # E2: gated fusion replaced by plain concatenation
    "additive",    # E3: gated fusion replaced by additive injection via W_addon
    "no_glu",      # E4: sigmoid gate replaced by ReLU gate
    "no_residual", # E5: ts_meta_residual injection removed
    "no_q_meta",   # E6: meta_quality forced to 1.0 (quality-gating disabled)
    "no_dropout",  # E7: sample meta-dropout disabled at training time
})


class TemporalStatPool(nn.Module):
    """Advanced statistical pooling for temporal features."""

    def forward(self, x):
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        maxv = x.max(dim=2)[0]
        minv = x.min(dim=2)[0]
        return torch.cat([mean, std, maxv, minv], dim=1)


class ResBlock1D(nn.Module):
    """Residual 1D block for ECG feature extraction."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class TSBackbone1D_v5(nn.Module):
    """Residual 1D backbone for 12-lead ECG."""

    def __init__(self, in_ch=12, base=64):
        super().__init__()
        self.conv_init = nn.Conv1d(in_ch, base, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_init = nn.BatchNorm1d(base)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base, base, num_blocks=2)
        self.layer2 = self._make_layer(base, base * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base * 2, base * 4, num_blocks=2, stride=2)

        self.pool = TemporalStatPool()
        self.out_dim = base * 4 * 4

    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

        layers = [ResBlock1D(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return x


class MetaMLP_v5(nn.Module):
    """MLP for structured metadata."""

    def __init__(self, in_dim, hid=128, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.BatchNorm1d(hid),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.BatchNorm1d(hid),
            nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, m):
        return self.net(m)


class EZNX_ATLAS_A_v5(nn.Module):
    """
    Multimodal ECG + metadata model.

    This version stays close to the strongest ECG baseline and uses metadata
    through the original gated fusion path. A small auxiliary metadata head is
    kept only to encourage the metadata branch to remain informative.

    Parameters
    ----------
    arch_mode : str
        Architecture ablation mode.  Must be one of ``VALID_ARCH_MODES``.
        Default is ``"standard"`` (full model used in all Group A runs).
        Group E ablations pass a non-default value.
    """

    def __init__(
        self,
        ts_in_ch=12,
        meta_dim=16,
        n_classes=5,
        ts_base=64,
        meta_hid=128,
        meta_out=128,
        meta_dropout_p=0.1,
        arch_mode: str = "standard",
    ):
        super().__init__()

        if arch_mode not in VALID_ARCH_MODES:
            raise ValueError(
                f"arch_mode={arch_mode!r} is not valid; "
                f"choose from {sorted(VALID_ARCH_MODES)}."
            )
        self.arch_mode = arch_mode

        if meta_dim % 2 != 0 or meta_dim < 4:
            raise ValueError("meta_dim must be an even integer for values + masks.")

        self.meta_dim = int(meta_dim)
        self.meta_value_dim = self.meta_dim // 2
        self.demo_value_dim = 2
        if self.meta_value_dim <= self.demo_value_dim:
            raise ValueError("meta_dim leaves no room for anthropometric features.")

        self.anthro_value_dim = self.meta_value_dim - self.demo_value_dim
        self.meta_hid = int(meta_hid)

        self.ts = TSBackbone1D_v5(ts_in_ch, base=ts_base)
        self.demo_encoder = MetaMLP_v5(self.demo_value_dim * 2, hid=64, out_dim=64, dropout=0.10)
        self.anthro_encoder = MetaMLP_v5(
            self.anthro_value_dim * 2,
            hid=96,
            out_dim=64,
            dropout=0.10,
        )
        self.meta_fuse = nn.Sequential(
            nn.Linear(130, self.meta_hid),
            nn.ReLU(),
            nn.BatchNorm1d(self.meta_hid),
            nn.Dropout(0.10),
            nn.Linear(self.meta_hid, meta_out),
            nn.ReLU(),
        )
        self.meta_dropout_p = float(meta_dropout_p)

        fuse_dim = self.ts.out_dim + meta_out
        self.gate = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, fuse_dim),
            nn.Sigmoid(),
        )

        self.head_ecg = nn.Linear(self.ts.out_dim, n_classes)
        self.head_meta = nn.Linear(meta_out, n_classes)
        self.head_fused = nn.Linear(fuse_dim, n_classes)

        # W_res: primary zero-initialised residual projection (always active in standard mode).
        self.ts_meta_residual = nn.Linear(meta_out, self.ts.out_dim)
        nn.init.zeros_(self.ts_meta_residual.weight)
        nn.init.zeros_(self.ts_meta_residual.bias)

        # W_addon: second zero-initialised residual projection (ℝ^{out_dim × meta_out}).
        # Instantiated in every arch_mode so the parameter count is always 3,951,951.
        # Active only when arch_mode="additive" (Group E, E3); receives zero gradient
        # and remains at zero throughout training in all other modes.
        self.W_addon = nn.Linear(meta_out, self.ts.out_dim)
        nn.init.zeros_(self.W_addon.weight)
        nn.init.zeros_(self.W_addon.bias)

    def forward(self, x_ts, x_meta, meta_present_mask=None):
        h_ts = self.ts(x_ts)
        logits_ecg = self.head_ecg(h_ts)

        use_meta = (x_meta is not None) and (x_meta.numel() > 0)
        meta_used_flag = False

        if use_meta:
            if meta_present_mask is None:
                meta_present_mask = torch.ones_like(x_meta)
            if x_meta.size(1) != self.meta_value_dim:
                raise ValueError(
                    f"Expected x_meta with {self.meta_value_dim} features, got {x_meta.size(1)}."
                )
            if meta_present_mask.size(1) != self.meta_value_dim:
                raise ValueError(
                    "meta_present_mask must match x_meta along the feature dimension."
                )

            demo_values = x_meta[:, : self.demo_value_dim]
            demo_mask = meta_present_mask[:, : self.demo_value_dim]
            anthro_values = x_meta[:, self.demo_value_dim :]
            anthro_mask = meta_present_mask[:, self.demo_value_dim :]

            h_demo = self.demo_encoder(torch.cat([demo_values, demo_mask], dim=1))
            h_anthro = self.anthro_encoder(torch.cat([anthro_values, anthro_mask], dim=1))

            demo_quality = demo_mask.float().mean(dim=1, keepdim=True)
            anthro_quality = anthro_mask.float().mean(dim=1, keepdim=True)
            meta_quality = torch.clamp(demo_quality + 0.5 * anthro_quality, max=1.0)

            h_anthro = h_anthro * anthro_quality
            h_m = self.meta_fuse(
                torch.cat([h_demo, h_anthro, demo_quality, anthro_quality], dim=1)
            )

            # E6: disable quality-gating by forcing meta_quality to 1.
            # Applied after meta_fuse so the quality scalars are still visible
            # as contextual inputs to the metadata MLP.
            if self.arch_mode == "no_q_meta":
                meta_quality = torch.ones_like(meta_quality)

            # E7: skip sample meta-dropout.
            if self.training and self.meta_dropout_p > 0 and self.arch_mode != "no_dropout":
                keep_mask = (
                    torch.rand(h_m.size(0), 1, device=h_m.device, dtype=h_m.dtype)
                    >= self.meta_dropout_p
                ).float()
                h_m = h_m * keep_mask

            h_m = h_m * meta_quality
            meta_used_flag = bool((h_m.abs().sum() > 0).item())
        else:
            h_m = h_ts.new_zeros((h_ts.size(0), self.head_meta.in_features))
            meta_quality = h_ts.new_zeros((h_ts.size(0), 1))

        logits_meta = self.head_meta(h_m)

        # E5: skip residual injection.
        if self.arch_mode != "no_residual":
            h_ts = h_ts + 0.10 * self.ts_meta_residual(h_m) * meta_quality

        # E1: zero the ECG representation before fusion so only metadata drives
        # the fused head (ECG auxiliary head still uses the full h_ts above).
        h_ts_fuse = torch.zeros_like(h_ts) if self.arch_mode == "meta_only" else h_ts

        # Gated fusion — behaviour depends on arch_mode.
        if self.arch_mode == "additive":
            # E3: additive injection via W_addon replaces the GLU gate.
            h_ts_fuse = h_ts_fuse + self.W_addon(h_m) * meta_quality
            h = torch.cat([h_ts_fuse, h_m], dim=1)
            z = h  # no gate multiplication in additive mode

        elif self.arch_mode == "concat":
            # E2: plain concatenation — no gate.
            h = torch.cat([h_ts_fuse, h_m], dim=1)
            z = h

        elif self.arch_mode == "no_glu":
            # E4: replace sigmoid gate with ReLU gate.
            h = torch.cat([h_ts_fuse, h_m], dim=1)
            g = torch.relu(self.gate[2](torch.relu(self.gate[0](h))))
            z = h * g

        else:
            # standard / meta_only / no_residual / no_q_meta / no_dropout:
            # standard GLU-style gated fusion.
            h = torch.cat([h_ts_fuse, h_m], dim=1)
            z = h * self.gate(h)

        logits_fused = self.head_fused(z) + 0.05 * meta_quality * logits_meta

        return {
            "logits_fused": logits_fused,
            "logits_ecg": logits_ecg,
            "logits_meta": logits_meta,
            "meta_used": torch.tensor(meta_used_flag, device=h_ts.device),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2
