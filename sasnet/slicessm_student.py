"""SASNet (SliceSSM student): in-plane CNN with a physically stepped bidirectional z-SSM and misalignment-aware slice fusion.

Short-axis cine stacks are anisotropic (5-10 mm between slice centres against about 1.4 mm in plane, varying per patient)
and adjacent slices are misaligned because each slice is acquired in its own breath-hold.  The network is a U-Net whose
convolutions are all in-plane (kernels ``1x3x3``, stride ``1x2x2``; one ``3x1x1`` mix at stage 3), so z is never
down-sampled.  Through-plane context comes from two modules:

* :class:`PhysZSSM`: a selective state-space scan along z, run in both directions for every in-plane cell.  The
  discretisation step is ``softplus(dt(x)) * (s_z / s_ref)``: the learned step is scaled by the patient's physical slice
  spacing ``s_z`` (mm, ``geometry["spacing_zyx"][0]``).  ``spacing_mode="none"`` drops the physical factor (ablation).
* :class:`SliceAlign`: at stage 3 a small head predicts an in-plane offset field between each pair of adjacent slices;
  both neighbours are warped onto the current slice before they are fused.  The offsets are supervised for free with
  :func:`shift_recovery_loss`: known per-slice shifts are injected into a view (:func:`apply_slice_shifts`) and the
  predicted offsets must change by exactly the injected relative shift.

``forward(image, geometry)`` takes a native volume ``[B, 1, D, H, W]`` in ``[0, 1]`` and a geometry mapping with
``shape`` (D, H, W) and ``spacing_zyx`` (mm); it returns ``{"logits": [B, 4, D, H, W], "features": (f0..f4)}``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DEFAULT_CHANNELS = (24, 48, 96, 160, 192)
DEFAULT_SPACING_REF_MM = 10.0
PARAMETER_CAP = 3_000_000
EXPORT_SCHEMA = "sasnet.export.v1"
NUM_CLASSES = 4  # background, RV, MYO, LV (LA is stored in class 3)
ENCODER_PREFIXES = ("e0.", "e1.", "e2.", "e3.", "e4.", "align3.", "zssm3.", "zssm4.")


# ---------------------------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------------------------

def native_shape(geometry: Any, image: Tensor) -> tuple[int, int, int]:
    """Native ``(D, H, W)`` from ``geometry["shape"]``; the image may be padded in H/W beyond it."""
    if geometry is None:
        return tuple(int(v) for v in image.shape[-3:])
    if isinstance(geometry, Mapping) and geometry.get("shape") is not None and len(geometry["shape"]) == 3:
        result = tuple(int(v) for v in geometry["shape"])
        if result[0] != image.shape[-3] or result[1] > image.shape[-2] or result[2] > image.shape[-1]:
            raise ValueError("geometry shape is incompatible with the image")
        return result
    raise ValueError("geometry must provide the native shape (D, H, W)")


def slice_spacing_mm(geometry: Any, batch: int, device: torch.device) -> Tensor:
    """Per-sample through-plane spacing ``[B]`` in mm (one mapping for the batch, or one mapping per sample)."""

    def one(g: Any) -> float:
        if not isinstance(g, Mapping) or "spacing_zyx" not in g:
            raise ValueError("SASNet requires geometry['spacing_zyx'] (physical slice spacing)")
        value = float(g["spacing_zyx"][0])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("slice spacing must be a finite positive number of mm")
        return value

    if isinstance(geometry, Mapping):
        values = [one(geometry)] * batch
    elif isinstance(geometry, Sequence) and not isinstance(geometry, (str, bytes)) and len(geometry) == batch:
        values = [one(g) for g in geometry]
    else:
        raise ValueError("geometry must be a mapping or one mapping per batch sample")
    return torch.tensor(values, dtype=torch.float32, device=device)


def _pad16(x: Tensor) -> tuple[Tensor, tuple[int, int]]:
    """Replicate-pad H and W up to multiples of 16 (four in-plane down-samplings)."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph, 0, 0), mode="replicate")
    return x, (h, w)


def skip_dropout(value: Tensor, probability: float, training: bool, generator: torch.Generator | None = None) -> Tensor:
    """Channel dropout of a skip/bottleneck tensor (UniMatch feature perturbation); a private generator keeps it replayable."""
    if not training or probability <= 0.0:
        return value
    if generator is None:
        return F.dropout3d(value, p=probability, training=True)
    keep = torch.rand((value.shape[0], value.shape[1], 1, 1, 1), device=value.device, generator=generator) >= probability
    return value * keep.to(value.dtype) / (1.0 - probability)


class ConvNormAct(nn.Module):
    """Conv3d (no bias) + InstanceNorm3d (affine) + GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: tuple[int, int, int] = (1, 3, 3), stride: tuple[int, int, int] = (1, 1, 1)) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=tuple(k // 2 for k in kernel), bias=False)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True, eps=1e-5)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------------------------------------------
# PhysZSSM
# ---------------------------------------------------------------------------------------------------------------

def _selective_scan(u: Tensor, dt: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
    """Diagonal selective scan.  ``u, dt``: ``[N, L, E]``; ``a``: ``[E, S]`` (negative); ``b, c``: ``[N, L, S]``.

    ``h_t = exp(dt_t * a) * h_{t-1} + dt_t * u_t * b_t``; ``y_t = <h_t, c_t> + d * u_t``.  ``L`` is the number of slices
    (a few dozen at most), so a plain loop is cheap and needs no custom kernel.
    """
    n, length, e = u.shape
    h = u.new_zeros(n, e, a.shape[1])
    ys = []
    for t in range(length):
        dt_t = dt[:, t, :, None]
        h = torch.exp(dt_t * a) * h + (dt_t * u[:, t, :, None]) * b[:, t, None, :]
        ys.append((h * c[:, t, None, :]).sum(-1))
    return torch.stack(ys, 1) + u * d


class _Direction(nn.Module):
    """Input-dependent ``dt, B, C`` and the diagonal ``A`` of one scan direction."""

    def __init__(self, inner: int, state: int, rank: int, dt_min: float, dt_max: float) -> None:
        super().__init__()
        self.x_proj = nn.Linear(inner, rank + 2 * state, bias=False)
        self.dt_proj = nn.Linear(rank, inner, bias=True)
        self.rank, self.state = rank, state
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state + 1, dtype=torch.float32)).repeat(inner, 1))
        self.d = nn.Parameter(torch.ones(inner))
        # Mamba-style dt initialisation: softplus(bias) log-uniform in [dt_min, dt_max].
        dt = torch.exp(torch.rand(inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, u: Tensor, step_scale: Tensor) -> Tensor:
        """``u`` ``[N, L, E]``; ``step_scale`` ``[N, L, 1]`` (physical factor x slice validity)."""
        dt_raw, b, c = self.x_proj(u).split((self.rank, self.state, self.state), dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw)) * step_scale
        return _selective_scan(u, dt, -torch.exp(self.a_log), b, c, self.d)


class PhysZSSM(nn.Module):
    """Pre-norm bidirectional selective SSM along z with a gated output and a residual connection.

    ``step = softplus(dt(x)) * (s_z / s_ref)`` with ``spacing_mode="physical"``; ``"none"`` uses ``step = softplus(dt(x))``.
    Invalid (padded) slices get ``step = 0``: they leave the state unchanged and their outputs are zeroed.

    ``zgate="token"`` (an explored variant, not the main model) multiplies the residual update by a per-token,
    per-channel gate ``sigmoid(W norm(x) + b)`` with ``W = 0`` and ``b = zgate_bias`` at initialisation.
    """

    def __init__(self, dim: int, *, state: int = 8, expand: float = 1.0, spacing_mode: str = "physical",
                 spacing_ref_mm: float = DEFAULT_SPACING_REF_MM, dt_min: float = 0.01, dt_max: float = 0.5,
                 zgate: str = "none", zgate_bias: float = 2.0) -> None:
        super().__init__()
        if zgate not in ("none", "token"):
            raise ValueError("zgate must be 'none' or 'token'")
        if spacing_mode not in ("physical", "none"):
            raise ValueError("spacing_mode must be 'physical' or 'none'")
        if spacing_ref_mm <= 0:
            raise ValueError("spacing_ref_mm must be positive")
        inner = max(4, int(round(dim * expand)))
        self.dim, self.inner, self.spacing_mode, self.spacing_ref_mm = dim, inner, spacing_mode, float(spacing_ref_mm)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * inner, bias=False)
        rank = max(1, math.ceil(dim / 16))
        self.forward_dir = _Direction(inner, state, rank, dt_min, dt_max)
        self.backward_dir = _Direction(inner, state, rank, dt_min, dt_max)
        self.out_proj = nn.Linear(inner, dim, bias=False)
        self.zgate = None
        if zgate == "token":
            self.zgate = nn.Linear(dim, dim)
            nn.init.zeros_(self.zgate.weight)
            nn.init.constant_(self.zgate.bias, float(zgate_bias))

    def forward(self, x: Tensor, spacing_mm: Tensor, slice_valid: Tensor | None = None) -> Tensor:
        """``x`` ``[B, C, D, H, W]``; ``spacing_mm`` ``[B]``; ``slice_valid`` ``[B, D]`` bool or ``None``."""
        bsz, ch, depth, height, width = x.shape
        tokens = x.permute(0, 3, 4, 2, 1).reshape(bsz * height * width, depth, ch)
        factor = spacing_mm / self.spacing_ref_mm if self.spacing_mode == "physical" else torch.ones_like(spacing_mm)
        scale = factor.to(x.dtype)[:, None].expand(bsz, depth)
        valid = torch.ones(bsz, depth, dtype=torch.bool, device=x.device) if slice_valid is None else slice_valid.to(torch.bool)
        scale = scale * valid.to(x.dtype)
        scale = scale[:, None, None, :].expand(bsz, height, width, depth).reshape(-1, depth, 1)
        vmask = valid[:, None, None, :].expand(bsz, height, width, depth).reshape(-1, depth, 1).to(x.dtype)
        normed = self.norm(tokens)
        u, gate = self.in_proj(normed).chunk(2, dim=-1)
        u = F.silu(u) * vmask
        y_f = self.forward_dir(u, scale)
        y_b = self.backward_dir(u.flip(1), scale.flip(1)).flip(1)
        delta = self.out_proj((y_f + y_b) * F.silu(gate)) * vmask
        if self.zgate is not None:
            delta = delta * torch.sigmoid(self.zgate(normed))
        out = tokens + delta
        return out.reshape(bsz, height, width, depth, ch).permute(0, 4, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------------------------------------------
# SliceAlign
# ---------------------------------------------------------------------------------------------------------------

def _warp_inplane(x: Tensor, flow: Tensor) -> Tensor:
    """Sample ``x`` ``[N, C, H, W]`` at ``p + flow``; ``flow`` ``[N, 2, H, W]`` in cells (dy, dx); zeros outside."""
    n, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype), torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    sy = gy[None] + flow[:, 0]
    sx = gx[None] + flow[:, 1]
    grid = torch.stack((2 * sx / max(w - 1, 1) - 1, 2 * sy / max(h - 1, 1) - 1), dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class SliceAlign(nn.Module):
    """Predict offsets between adjacent slices, warp both neighbours onto each slice and fuse them residually.

    ``flow[:, :, i]`` (cells, (dy, dx)) maps a location ``p`` of slice ``i`` to ``p + flow`` in slice ``i + 1``.  The last
    convolution of the offset head and the fusion projection are zero-initialised, so the module starts as the identity.
    """

    def __init__(self, dim: int, *, hidden: int = 32, max_offset_cells: float = 4.0) -> None:
        super().__init__()
        self.max_offset_cells = float(max_offset_cells)
        self.reduce = nn.Conv3d(dim, hidden, 1)
        self.offset = nn.Sequential(
            nn.Conv3d(2 * hidden, hidden, (1, 3, 3), padding=(0, 1, 1)), nn.GELU(),
            nn.Conv3d(hidden, 2, (1, 3, 3), padding=(0, 1, 1)),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.fuse = nn.Conv3d(2 * dim, dim, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def flows(self, x: Tensor) -> Tensor:
        """Offsets ``[B, 2, D-1, H, W]`` between slices ``i`` and ``i+1``."""
        r = self.reduce(x)
        pair = torch.cat((r[:, :, :-1], r[:, :, 1:]), dim=1)
        return torch.tanh(self.offset(pair)) * self.max_offset_cells

    def forward(self, x: Tensor, slice_valid: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, ch, depth, height, width = x.shape
        if depth < 2:
            return x, None
        flow = self.flows(x)
        pairs = bsz * (depth - 1)
        nxt = x[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(pairs, ch, height, width)
        prv = x[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(pairs, ch, height, width)
        f = flow.permute(0, 2, 1, 3, 4).reshape(pairs, 2, height, width)
        next_on_i = _warp_inplane(nxt, f).reshape(bsz, depth - 1, ch, height, width).permute(0, 2, 1, 3, 4)
        prev_on_next = _warp_inplane(prv, -f).reshape(bsz, depth - 1, ch, height, width).permute(0, 2, 1, 3, 4)
        if slice_valid is not None:
            v = slice_valid.to(x.dtype)
            pair_valid = (v[:, :-1] * v[:, 1:])[:, None, :, None, None]
            next_on_i, prev_on_next = next_on_i * pair_valid, prev_on_next * pair_valid
        zero = x.new_zeros(bsz, ch, 1, height, width)
        aligned_next = torch.cat((next_on_i, zero), dim=2)   # slice i sees warped i+1 (none for the last slice)
        aligned_prev = torch.cat((zero, prev_on_next), dim=2)  # slice i sees warped i-1 (none for the first slice)
        out = x + self.fuse(torch.cat((aligned_next, aligned_prev), dim=1))
        if slice_valid is not None:
            out = out * slice_valid.to(x.dtype)[:, None, :, None, None]
        return out, flow


# ---------------------------------------------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------------------------------------------

class SASNet(nn.Module):
    """In-plane U-Net (no z down-sampling) with :class:`SliceAlign` at stage 3 and :class:`PhysZSSM` at stages 3 and 4.

    Options (``model`` section of a config): ``channels`` (five widths), ``align`` (SliceAlign on/off),
    ``ssm_blocks`` (PhysZSSM blocks at stages 3 and 4), ``state``, ``spacing_mode`` (``physical``/``none``),
    ``spacing_ref_mm``, ``zgate`` (``{"mode": "token", "bias": 2.0}`` or absent).
    """

    def __init__(self, *, channels: Sequence[int] = DEFAULT_CHANNELS, align: bool = True, ssm_blocks: Sequence[int] = (1, 2),
                 state: int = 8, spacing_mode: str = "physical", spacing_ref_mm: float = DEFAULT_SPACING_REF_MM,
                 zgate: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if len(channels) != 5 or any(c < 8 or c % 8 for c in channels):
            raise ValueError("channels must be five widths that are multiples of 8")
        c0, c1, c2, c3, c4 = channels
        n3, n4 = (int(v) for v in ssm_blocks)
        zgate = dict(zgate or {})
        self.channels = channels
        self.options = {"align": bool(align), "ssm_blocks": [n3, n4], "state": int(state), "spacing_mode": str(spacing_mode),
                        "spacing_ref_mm": float(spacing_ref_mm), **({"zgate": zgate} if zgate.get("mode", "none") != "none" else {})}
        self.e0 = nn.Sequential(ConvNormAct(1, c0), ConvNormAct(c0, c0))
        self.e1 = nn.Sequential(ConvNormAct(c0, c1, stride=(1, 2, 2)), ConvNormAct(c1, c1))
        self.e2 = nn.Sequential(ConvNormAct(c1, c2, stride=(1, 2, 2)), ConvNormAct(c2, c2))
        self.e3 = nn.Sequential(ConvNormAct(c2, c3, stride=(1, 2, 2)), ConvNormAct(c3, c3, kernel=(3, 1, 1)), ConvNormAct(c3, c3))
        self.e4 = nn.Sequential(ConvNormAct(c3, c4, stride=(1, 2, 2)), ConvNormAct(c4, c4))
        self.align3 = SliceAlign(c3) if align else None

        def ssm(dim: int) -> PhysZSSM:
            return PhysZSSM(dim, state=int(state), spacing_mode=str(spacing_mode), spacing_ref_mm=float(spacing_ref_mm),
                            zgate=str(zgate.get("mode", "none")), zgate_bias=float(zgate.get("bias", 2.0)))

        self.zssm3 = nn.ModuleList(ssm(c3) for _ in range(n3))
        self.zssm4 = nn.ModuleList(ssm(c4) for _ in range(n4))
        self.up4, self.dec4 = nn.ConvTranspose3d(c4, c3, (1, 2, 2), (1, 2, 2)), ConvNormAct(2 * c3, c3)
        self.up3, self.dec3 = nn.ConvTranspose3d(c3, c2, (1, 2, 2), (1, 2, 2)), ConvNormAct(2 * c2, c2)
        self.up2, self.dec2 = nn.ConvTranspose3d(c2, c1, (1, 2, 2), (1, 2, 2)), ConvNormAct(2 * c1, c1)
        self.up1, self.dec1 = nn.ConvTranspose3d(c1, c0, (1, 2, 2), (1, 2, 2)), ConvNormAct(2 * c0, c0)
        self.logits_head = nn.Conv3d(c0, NUM_CLASSES, 1)
        # Feature-grid phase (step, offset) of stage k in native voxels: every down-sampling conv is 1x3x3 / stride 1x2x2
        # with padding 0x1x1, so stage k is a (1, 2^k, 2^k) lattice centred on native voxel 0.
        self.phases = tuple(((1, 2 ** k, 2 ** k), (0.0, 0.0, 0.0)) for k in range(5))

    @staticmethod
    def _check(image: Tensor) -> None:
        if image.ndim != 5 or image.shape[1] != 1 or not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError("input must be finite floating [B,1,D,H,W]")

    def _encode_to_e3(self, image: Tensor, geometry: Any, slice_valid: Tensor | None):
        self._check(image)
        native = native_shape(geometry, image)
        x = image[..., : native[1], : native[2]]
        if torch.any(x < 0) or torch.any(x > 1):
            raise ValueError("image must be normalized to [0,1]")
        spacing = slice_spacing_mm(geometry, image.shape[0], image.device)
        x, hw = _pad16(x)
        f0 = self.e0(x)
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        flow = None
        if self.align3 is not None:
            f3, flow = self.align3(f3, slice_valid)
        for block in self.zssm3:
            f3 = block(f3, spacing, slice_valid)
        return hw, spacing, (f0, f1, f2, f3), flow

    def encode(self, image: Tensor, geometry: Any) -> tuple[Tensor, Tensor]:
        """Stage-3 and stage-4 features (after SliceAlign / PhysZSSM), the taps distilled from CineMA."""
        _, spacing, (_, _, _, f3), _ = self._encode_to_e3(image, geometry, None)
        f4 = self.e4(f3)
        for block in self.zssm4:
            f4 = block(f4, spacing, None)
        return f3, f4

    def slice_flows(self, image: Tensor, geometry: Any) -> Tensor | None:
        """Stage-3 adjacent-slice offsets only (encoder up to stage 3), for :func:`shift_recovery_loss`."""
        if self.align3 is None:
            return None
        native = native_shape(geometry, image)
        x, _ = _pad16(image[..., : native[1], : native[2]])
        f3 = self.e3(self.e2(self.e1(self.e0(x))))
        return self.align3.flows(f3) if f3.shape[2] >= 2 else None

    def forward(self, image: Tensor, geometry: Any, *, skip_dropout_p: float = 0.0, dropout_generator: torch.Generator | None = None,
                slice_valid: Tensor | None = None) -> dict[str, Any]:
        (nh, nw), spacing, (f0, f1, f2, f3), flow = self._encode_to_e3(image, geometry, slice_valid)
        f4 = self.e4(f3)
        for block in self.zssm4:
            f4 = block(f4, spacing, slice_valid)

        def drop(t: Tensor) -> Tensor:
            return skip_dropout(t, skip_dropout_p, self.training, dropout_generator)

        d3 = self.dec4(torch.cat([self.up4(drop(f4)), drop(f3)], 1))
        d2 = self.dec3(torch.cat([self.up3(d3), drop(f2)], 1))
        d1 = self.dec2(torch.cat([self.up2(d2), drop(f1)], 1))
        d0 = self.dec1(torch.cat([self.up1(d1), drop(f0)], 1))
        logits = self.logits_head(d0)[..., :nh, :nw]
        out: dict[str, Any] = {"logits": logits, "features": (f0, f1, f2, f3, f4)}
        if flow is not None and self.training:
            out["slice_flow"] = flow
        return out


# ---------------------------------------------------------------------------------------------------------------
# Slice-shift injection and free supervision of the offset head
# ---------------------------------------------------------------------------------------------------------------

def sample_slice_shifts(batch: int, depth: int, max_px: int, *, device: torch.device | str = "cpu",
                        generator: torch.Generator | None = None) -> Tensor:
    """Integer per-slice in-plane shifts ``[B, D, 2]`` (dy, dx) uniform in ``[-max_px, max_px]``."""
    if max_px < 0:
        raise ValueError("max_px must be non-negative")
    return torch.randint(-max_px, max_px + 1, (batch, depth, 2), generator=generator).to(device=device, dtype=torch.float32)


def _warp_nearest(x: Tensor, flow: Tensor) -> Tensor:
    n, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype), torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack((2 * (gx[None] + flow[:, 1]) / max(w - 1, 1) - 1, 2 * (gy[None] + flow[:, 0]) / max(h - 1, 1) - 1), dim=-1)
    return F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=True)


def apply_slice_shifts(x: Tensor, shifts_px: Tensor, *, mode: str = "bilinear") -> tuple[Tensor, Tensor]:
    """Translate every slice of ``x`` ``[B, C, D, H, W]`` by ``shifts_px`` ``[B, D, 2]`` pixels (content moves by +shift).

    Returns the shifted tensor and an in-bounds mask ``[B, 1, D, H, W]``.  Use ``mode="nearest"`` for label maps.
    """
    if x.ndim != 5 or shifts_px.shape != (x.shape[0], x.shape[2], 2):
        raise ValueError("x must be [B,C,D,H,W] and shifts [B,D,2]")
    bsz, ch, depth, height, width = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(bsz * depth, ch, height, width)
    is_int = not flat.is_floating_point()
    src = flat.to(torch.float32) if is_int else flat
    flow = -shifts_px.reshape(bsz * depth, 2, 1, 1).to(src.dtype).expand(-1, -1, height, width)
    moved = _warp_inplane(src, flow) if mode == "bilinear" else _warp_nearest(src, flow)
    inside = _warp_inplane(torch.ones_like(src[:, :1]), flow) > 0.999
    if is_int:
        moved = moved.round().to(flat.dtype)
    moved = moved.reshape(bsz, depth, ch, height, width).permute(0, 2, 1, 3, 4)
    inside = inside.reshape(bsz, depth, 1, height, width).permute(0, 2, 1, 3, 4)
    return moved, inside


def shift_recovery_loss(flow_shifted: Tensor, flow_clean: Tensor, shifts_px: Tensor, *, stride: int = 8,
                        slice_valid: Tensor | None = None) -> Tensor:
    """Smooth-L1 between the change of predicted offsets and the injected relative shift.

    The natural misalignment of the stack is unknown, so the target is a difference: on the shifted view the offset from
    slice ``i`` to ``i+1`` must grow by ``(s_{i+1} - s_i) / stride`` cells relative to the clean view (``flow_clean`` is
    detached).  ``stride`` is the in-plane stride of the stage that predicts the offsets (8 for stage 3).
    """
    if flow_shifted.shape != flow_clean.shape or flow_shifted.ndim != 5 or flow_shifted.shape[1] != 2:
        raise ValueError("flows must both be [B,2,D-1,H,W]")
    rel = (shifts_px[:, 1:] - shifts_px[:, :-1]) / float(stride)  # [B, D-1, 2]
    target = rel.permute(0, 2, 1)[..., None, None].to(flow_shifted.dtype)
    err = F.smooth_l1_loss(flow_shifted - flow_clean.detach(), target.expand_as(flow_shifted), reduction="none")
    if slice_valid is None:
        return err.mean()
    v = slice_valid.to(err.dtype)
    w = (v[:, :-1] * v[:, 1:])[:, None, :, None, None].expand_as(err)
    return (err * w).sum() / w.sum().clamp_min(1.0)


def slice_shift_aux(model: nn.Module, images: Sequence[tuple[Tensor, Any]], *, seed: int, update: int, max_px: int) -> Tensor | None:
    """Shift-recovery loss averaged over ``(image, geometry)`` pairs; deterministic in ``(seed, update)`` for resume.

    Only the offset head and the encoder up to stage 3 receive gradient; the clean-view offsets carry no gradient.
    Returns ``None`` without an offset head or when every image has a single slice.
    """
    if getattr(model, "align3", None) is None:
        return None
    gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(update))
    losses = []
    for image, geometry in images:
        if image.shape[2] < 2:
            continue
        shifts = sample_slice_shifts(image.shape[0], image.shape[2], max_px, device=image.device, generator=gen)
        shifted, _ = apply_slice_shifts(image, shifts)
        with torch.no_grad():
            clean = model.slice_flows(image, geometry)
        moved = model.slice_flows(shifted.clamp(0.0, 1.0), geometry)
        if clean is None or moved is None:
            continue
        losses.append(shift_recovery_loss(moved, clean, shifts, stride=8))
    return torch.stack(losses).mean() if losses else None


# ---------------------------------------------------------------------------------------------------------------
# Construction and export
# ---------------------------------------------------------------------------------------------------------------

def build_model(model_cfg: Mapping[str, Any] | None = None, *, parameter_cap: int | None = PARAMETER_CAP) -> SASNet:
    """Build from the ``model`` config section; the default budget is 3 M parameters (``parameter_cap=None`` lifts it)."""
    cfg = dict(model_cfg or {})
    cap = cfg.pop("parameter_cap", parameter_cap)
    model = SASNet(**cfg)
    total = sum(p.numel() for p in model.parameters())
    if cap is not None and total > int(cap):
        raise ValueError(f"SASNet has {total} parameters, above the {cap} cap")
    return model


def export_model(model: SASNet, path: Any, *, epoch: int | None = None) -> dict[str, Any]:
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"schema": EXPORT_SCHEMA, "channels": list(model.channels), "options": model.options, "state": state, "selected_epoch": epoch}, path)
    return {"path": str(path), "parameters": sum(v.numel() for v in state.values())}


def load_model(path: Any, *, map_location: Any = "cpu") -> SASNet:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("schema") != EXPORT_SCHEMA:
        raise ValueError("not a SASNet export")
    model = SASNet(channels=payload["channels"], **payload["options"])
    model.load_state_dict(payload["state"], strict=True)
    return model.eval()


__all__ = [
    "SASNet", "PhysZSSM", "SliceAlign", "build_model", "export_model", "load_model", "apply_slice_shifts", "sample_slice_shifts",
    "shift_recovery_loss", "slice_shift_aux", "slice_spacing_mm", "skip_dropout", "DEFAULT_CHANNELS", "PARAMETER_CAP", "ENCODER_PREFIXES",
]
