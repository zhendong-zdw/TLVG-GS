import torch
import torch.nn.functional as F

# ---------- RGB -> Lab (differentiable) ----------
def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to linear RGB. x in [0,1]"""
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def rgb_to_lab(rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert RGB to Lab color space.
    
    Args:
        rgb: [3, H, W] in [0, 1], sRGB
        eps: small epsilon for numerical stability
    
    Returns:
        Lab [3, H, W], L in [0,100], a,b roughly [-128,127]
    """
    # DIAGNOSTIC: Check input
    if torch.isnan(rgb).any() or torch.isinf(rgb).any():
        print(f"DIAGNOSTIC: rgb_to_lab input contains NaN/Inf")
        print(f"  rgb: NaN={torch.isnan(rgb).sum().item()}, Inf={torch.isinf(rgb).sum().item()}, shape={rgb.shape}")
        if torch.isfinite(rgb).any():
            finite_rgb = rgb[torch.isfinite(rgb)]
            print(f"  rgb (finite): min={finite_rgb.min().item():.6f}, max={finite_rgb.max().item():.6f}")
    
    r, g, b = rgb[0], rgb[1], rgb[2]
    rgb_lin = _srgb_to_linear(torch.stack([r, g, b], dim=0))
    
    # DIAGNOSTIC: Check after linear conversion
    if torch.isnan(rgb_lin).any() or torch.isinf(rgb_lin).any():
        print(f"DIAGNOSTIC: rgb_to_lab rgb_lin contains NaN/Inf after _srgb_to_linear")
        print(f"  rgb_lin: NaN={torch.isnan(rgb_lin).sum().item()}, Inf={torch.isinf(rgb_lin).sum().item()}")
        print(f"  rgb range: [{rgb.min().item():.6f}, {rgb.max().item():.6f}]")

    # sRGB D65 matrix
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], device=rgb.device, dtype=rgb.dtype)

    # XYZ: [3, H, W]
    xyz = torch.einsum("ij,jhw->ihw", M, rgb_lin)
    X, Y, Z = xyz[0], xyz[1], xyz[2]
    
    # DIAGNOSTIC: Check XYZ
    if torch.isnan(xyz).any() or torch.isinf(xyz).any():
        print(f"DIAGNOSTIC: rgb_to_lab xyz contains NaN/Inf")
        print(f"  xyz: NaN={torch.isnan(xyz).sum().item()}, Inf={torch.isinf(xyz).sum().item()}")
        print(f"  X range: [{X.min().item():.6f}, {X.max().item():.6f}]")
        print(f"  Y range: [{Y.min().item():.6f}, {Y.max().item():.6f}]")
        print(f"  Z range: [{Z.min().item():.6f}, {Z.max().item():.6f}]")

    # D65 white point
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = X / Xn
    y = Y / Yn
    z = Z / Zn
    
    # DIAGNOSTIC: Check normalized xyz (check for zero or negative)
    if (x <= 0).any() or (y <= 0).any() or (z <= 0).any():
        neg_x = (x <= 0).sum().item()
        neg_y = (y <= 0).sum().item()
        neg_z = (z <= 0).sum().item()
        if neg_x > 0 or neg_y > 0 or neg_z > 0:
            print(f"DIAGNOSTIC: rgb_to_lab has non-positive xyz: x<=0={neg_x}, y<=0={neg_y}, z<=0={neg_z}")
            print(f"  X min={X.min().item():.6f}, Y min={Y.min().item():.6f}, Z min={Z.min().item():.6f}")
    
    # FIX: Clamp x, y, z to small positive value to avoid numerical issues
    # The Lab conversion requires positive values, and very small values can cause NaN in backward
    x = torch.clamp(x, min=eps)
    y = torch.clamp(y, min=eps)
    z = torch.clamp(z, min=eps)

    def f(t):
        delta = 6/29
        # Use more numerically stable computation
        # For t <= delta^3: f(t) = t / (3*delta^2) + 4/29
        # For t > delta^3: f(t) = t^(1/3)
        # Clamp t to ensure it's positive
        t_clamped = torch.clamp(t, min=eps)
        return torch.where(t_clamped > delta**3, t_clamped ** (1/3), t_clamped / (3*delta**2) + 4/29)

    fx, fy, fz = f(x), f(y), f(z)
    
    # DIAGNOSTIC: Check f(x), f(y), f(z)
    if torch.isnan(fx).any() or torch.isnan(fy).any() or torch.isnan(fz).any():
        print(f"DIAGNOSTIC: rgb_to_lab f(xyz) contains NaN")
        print(f"  fx: NaN={torch.isnan(fx).sum().item()}, fy: NaN={torch.isnan(fy).sum().item()}, fz: NaN={torch.isnan(fz).sum().item()}")
        print(f"  x range: [{x.min().item():.6f}, {x.max().item():.6f}]")
        print(f"  y range: [{y.min().item():.6f}, {y.max().item():.6f}]")
        print(f"  z range: [{z.min().item():.6f}, {z.max().item():.6f}]")
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    lab = torch.stack([L, a, b], dim=0)
    
    # DIAGNOSTIC: Check final Lab
    if torch.isnan(lab).any() or torch.isinf(lab).any():
        print(f"DIAGNOSTIC: rgb_to_lab final Lab contains NaN/Inf")
        print(f"  Lab: NaN={torch.isnan(lab).sum().item()}, Inf={torch.isinf(lab).sum().item()}")
        print(f"  L: NaN={torch.isnan(L).sum().item()}, a: NaN={torch.isnan(a).sum().item()}, b: NaN={torch.isnan(b).sum().item()}")
    
    return lab

def ab_stat_loss(render_rgb: torch.Tensor,
                 ref_rgb: torch.Tensor,
                 mask: torch.Tensor = None,
                 eps: float = 1e-6) -> torch.Tensor:
    """
    Global Lab color preservation loss using a/b channel statistics.
    
    Only constrains a/b channels (not L) to preserve style's brightness/contrast.
    
    Args:
        render_rgb: [3, H, W] in [0, 1], stylized render image
        ref_rgb: [3, H, W] in [0, 1], reference image (pre or content)
        mask: [1, H, W] or [H, W], 1 for valid region, optional
        eps: small epsilon for numerical stability
    
    Returns:
        Loss value (scalar tensor)
    """
    # DIAGNOSTIC: Check inputs
    if torch.isnan(render_rgb).any() or torch.isinf(render_rgb).any():
        print(f"DIAGNOSTIC: ab_stat_loss input render_rgb contains NaN/Inf")
        print(f"  render_rgb: NaN={torch.isnan(render_rgb).sum().item()}, Inf={torch.isinf(render_rgb).sum().item()}, shape={render_rgb.shape}")
        if torch.isfinite(render_rgb).any():
            finite_rgb = render_rgb[torch.isfinite(render_rgb)]
            print(f"  render_rgb (finite): min={finite_rgb.min().item():.6f}, max={finite_rgb.max().item():.6f}")
    
    if torch.isnan(ref_rgb).any() or torch.isinf(ref_rgb).any():
        print(f"DIAGNOSTIC: ab_stat_loss input ref_rgb contains NaN/Inf")
        print(f"  ref_rgb: NaN={torch.isnan(ref_rgb).sum().item()}, Inf={torch.isinf(ref_rgb).sum().item()}, shape={ref_rgb.shape}")
    
    lab_r = rgb_to_lab(render_rgb)
    lab_ref = rgb_to_lab(ref_rgb)
    
    # DIAGNOSTIC: Check Lab conversion
    if torch.isnan(lab_r).any() or torch.isinf(lab_r).any():
        print(f"DIAGNOSTIC: ab_stat_loss lab_r contains NaN/Inf after rgb_to_lab")
        print(f"  lab_r: NaN={torch.isnan(lab_r).sum().item()}, Inf={torch.isinf(lab_r).sum().item()}")
        print(f"  render_rgb range: [{render_rgb.min().item():.6f}, {render_rgb.max().item():.6f}]")
    
    if torch.isnan(lab_ref).any() or torch.isinf(lab_ref).any():
        print(f"DIAGNOSTIC: ab_stat_loss lab_ref contains NaN/Inf after rgb_to_lab")
        print(f"  lab_ref: NaN={torch.isnan(lab_ref).sum().item()}, Inf={torch.isinf(lab_ref).sum().item()}")

    ab_r = lab_r[1:3]      # [2, H, W] - only a and b channels
    ab_ref = lab_ref[1:3]  # [2, H, W]
    
    # DIAGNOSTIC: Check ab channels
    if torch.isnan(ab_r).any() or torch.isinf(ab_r).any():
        print(f"DIAGNOSTIC: ab_stat_loss ab_r contains NaN/Inf")
        print(f"  ab_r: NaN={torch.isnan(ab_r).sum().item()}, Inf={torch.isinf(ab_r).sum().item()}")
        print(f"  lab_r[1] (a channel): NaN={torch.isnan(lab_r[1]).sum().item()}, Inf={torch.isinf(lab_r[1]).sum().item()}")
        print(f"  lab_r[2] (b channel): NaN={torch.isnan(lab_r[2]).sum().item()}, Inf={torch.isinf(lab_r[2]).sum().item()}")

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.clamp(0, 1)
        # expand to [2, H, W]
        m = mask.expand_as(ab_r)

        # masked mean/std
        w = m.sum(dim=(1, 2)).clamp_min(eps)
        mu_r = (ab_r * m).sum(dim=(1, 2)) / w
        mu_ref = (ab_ref * m).sum(dim=(1, 2)) / w

        var_r = ((ab_r - mu_r[:, None, None])**2 * m).sum(dim=(1, 2)) / w
        var_ref = ((ab_ref - mu_ref[:, None, None])**2 * m).sum(dim=(1, 2)) / w
    else:
        mu_r = ab_r.mean(dim=(1, 2))
        mu_ref = ab_ref.mean(dim=(1, 2))
        var_r = ab_r.var(dim=(1, 2), unbiased=False)
        var_ref = ab_ref.var(dim=(1, 2), unbiased=False)
    
    # DIAGNOSTIC: Check mean and variance
    if torch.isnan(mu_r).any() or torch.isnan(var_r).any():
        print(f"DIAGNOSTIC: ab_stat_loss mu_r or var_r contains NaN")
        print(f"  mu_r: NaN={torch.isnan(mu_r).sum().item()}, shape={mu_r.shape}, values={mu_r}")
        print(f"  var_r: NaN={torch.isnan(var_r).sum().item()}, shape={var_r.shape}, values={var_r}")
        print(f"  ab_r stats: min={ab_r.min().item():.6f}, max={ab_r.max().item():.6f}, mean={ab_r.mean().item():.6f}")

    std_r = torch.sqrt(var_r + eps)
    std_ref = torch.sqrt(var_ref + eps)
    
    # DIAGNOSTIC: Check std
    if torch.isnan(std_r).any() or torch.isnan(std_ref).any():
        print(f"DIAGNOSTIC: ab_stat_loss std_r or std_ref contains NaN")
        print(f"  std_r: NaN={torch.isnan(std_r).sum().item()}, var_r+eps min={(var_r+eps).min().item():.6f}")

    # L1 loss on mean and std
    loss = (mu_r - mu_ref).abs().mean() + (std_r - std_ref).abs().mean()
    
    # DIAGNOSTIC: Check final loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"DIAGNOSTIC: ab_stat_loss final loss is NaN/Inf: {loss.item()}")
        print(f"  mu_diff: {(mu_r - mu_ref).abs().mean().item()}")
        print(f"  std_diff: {(std_r - std_ref).abs().mean().item()}")
    
    return loss

def ab_patch_stat_loss(render_rgb: torch.Tensor,
                      ref_rgb: torch.Tensor,
                      grid: int = 8,
                      eps: float = 1e-6) -> torch.Tensor:
    """
    Patch-wise Lab color preservation loss.
    
    Divides image into grid x grid patches and computes statistics for each patch.
    More effective for local color shifts (e.g., "face green, wall yellow").
    
    Args:
        render_rgb: [3, H, W] in [0, 1], stylized render image
        ref_rgb: [3, H, W] in [0, 1], reference image
        grid: number of patches per dimension (grid x grid patches)
        eps: small epsilon for numerical stability
    
    Returns:
        Loss value (scalar tensor)
    """
    lab_r = rgb_to_lab(render_rgb)[1:3].unsqueeze(0)    # [1, 2, H, W] - only a and b
    lab_ref = rgb_to_lab(ref_rgb)[1:3].unsqueeze(0)     # [1, 2, H, W]
    H, W = render_rgb.shape[1], render_rgb.shape[2]

    # pool to grid x grid stats using avg pooling
    kh, kw = max(1, H // grid), max(1, W // grid)
    # mean
    mu_r = F.avg_pool2d(lab_r, kernel_size=(kh, kw), stride=(kh, kw))
    mu_ref = F.avg_pool2d(lab_ref, kernel_size=(kh, kw), stride=(kh, kw))
    # second moment
    m2_r = F.avg_pool2d(lab_r**2, kernel_size=(kh, kw), stride=(kh, kw))
    m2_ref = F.avg_pool2d(lab_ref**2, kernel_size=(kh, kw), stride=(kh, kw))

    std_r = torch.sqrt((m2_r - mu_r**2).clamp_min(0) + eps)
    std_ref = torch.sqrt((m2_ref - mu_ref**2).clamp_min(0) + eps)

    # L1 loss on mean and std
    return (mu_r - mu_ref).abs().mean() + (std_r - std_ref).abs().mean()

def content_loss_fn(render_feats_list, scene_feats_list):
    content_loss = 0
    for (render_feat, scene_feat) in zip(render_feats_list, scene_feats_list):
        content_loss += torch.mean((render_feat - scene_feat) ** 2)
    return content_loss

def image_tv_loss_fn(render_image):
    image = render_image.unsqueeze(0)
    w_variance = torch.mean(torch.pow(image[:, :, :-1] - image[:, :, 1:], 2))
    h_variance = torch.mean(torch.pow(image[:, :-1, :] - image[:, 1:, :], 2))
    return (h_variance + w_variance) / 2.0

def _footprint_cov2d(
    sum_E: torch.Tensor,
    sum_E_dx: torch.Tensor,
    sum_E_dy: torch.Tensor,
    sum_E_xx: torch.Tensor,
    sum_E_xy: torch.Tensor,
    sum_E_yy: torch.Tensor,
    eps: float = 1e-6,
):
    # DIAGNOSTIC: Check inputs
    if torch.isnan(sum_E).any() or torch.isinf(sum_E).any():
        print(f"DIAGNOSTIC: _footprint_cov2d input sum_E contains NaN/Inf: NaN={torch.isnan(sum_E).sum().item()}, Inf={torch.isinf(sum_E).sum().item()}")
    
    E = sum_E.clamp_min(eps)
    mx = sum_E_dx / E
    my = sum_E_dy / E
    xx = sum_E_xx / E - mx * mx
    xy = sum_E_xy / E - mx * my
    yy = sum_E_yy / E - my * my
    
    # DIAGNOSTIC: Check outputs
    if torch.isnan(xx).any() or torch.isnan(xy).any() or torch.isnan(yy).any():
        print(f"DIAGNOSTIC: _footprint_cov2d outputs contain NaN")
        print(f"  xx: NaN={torch.isnan(xx).sum().item()}, xy: NaN={torch.isnan(xy).sum().item()}, yy: NaN={torch.isnan(yy).sum().item()}")
        print(f"  E stats: min={E.min().item():.6f}, max={E.max().item():.6f}, mean={E.mean().item():.6f}")
        print(f"  mx stats: min={mx.min().item():.6f}, max={mx.max().item():.6f}, mean={mx.mean().item():.6f}")
        print(f"  my stats: min={my.min().item():.6f}, max={my.max().item():.6f}, mean={my.mean().item():.6f}")
    
    return E, xx, xy, yy

def _eig2x2(xx: torch.Tensor, xy: torch.Tensor, yy: torch.Tensor, eps: float = 1e-12):
    # DIAGNOSTIC: Check inputs
    if torch.isnan(xx).any() or torch.isnan(xy).any() or torch.isnan(yy).any():
        print(f"DIAGNOSTIC: _eig2x2 inputs contain NaN: xx={torch.isnan(xx).sum().item()}, xy={torch.isnan(xy).sum().item()}, yy={torch.isnan(yy).sum().item()}")
    
    tr = xx + yy
    delta = torch.sqrt((xx - yy) ** 2 + 4 * xy * xy + eps)
    lam1 = 0.5 * (tr + delta)
    lam2 = 0.5 * (tr - delta)
    vx = 2 * xy
    vy = yy - xx + delta
    n = torch.sqrt(vx * vx + vy * vy + eps)
    v1 = torch.stack((vx / n, vy / n), dim=-1)
    
    # DIAGNOSTIC: Check outputs
    if torch.isnan(lam1).any() or torch.isnan(lam2).any():
        print(f"DIAGNOSTIC: _eig2x2 outputs contain NaN: lam1={torch.isnan(lam1).sum().item()}, lam2={torch.isnan(lam2).sum().item()}")
        print(f"  tr stats: min={tr.min().item():.6f}, max={tr.max().item():.6f}, mean={tr.mean().item():.6f}")
        print(f"  delta stats: min={delta.min().item():.6f}, max={delta.max().item():.6f}, mean={delta.mean().item():.6f}")
        print(f"  (xx-yy)^2 stats: min={((xx-yy)**2).min().item():.6f}, max={((xx-yy)**2).max().item():.6f}")
        print(f"  4*xy^2 stats: min={(4*xy*xy).min().item():.6f}, max={(4*xy*xy).max().item():.6f}")
    
    return lam1, lam2, v1

def brush_shape_loss_fn(
    sum_E: torch.Tensor,
    sum_E_dx: torch.Tensor,
    sum_E_dy: torch.Tensor,
    sum_E_xx: torch.Tensor,
    sum_E_xy: torch.Tensor,
    sum_E_yy: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    _, xx, xy, yy = _footprint_cov2d(sum_E, sum_E_dx, sum_E_dy, sum_E_xx, sum_E_xy, sum_E_yy, eps=eps)
    lam1, lam2, _ = _eig2x2(xx, xy, yy)
    lam1 = lam1.clamp_min(0)
    lam2 = lam2.clamp_min(0)
    
    # DIAGNOSTIC: Check before division
    if torch.isnan(lam1).any() or torch.isnan(lam2).any():
        print(f"DIAGNOSTIC: brush_shape_loss_fn: lam1 or lam2 contains NaN")
        print(f"  lam1: NaN={torch.isnan(lam1).sum().item()}, shape={lam1.shape}, range=[{lam1.min().item():.6f}, {lam1.max().item():.6f}]")
        print(f"  lam2: NaN={torch.isnan(lam2).sum().item()}, shape={lam2.shape}, range=[{lam2.min().item():.6f}, {lam2.max().item():.6f}]")
    
    ratio = lam2 / (lam1 + eps)
    
    # DIAGNOSTIC: Check ratio
    if torch.isnan(ratio).any() or torch.isinf(ratio).any():
        print(f"DIAGNOSTIC: brush_shape_loss_fn: ratio contains NaN/Inf")
        print(f"  ratio: NaN={torch.isnan(ratio).sum().item()}, Inf={torch.isinf(ratio).sum().item()}")
        print(f"  lam1+eps stats: min={(lam1+eps).min().item():.6f}, max={(lam1+eps).max().item():.6f}, mean={(lam1+eps).mean().item():.6f}")
        print(f"  lam2 stats: min={lam2.min().item():.6f}, max={lam2.max().item():.6f}, mean={lam2.mean().item():.6f}")
    
    loss = ratio.mean()
    
    # DIAGNOSTIC: Check final loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"DIAGNOSTIC: brush_shape_loss_fn: final loss is NaN/Inf: {loss.item()}")
    
    return loss

def stroke_direction_loss_fn(sum_E_dt2: torch.Tensor, sum_E_dn2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Stroke direction loss: ratio of normal to tangent direction energy.
    
    Args:
        sum_E_dt2: Energy in tangent direction [N]
        sum_E_dn2: Energy in normal direction [N]
        eps: Small epsilon for numerical stability
    
    Returns:
        Mean ratio loss (scalar)
    """
    # DIAGNOSTIC: Check for invalid values - report but don't fix
    if not torch.isfinite(sum_E_dt2).all() or not torch.isfinite(sum_E_dn2).all():
        nan_dt2 = torch.isnan(sum_E_dt2).sum().item()
        inf_dt2 = torch.isinf(sum_E_dt2).sum().item()
        nan_dn2 = torch.isnan(sum_E_dn2).sum().item()
        inf_dn2 = torch.isinf(sum_E_dn2).sum().item()
        print(f"DIAGNOSTIC: stroke_direction_loss inputs contain NaN/Inf")
        print(f"  sum_E_dt2: NaN={nan_dt2}, Inf={inf_dt2}, shape={sum_E_dt2.shape}")
        print(f"  sum_E_dn2: NaN={nan_dn2}, Inf={inf_dn2}, shape={sum_E_dn2.shape}")
        if torch.isfinite(sum_E_dt2).any():
            finite_dt2 = sum_E_dt2[torch.isfinite(sum_E_dt2)]
            print(f"  sum_E_dt2 (finite): min={finite_dt2.min().item():.6f}, max={finite_dt2.max().item():.6f}, mean={finite_dt2.mean().item():.6f}")
        if torch.isfinite(sum_E_dn2).any():
            finite_dn2 = sum_E_dn2[torch.isfinite(sum_E_dn2)]
            print(f"  sum_E_dn2 (finite): min={finite_dn2.min().item():.6f}, max={finite_dn2.max().item():.6f}, mean={finite_dn2.mean().item():.6f}")
        # Don't return 0 - let NaN propagate to see where it comes from
    
    # Clamp to prevent extreme values
    sum_E_dt2 = torch.clamp(sum_E_dt2, min=0.0, max=1e10)
    sum_E_dn2 = torch.clamp(sum_E_dn2, min=0.0, max=1e10)
    
    # Compute ratio with better numerical stability
    # If sum_E_dt2 is too small, the ratio becomes very large, so we need a larger eps
    denominator = sum_E_dt2 + eps
    ratio = sum_E_dn2 / denominator
    
    # Clamp ratio to prevent extreme values
    ratio = torch.clamp(ratio, min=0.0, max=1e6)
    
    # Check result for NaN/Inf
    if not torch.isfinite(ratio).all():
        print(f"Warning: stroke_direction_loss ratio contains NaN/Inf, returning 0")
        return torch.tensor(0.0, device=sum_E_dt2.device, dtype=sum_E_dt2.dtype, requires_grad=True)
    
    return ratio.mean()