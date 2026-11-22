"""Self-contained, fully vectorized PushT environment on torch.

Includes the T-block geometry, contact manifolds, impulse solver, exact
coverage reward, and vectorized RGB rasterizer. Import `PushTEnv` from here;
no other local modules are required.

On Apple Silicon the environment defaults to `mps`; use `device="cpu"` and
`dtype=torch.float64` for gym-pusht parity runs.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# ---------------------------------------------------------------- geometry
# 512x512 arena, walls are capsules of radius 2 centred on the arena border
WALL_MIN = 5.0
WALL_MAX = 506.0
WALL_RADIUS = 2.0

AGENT_RADIUS = 15.0

# T block: horizontal bar + stem, local coordinates relative to body.position
BLOCK_CENTERS = ((0.0, 15.0), (0.0, 75.0))
BLOCK_HALFS = ((60.0, 15.0), (15.0, 45.0))
BLOCK_CORNERS = (
    (-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (-60.0, 30.0),
    (-15.0, 30.0), (15.0, 30.0), (15.0, 120.0), (-15.0, 120.0),
)
BLOCK_AREA = 120.0 * 30.0 + 30.0 * 90.0

# pymunk body: mass 1, moment 3000 (about the body origin), CoG at (0, 45)
BLOCK_MASS = 1.0
BLOCK_INERTIA = 3000.0
BLOCK_COG = (0.0, 45.0)

GOAL_POSE = (256.0, 256.0, math.pi / 4)

# wall boxes approximating the pymunk capsules (inner faces at 7 / 504)
WALL_CENTERS = ((5.0, 255.5), (255.5, 5.0), (506.0, 255.5), (255.5, 506.0))
WALL_HALFS = ((2.0, 253.5), (253.5, 2.0), (2.0, 253.5), (253.5, 2.0))

# contact layout: 2 (agent vs each T rectangle) + 2 * 4 * 2 (rect vs wall)
N_CONTACTS = 18


_TENSOR_CACHE = {}
_ARANGE_CACHE = {}


def _t(values, ref: torch.Tensor) -> torch.Tensor:
    """Constant tensor on ``ref``'s device/dtype, cached across calls."""
    key = (values, ref.device, ref.dtype)
    out = _TENSOR_CACHE.get(key)
    if out is None:
        out = torch.tensor(values, dtype=ref.dtype, device=ref.device)
        _TENSOR_CACHE[key] = out
    return out


def _arange(n: int, ref: torch.Tensor) -> torch.Tensor:
    key = (n, ref.device)
    out = _ARANGE_CACHE.get(key)
    if out is None:
        out = torch.arange(n, device=ref.device)
        _ARANGE_CACHE[key] = out
    return out


def rotate(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate row vectors ``v`` about ``theta`` -> same shape as ``v``."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack((c * v[..., 0] - s * v[..., 1],
                        s * v[..., 0] + c * v[..., 1]), dim=-1)


def rotate_inv(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack((c * v[..., 0] + s * v[..., 1],
                        -s * v[..., 0] + c * v[..., 1]), dim=-1)


def rect_corners(pose: torch.Tensor, center, half) -> torch.Tensor:
    """World corners of a rectangle attached to ``pose`` -> (..., 4, 2)."""
    signs = _t(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)), pose)
    local = _t(center, pose) + signs * _t(half, pose)
    return pose[..., None, :2] + rotate(local, pose[..., 2, None])


def _gather_next(poly: torch.Tensor, count: torch.Tensor):
    """Cyclic successor over the valid prefix of a fixed-size polygon."""
    length = poly.shape[1]
    idx = _arange(length, poly)
    nxt = (idx[None, :] + 1) % torch.clamp(count[:, None], min=1)
    onehot = (nxt[..., None] == idx[None, None, :]).to(poly.dtype)
    return torch.sum(poly[:, None, :, :] * onehot[..., None], dim=2), onehot


def _clip_halfplane(poly: torch.Tensor, count: torch.Tensor, f: torch.Tensor):
    """Sutherland-Hodgman clip against ``f >= 0`` (SDF)."""
    p_next, onehot = _gather_next(poly, count)
    f_next = torch.sum(f[:, None, :] * onehot, dim=2)

    n, length = poly.shape[:2]
    idx = _arange(length, poly)
    active = idx[None, :] < count[:, None]

    inside = f >= 0.0
    inside_next = f_next >= 0.0
    cross = inside != inside_next

    cand_a = poly
    valid_a = inside & active

    denom = f - f_next
    safe = torch.abs(denom) > 1e-8
    t = torch.where(safe, f / torch.where(safe, denom, torch.ones_like(denom)), torch.zeros_like(f))
    cand_b = poly + t[..., None] * (p_next - poly)
    valid_b = cross & active

    vals = torch.stack((cand_a, cand_b), dim=2).reshape(n, 2 * length, 2)
    valid = torch.stack((valid_a, valid_b), dim=2).reshape(n, 2 * length)

    rank = torch.cumsum(valid.to(torch.int32), dim=1) - 1
    onehot = (rank[..., None] == idx[None, None, :]) & valid[..., None]
    out = torch.sum(vals[:, :, None, :] * onehot[..., None], dim=1)
    out_count = torch.clamp(torch.sum(valid, dim=1), max=length)
    return out, out_count


def polygon_area(poly: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    p_next, _ = _gather_next(poly, count)
    length = poly.shape[1]
    active = _arange(length, poly)[None, :] < count[:, None]
    cross = poly[..., 0] * p_next[..., 1] - poly[..., 1] * p_next[..., 0]
    return 0.5 * torch.abs(torch.sum(torch.where(active, cross, torch.zeros_like(cross)), dim=1))


def rect_intersection_area(block_pose: torch.Tensor, block_center, block_half,
                           goal_pose: torch.Tensor, goal_center, goal_half) -> torch.Tensor:
    """Exact area of two oriented rectangles' intersection -> (N,)."""
    corners = rect_corners(block_pose, block_center, block_half)

    goal_center_t = _t(goal_center, goal_pose)
    goal_world_center = goal_pose[..., None, :2] + rotate(goal_center_t, goal_pose[..., 2, None])
    local = rotate_inv(corners - goal_world_center, goal_pose[..., 2, None])

    n = corners.shape[0]
    poly = torch.cat((local, torch.zeros((n, 4, 2), dtype=local.dtype, device=local.device)), dim=1)
    count = torch.full((n,), 4, dtype=torch.int32, device=local.device)

    hx, hy = goal_half
    for bound, sign, axis in ((hx, 1.0, 0), (hx, -1.0, 0), (hy, 1.0, 1), (hy, -1.0, 1)):
        f = bound - sign * poly[..., axis]
        poly, count = _clip_halfplane(poly, count, f)

    return polygon_area(poly, count)


def coverage(block_pose: torch.Tensor) -> torch.Tensor:
    """Coverage of the fixed goal T by the block T -> (N,)."""
    n = block_pose.shape[0]
    goal_pose = _t(GOAL_POSE, block_pose).expand(n, 3)
    total = torch.zeros((n,), dtype=block_pose.dtype, device=block_pose.device)
    for i in range(2):
        for j in range(2):
            total = total + rect_intersection_area(
                block_pose, BLOCK_CENTERS[i], BLOCK_HALFS[i],
                goal_pose, BLOCK_CENTERS[j], BLOCK_HALFS[j],
            )
    return torch.clamp(total / BLOCK_AREA, 0.0, 1.0)


def keypoints(block_pose: torch.Tensor) -> torch.Tensor:
    """World T keypoints, same ordering as gym-pusht -> (N, 8, 2)."""
    corners = _t(BLOCK_CORNERS, block_pose)
    return block_pose[..., None, :2] + rotate(corners, block_pose[..., 2, None])


def obb_wall_contacts_batched(centers, u, v, half, wall_centers, wall_halfs):
    """All 2 rect x 4 wall manifolds at once -> (N, 16, ...)."""
    n = centers.shape[0]
    c = centers[:, :, None, :]                       # N,2,1,2
    u_b = u[:, None, None, :]                        # N,1,1,2
    v_b = v[:, None, None, :]
    u_ax = u[:, None, None, None, :]                  # N,1,1,1,2
    v_ax = v[:, None, None, None, :]
    ex = _t((1.0, 0.0), u)[None, None, None, :].expand_as(u_b)
    ey = _t((0.0, 1.0), u)[None, None, None, :].expand_as(u_b)
    axes = torch.stack((u_b, v_b, ex, ey), dim=-2)   # N,1,1,4,2
    wc = wall_centers[None, None, :, :]              # 1,1,4,2
    wh = wall_halfs[None, None, :, :]
    hx = half[:, 0][None, :, None, None]
    hy = half[:, 1][None, :, None, None]

    rb = hx * torch.abs(torch.sum(u_ax * axes, dim=-1)) + \
        hy * torch.abs(torch.sum(v_ax * axes, dim=-1))
    rw = wh[..., 0, None] * torch.abs(axes[..., 0]) + wh[..., 1, None] * torch.abs(axes[..., 1])
    d = c - wc
    dist = torch.abs(torch.sum(d[:, :, :, None, :] * axes, dim=-1))
    overlap = rb + rw - dist

    colliding = torch.all(overlap > 0.0, dim=-1)     # N,2,4
    big = torch.full_like(overlap, 1e9)
    idx = torch.argmin(torch.where(colliding[..., None], overlap, big), dim=-1)
    onehot = (idx[..., None] == _arange(4, u)[None, None, None, :]).to(axes.dtype)
    axis = torch.sum(axes * onehot[..., None], dim=-2)  # N,2,4,2
    ov = torch.sum(overlap * onehot, dim=-1)            # N,2,4
    side = torch.sum(d * axis, dim=-1)
    n = axis * torch.where(side >= 0.0, torch.ones_like(side), -torch.ones_like(side))[..., None]

    du = torch.sum(u_b * n, dim=-1)
    dv = torch.sum(v_b * n, dim=-1)
    use_u = torch.abs(du) >= torch.abs(dv)
    face_u = -torch.where(du >= 0.0, torch.ones_like(du), -torch.ones_like(du))[..., None] * u_b
    face_v = -torch.where(dv >= 0.0, torch.ones_like(dv), -torch.ones_like(dv))[..., None] * v_b

    p1_u = c + face_u * hx - v_b * hy
    p2_u = c + face_u * hx + v_b * hy
    p1_v = c + face_v * hy - u_b * hx
    p2_v = c + face_v * hy + u_b * hx
    p1 = torch.where(use_u[..., None], p1_u, p1_v)
    p2 = torch.where(use_u[..., None], p2_u, p2_v)
    point = torch.stack((p1, p2), dim=-2)              # N,2,4,2,2

    rw_n = wh[..., 0] * torch.abs(n[..., 0]) + wh[..., 1] * torch.abs(n[..., 1])
    plane = torch.sum(wc * n, dim=-1) + rw_n
    pen1 = torch.clamp_min(plane - torch.sum(p1 * n, dim=-1), 0.0)
    pen2 = torch.clamp_min(plane - torch.sum(p2 * n, dim=-1), 0.0)
    pen = torch.stack((pen1, pen2), dim=-1)            # N,2,4,2
    valid = colliding[..., None] & (pen > 0.0)
    n = n[..., None, :].expand_as(point)

    axis_id = torch.where(use_u, torch.where(du >= 0.0, 0, 1), torch.where(dv >= 0.0, 2, 3))
    base = (_arange(2, u)[None, :, None] * 4 + _arange(4, u)[None, None, :]) * 4 + axis_id
    feature = (base[..., None] * 2 + _arange(2, u)[None, None, None, :]).to(torch.int64)

    m = n.shape[0]
    return (n.reshape(m, 16, 2), point.reshape(m, 16, 2), pen.reshape(m, 16),
            valid.reshape(m, 16), feature.reshape(m, 16))


def circle_obb_contact(agent_pos: torch.Tensor, agent_vel: torch.Tensor,
                       center: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                       half, radius: float, rect_idx: int = 0):
    """One-point manifold between the pusher circle and one OBB."""
    d = agent_pos - center
    dx = torch.sum(d * u, dim=-1)
    dy = torch.sum(d * v, dim=-1)
    hx, hy = _t(half, center)

    qx = torch.clamp(dx, -hx, hx)
    qy = torch.clamp(dy, -hy, hy)
    delta = torch.stack((dx - qx, dy - qy), dim=-1)
    dist = torch.sqrt(torch.clamp(torch.sum(delta * delta, dim=-1), min=1e-12))

    outside = dist > 1e-4
    delta_world = delta[..., 0, None] * u + delta[..., 1, None] * v
    n_out = -delta_world / dist[..., None]
    point_out = center + qx[..., None] * u + qy[..., None] * v
    pen_out = radius - dist

    gap_x = hx - torch.abs(dx)
    gap_y = hy - torch.abs(dy)
    use_x = gap_x <= gap_y
    sx = torch.where(dx >= 0.0, torch.ones_like(dx), -torch.ones_like(dx))
    sy = torch.where(dy >= 0.0, torch.ones_like(dy), -torch.ones_like(dy))
    n_in = -torch.where(use_x[:, None], sx[:, None] * u, sy[:, None] * v)
    point_in = center + torch.where(use_x[:, None], sx[:, None] * hx * u, qx[:, None] * u) + \
        torch.where(use_x[:, None], qy[:, None] * v, sy[:, None] * hy * v)
    pen_in = radius + torch.where(use_x, gap_x, gap_y)

    n = torch.where(outside[:, None], n_out, n_in)
    point = torch.where(outside[:, None], point_out, point_in)
    pen = torch.where(outside, pen_out, pen_in)
    valid = pen > 0.0
    v_other = agent_vel
    fx = torch.where(dx < -hx, 0, torch.where(dx > hx, 2, 1))
    fy = torch.where(dy < -hy, 0, torch.where(dy > hy, 2, 1))
    feature = (rect_idx * 9 + fx * 3 + fy).to(torch.int64)
    return n, point, pen, valid, v_other, feature


def build_contacts(agent_pos: torch.Tensor, agent_vel: torch.Tensor,
                   block_pose: torch.Tensor, walls_first: bool = True):
    """Assemble the fixed contact array for the whole batch."""
    theta = block_pose[:, 2]
    p = block_pose[:, :2]
    centers = _t(BLOCK_CENTERS, p)
    c = p[:, None, :] + rotate(centers, theta[:, None])
    ex = _t((1.0, 0.0), p).expand_as(p)
    ey = _t((0.0, 1.0), p).expand_as(p)
    u = rotate(ex, theta)
    v = rotate(ey, theta)

    normals, points, pens, valids, others, features = [], [], [], [], [], []

    def add_walls():
        n, pt, pen, valid, feat = obb_wall_contacts_batched(
            c, u, v, _t(BLOCK_HALFS, p), _t(WALL_CENTERS, p), _t(WALL_HALFS, p))
        normals.append(n)
        points.append(pt)
        pens.append(pen)
        valids.append(valid)
        others.append(torch.zeros_like(pt))
        features.append(feat)

    def add_circles():
        for i in range(2):
            n, pt, pen, valid, vo, feat = circle_obb_contact(
                agent_pos, agent_vel, c[:, i], u, v, BLOCK_HALFS[i], AGENT_RADIUS,
                rect_idx=i)
            normals.append(n[:, None, :])
            points.append(pt[:, None, :])
            pens.append(pen[:, None])
            valids.append(valid[:, None])
            others.append(vo[:, None, :])
            features.append(feat[:, None])

    if walls_first:
        add_walls()
        add_circles()
    else:
        add_circles()
        add_walls()

    normal = torch.cat(normals, dim=1)
    point = torch.cat(points, dim=1)
    pen = torch.cat(pens, dim=1)
    valid = torch.cat(valids, dim=1)
    v_other = torch.cat(others, dim=1)
    feature = torch.cat(features, dim=1)
    return normal, point, pen, valid, v_other, feature

# ---------------------------------------------------------------- rasterizer
WHITE = (255.0, 255.0, 255.0)
GOAL_GREEN = (144.0, 238.0, 144.0)
WALL_GRAY = (211.0, 211.0, 211.0)
BLOCK_FILL = (142.8, 163.2, 183.6)
BLOCK_EDGE = (119.0, 136.0, 153.0)
AGENT_FILL = (78.0, 126.0, 255.0)
AGENT_INNER = (93.6, 151.2, 255.0)


def _coords(height: int, width: int, scale: int, device):
    ys = (torch.arange(height * scale, device=device, dtype=torch.float32) + 0.5) * (512.0 / (height * scale))
    xs = (torch.arange(width * scale, device=device, dtype=torch.float32) + 0.5) * (512.0 / (width * scale))
    return xs[None, :], ys[:, None]


def _rect_mask(xs, ys, pose, center, half):
    """Point-in-OBB mask -> (N, H, W)."""
    center_t = _t(center, pose)
    half_t = _t(half, pose)
    theta = pose[:, 2, None, None]
    c = pose[:, None, None, :2] + torch.stack(
        (torch.cos(theta) * center_t[0] - torch.sin(theta) * center_t[1],
         torch.sin(theta) * center_t[0] + torch.cos(theta) * center_t[1]), dim=-1)
    dx = xs[None, :, :] - c[..., 0]
    dy = ys[None, :, :] - c[..., 1]
    lx = torch.cos(theta) * dx + torch.sin(theta) * dy
    ly = -torch.sin(theta) * dx + torch.cos(theta) * dy
    return (torch.abs(lx) <= half_t[0]) & (torch.abs(ly) <= half_t[1]), lx, ly


def _segment_mask(xs, ys, a, b, radius: float):
    ax, ay = a
    bx, by = b
    px = xs - ax
    py = ys - ay
    dx = bx - ax
    dy = by - ay
    t = torch.clamp((px * dx + py * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    qx = px - t * dx
    qy = py - t * dy
    return qx * qx + qy * qy <= radius * radius


def render(agent_pos: torch.Tensor, block_pose: torch.Tensor, height: int = 96,
           width: int = 96, antialias: int = 1) -> torch.Tensor:
    """Render a batch of states -> (N, H, W, 3) uint8."""
    device = agent_pos.device
    n = agent_pos.shape[0]
    scale = max(1, int(antialias))
    xs, ys = _coords(height, width, scale, device)
    h, w = height * scale, width * scale

    img = _t(WHITE, agent_pos).expand(n, h, w, 3).clone()
    goal = _t(GOAL_POSE, agent_pos).expand(n, 3)
    goal_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    for i in range(2):
        mask, _, _ = _rect_mask(xs, ys, goal, BLOCK_CENTERS[i], BLOCK_HALFS[i])
        goal_mask = goal_mask | mask[0]
    img = torch.where(goal_mask[None, :, :, None], _t(GOAL_GREEN, img), img)

    wall_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    for a, b in (
        ((5.0, 506.0), (5.0, 5.0)),
        ((5.0, 5.0), (506.0, 5.0)),
        ((506.0, 5.0), (506.0, 506.0)),
        ((5.0, 506.0), (506.0, 506.0)),
    ):
        wall_mask = wall_mask | _segment_mask(xs, ys, a, b, 2.0)
    img = torch.where(wall_mask[None, :, :, None], _t(WALL_GRAY, img), img)

    for i in range(2):
        mask, lx, ly = _rect_mask(xs, ys, block_pose, BLOCK_CENTERS[i], BLOCK_HALFS[i])
        half = _t(BLOCK_HALFS[i], block_pose)
        edge = mask & ((half[0] - torch.abs(lx) <= 2.0) | (half[1] - torch.abs(ly) <= 2.0))
        img = torch.where(mask[..., None], _t(BLOCK_FILL, img), img)
        img = torch.where(edge[..., None], _t(BLOCK_EDGE, img), img)

    ap = agent_pos.float()
    ax = ap[:, None, None, 0] - xs[None, :, :]
    ay = ap[:, None, None, 1] - ys[None, :, :]
    d2 = ax * ax + ay * ay
    img = torch.where((d2 <= 15.0 ** 2)[..., None], _t(AGENT_FILL, img), img)
    img = torch.where((d2 <= 11.0 ** 2)[..., None], _t(AGENT_INNER, img), img)

    if scale > 1:
        img = img.reshape(n, height, scale, width, scale, 3).mean(dim=(2, 4))

    return torch.clamp(img, 0.0, 255.0).to(torch.uint8)

# ---------------------------------------------------------------- environment

DT = 0.01
CONTROL_HZ = 10
N_SUBSTEPS = int(round(1.0 / (DT * CONTROL_HZ)))
# Chipmunk default: pow(1 - 0.1, 60) -> 10% correction per second at 60 Hz
COLLISION_BIAS = (1.0 - 0.1) ** 60.0
COLLISION_SLOP = 0.1
BIAS_COEF = 1.0 - COLLISION_BIAS ** DT

OBS_TYPES = ('state', 'environment_state_agent_pos', 'pixels', 'pixels_agent_pos')


class PushTEnv:
    """Vectorized Push-T environment running on torch tensors.

    Parameters
    ----------
    num_envs:
        Number of independent environments in the batch.
    obs_type:
        ``state`` -> (N, 5), ``environment_state_agent_pos`` -> keypoints,
        ``pixels`` -> (N, H, W, 3) uint8, ``pixels_agent_pos`` -> both.
    max_episode_steps:
        Truncation horizon, matching gym-pusht's 300.
    auto_reset:
        Reset finished environments inside ``step`` (gym-vector style).
    solver:
        ``sequential`` runs Chipmunk-style Gauss-Seidel sweeps (best parity);
        ``jacobi`` relaxes all contacts at once (faster, tiny extra drift in
        rare multi-contact configurations).
    solver_iterations:
        Contact solver sweeps per 10 ms physics substep.
    device:
        Torch device. Defaults to Apple ``mps`` when available, else CUDA/CPU.
    dtype:
        ``torch.float32`` for normal use; ``torch.float64`` on CPU gives
        near bit-exact parity with pymunk's double-precision solver.
    """

    def __init__(
        self,
        num_envs: int = 1,
        obs_type: str = 'state',
        max_episode_steps: int = 300,
        observation_width: int = 96,
        observation_height: int = 96,
        observation_antialias: int = 2,
        auto_reset: bool = False,
        solver_iterations: int = 10,
        solver: str = 'sequential',
        relaxation: float = 0.5,
        warm_start: bool = True,
        cache_persistence: int = 3,
        contact_order: str = 'walls',
        seed: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if obs_type not in OBS_TYPES:
            raise ValueError(f'obs_type must be one of {OBS_TYPES}, got {obs_type!r}')

        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'mps' and dtype == torch.float64:
            raise ValueError('float64 is not supported on MPS; use device="cpu" for parity runs')

        self.dtype = dtype
        self._np_dtype = np.float64 if dtype == torch.float64 else np.float32
        self.num_envs = num_envs
        self.obs_type = obs_type
        self.max_episode_steps = max_episode_steps
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.observation_antialias = observation_antialias
        self.auto_reset = auto_reset
        self.solver_iterations = solver_iterations
        if solver not in ('jacobi', 'sequential'):
            raise ValueError("solver must be 'jacobi' or 'sequential'")
        self.solver = solver
        self.relaxation = relaxation
        self.warm_start = warm_start
        self.cache_persistence = cache_persistence
        if contact_order not in ('walls', 'circles'):
            raise ValueError("contact_order must be 'walls' or 'circles'")
        self.contact_order = contact_order

        self.k_p = 100.0
        self.k_v = 20.0

        self._rng = np.random.default_rng(seed)
        self.seed_value = seed

        z = lambda *shape: torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.t = torch.zeros((num_envs,), dtype=torch.int64, device=self.device)
        self.agent_pos = z(num_envs, 2)
        self.agent_vel = z(num_envs, 2)
        self.block_p = z(num_envs, 2)
        self.block_theta = z(num_envs)
        self.block_v = z(num_envs, 2)
        self.block_w = z(num_envs)
        self.v_bias = z(num_envs, 2)
        self.w_bias = z(num_envs)
        self.cached_jn = z(num_envs, N_CONTACTS)
        self.cache_age = torch.full((num_envs, N_CONTACTS), cache_persistence,
                                    dtype=torch.int64, device=self.device)
        self.cached_feature = torch.full((num_envs, N_CONTACTS), -1,
                                         dtype=torch.int64, device=self.device)
        self.n_contacts = torch.zeros((num_envs,), dtype=torch.int64, device=self.device)

        self.reset(seed=seed)

    # ---------------------------------------------------------------- state

    @property
    def block_pose(self) -> torch.Tensor:
        return torch.cat((self.block_p, self.block_theta[:, None]), dim=1)

    @property
    def state(self) -> torch.Tensor:
        angle = self.block_theta % (2.0 * math.pi)
        return torch.cat((self.agent_pos, self.block_p, angle[:, None]), dim=1)

    def _cog(self) -> torch.Tensor:
        return self.block_p + rotate(self._const(BLOCK_COG).expand_as(self.block_p), self.block_theta)

    def _const(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=self.dtype, device=self.device)

    # ---------------------------------------------------------------- reset

    def _sample(self, n: int):
        agent = self._rng.integers(50, 450, size=(n, 2)).astype(self._np_dtype)
        block = self._rng.integers(100, 400, size=(n, 2)).astype(self._np_dtype)
        angle = self._rng.uniform(-math.pi, math.pi, size=(n,)).astype(self._np_dtype)
        return agent, block, angle

    def _place_block(self, block: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        # gym-pusht sets position first and angle second; the body starts at 0.
        cog = self._const(BLOCK_COG).expand_as(block)
        return block + cog - rotate(cog, angle)

    @torch.no_grad()
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.seed_value = seed

        n = self.num_envs
        if options is not None and options.get('reset_to_state') is not None:
            state = np.asarray(options['reset_to_state'], dtype=self._np_dtype)
            if state.ndim == 1:
                state = np.broadcast_to(state, (n, state.shape[0]))
            if state.shape != (n, 5):
                raise ValueError(f'reset_to_state must have shape (5,) or ({n}, 5), got {state.shape}')
            agent = state[:, :2]
            block = state[:, 2:4]
            angle = state[:, 4]
        else:
            agent, block, angle = self._sample(n)

        block = self._place_block(torch.tensor(block, dtype=self.dtype, device=self.device),
                                  torch.tensor(angle, dtype=self.dtype, device=self.device))

        self.t = torch.zeros((n,), dtype=torch.int64, device=self.device)
        self.agent_pos = torch.tensor(agent, dtype=self.dtype, device=self.device)
        self.agent_vel = torch.zeros_like(self.agent_pos)
        self.block_p = block
        self.block_theta = torch.tensor(angle, dtype=self.dtype, device=self.device)
        self.block_v = torch.zeros_like(self.block_p)
        self.block_w = torch.zeros_like(self.block_theta)
        self.v_bias = torch.zeros_like(self.block_v)
        self.w_bias = torch.zeros_like(self.block_w)
        self.cached_jn = torch.zeros((n, N_CONTACTS), dtype=self.dtype, device=self.device)
        self.cache_age = torch.full((n, N_CONTACTS), self.cache_persistence,
                                    dtype=torch.int64, device=self.device)
        self.cached_feature = torch.full((n, N_CONTACTS), -1,
                                         dtype=torch.int64, device=self.device)
        self.n_contacts = torch.zeros((n,), dtype=torch.int64, device=self.device)

        # gym-pusht calls space.step(dt) inside _set_state: positions do not
        # move (zero velocity) but bias impulses are stored for the first step.
        self._prestep()

        return self._obs(), self._info(coverage=self._coverage())

    @torch.no_grad()
    def reset_slots(self, indices, states):
        """Reset selected environments to pre-quirk ``reset_to_state`` rows."""
        if len(indices) == 0:
            return

        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        st = torch.as_tensor(np.stack(states), dtype=self.dtype, device=self.device)
        agent, block, angle = st[:, :2], st[:, 2:4], st[:, 4]
        block_p = self._place_block(block, angle)

        self.agent_pos[idx] = agent
        self.agent_vel[idx] = 0.
        self.block_p[idx] = block_p
        self.block_theta[idx] = angle
        self.block_v[idx] = 0.
        self.block_w[idx] = 0.
        self.v_bias[idx] = 0.
        self.w_bias[idx] = 0.
        self.cached_jn[idx] = 0.
        self.cache_age[idx] = self.cache_persistence
        self.cached_feature[idx] = -1
        self.t[idx] = 0

        normal, point, pen, valid, v_other, feature = build_contacts(
            self.agent_pos, self.agent_vel, self.block_pose,
            walls_first=self.contact_order == 'walls')
        v, w, vb, wb, jn = self._solve(
            self.block_v, self.block_w, self.v_bias, self.w_bias, self._cog(),
            normal, point, pen, valid, v_other, feature, warm_start=False)
        new_cached, new_age, new_feature = self._cache_step(valid, jn, feature)
        self.v_bias[idx] = vb[idx]
        self.w_bias[idx] = wb[idx]
        self.cached_jn[idx] = new_cached[idx]
        self.cache_age[idx] = new_age[idx]
        self.cached_feature[idx] = new_feature[idx]

    def _prestep(self):
        normal, point, pen, valid, v_other, feature = build_contacts(
            self.agent_pos, self.agent_vel, self.block_pose,
            walls_first=self.contact_order == 'walls')
        v, w, vb, wb, jn = self._solve(
            self.block_v, self.block_w, self.v_bias, self.w_bias, self._cog(),
            normal, point, pen, valid, v_other, feature, warm_start=False)
        self.v_bias, self.w_bias = vb, wb
        self.cached_jn, self.cache_age, self.cached_feature = self._cache_step(valid, jn, feature)

    def _cache_step(self, valid, jn, feature):
        cached = torch.where(valid, jn, self.cached_jn)
        age = torch.where(valid, torch.zeros_like(self.cache_age), self.cache_age + 1)
        age = torch.clamp(age, max=self.cache_persistence)
        cached = torch.where(age >= self.cache_persistence,
                             torch.zeros_like(cached), cached)
        cached_feature = torch.where(valid, feature, self.cached_feature)
        return cached, age, cached_feature

    # ---------------------------------------------------------------- physics

    def _solve(self, v, w, v_bias, w_bias, cog, normal, point, pen, valid,
               v_other, feature, warm_start: bool = True):
        r = point - cog[:, None, :]
        rn = r[..., 0] * normal[..., 1] - r[..., 1] * normal[..., 0]
        n_mass = 1.0 / (1.0 / BLOCK_MASS + rn * rn / BLOCK_INERTIA)
        bias = BIAS_COEF / DT * torch.clamp_min(pen - COLLISION_SLOP, 0.0)
        perp = torch.stack((-r[..., 1], r[..., 0]), dim=-1)

        # GPU Jacobi solves the whole contact system as a small dense LCP.
        # Precompute the contact mass matrix and pre-cache normal velocities.
        use_matrix = self.solver == 'jacobi' and self.device.type in ('mps', 'cuda')
        if use_matrix:
            A = torch.einsum('nci,ndi->ncd', normal, normal) / BLOCK_MASS + \
                rn[:, :, None] * rn[:, None, :] / BLOCK_INERTIA
            diag = torch.diagonal(A, dim1=1, dim2=2)
            diag_safe = torch.where(valid, diag, torch.ones_like(diag))
            b_kin = torch.sum((v[:, None, :] + w[:, None, None] * perp - v_other) * normal, dim=-1)
            b_bias = torch.sum((v_bias[:, None, :] + w_bias[:, None, None] * perp) * normal, dim=-1)

        if warm_start:
            # same contact feature, within Chipmunk's persistence window
            same = valid & (self.cache_age < self.cache_persistence) & \
                (feature == self.cached_feature)
            # continuous contact: cached impulse is applied to velocity.
            # reactivated contact: the stale accumulator is kept but is *not*
            # applied to velocity (Chipmunk first-contact behaviour).
            continuous = same & (self.cache_age == 0)
            reactivated = same & (self.cache_age > 0)
            accum = torch.where(same, self.cached_jn, torch.zeros_like(self.cached_jn))
            apply = torch.where(continuous, self.cached_jn, torch.zeros_like(self.cached_jn))
        else:
            accum = torch.zeros_like(self.cached_jn)
            apply = torch.zeros_like(self.cached_jn)

        v = v + torch.sum(apply[..., None] * normal, dim=1) / BLOCK_MASS
        w = w + torch.sum(apply * rn, dim=1) / BLOCK_INERTIA

        if use_matrix:
            # Kinetics and bias share A; solve both as a (N, 2, C) block.
            # b_eff accounts for the cached impulse already applied to v/w.
            b_eff = b_kin + torch.bmm(A, (apply - accum)[..., None])[..., 0]
            rhs = torch.stack((b_eff, b_bias - bias), dim=1)
            jn_all = torch.stack((accum, torch.zeros_like(accum)), dim=1)
            valid_b = valid[:, None, :]
            for _ in range(self.solver_iterations):
                vn = rhs + torch.bmm(A, jn_all.transpose(1, 2)).transpose(1, 2)
                jn_all = torch.clamp_min(
                    jn_all - self.relaxation * vn / diag_safe[:, None, :], 0.0)
                jn_all = torch.where(valid_b, jn_all, torch.zeros_like(jn_all))

            jn, jb = jn_all[:, 0], jn_all[:, 1]
            jn_delta = jn - accum
            v = v + torch.einsum('nc,nci->ni', jn_delta, normal) / BLOCK_MASS
            w = w + torch.sum(jn_delta * rn, dim=1) / BLOCK_INERTIA
            v_bias = v_bias + torch.einsum('nc,nci->ni', jb, normal) / BLOCK_MASS
            w_bias = w_bias + torch.sum(jb * rn, dim=1) / BLOCK_INERTIA
        elif self.solver == 'jacobi':
            # CPU: all-contact Jacobi avoids batched small matmuls.
            jn = accum
            jb = torch.zeros_like(pen)
            for _ in range(self.solver_iterations):
                vb = v_bias[:, None, :] + w_bias[:, None, None] * perp
                vbn = torch.sum(vb * normal, dim=-1)
                jb_new = torch.clamp_min(jb + self.relaxation * (bias - vbn) * n_mass, 0.0)
                jb_delta = torch.where(valid, jb_new - jb, torch.zeros_like(jb))
                v_bias = v_bias + torch.sum(jb_delta[..., None] * normal, dim=1) / BLOCK_MASS
                w_bias = w_bias + torch.sum(jb_delta * rn, dim=1) / BLOCK_INERTIA
                jb = torch.where(valid, jb_new, jb)

                vr = (v[:, None, :] + w[:, None, None] * perp) - v_other
                vrn = torch.sum(vr * normal, dim=-1)
                jn_new = torch.clamp_min(jn + self.relaxation * (-vrn) * n_mass, 0.0)
                jn_delta = torch.where(valid, jn_new - jn, torch.zeros_like(jn))
                v = v + torch.sum(jn_delta[..., None] * normal, dim=1) / BLOCK_MASS
                w = w + torch.sum(jn_delta * rn, dim=1) / BLOCK_INERTIA
                jn = torch.where(valid, jn_new, jn)
        else:
            jn = accum
            jb = torch.zeros_like(pen)
            for _ in range(self.solver_iterations):
                # Gauss-Seidel impulses, one contact slot at a time
                for c in range(N_CONTACTS):
                    n_c = normal[:, c]
                    v_other_c = v_other[:, c]
                    valid_c = valid[:, c]

                    vb = v_bias + w_bias[:, None] * perp[:, c]
                    vbn = torch.sum(vb * n_c, dim=-1)
                    jb_new = torch.clamp_min(jb[:, c] + (bias[:, c] - vbn) * n_mass[:, c], 0.0)
                    jb_delta = torch.where(valid_c, jb_new - jb[:, c], torch.zeros_like(jb_new))
                    v_bias = v_bias + jb_delta[:, None] * n_c / BLOCK_MASS
                    w_bias = w_bias + jb_delta * rn[:, c] / BLOCK_INERTIA
                    jb[:, c] = torch.where(valid_c, jb_new, jb[:, c])

                    vr = (v + w[:, None] * perp[:, c]) - v_other_c
                    vrn = torch.sum(vr * n_c, dim=-1)
                    jn_new = torch.clamp_min(jn[:, c] + (-vrn) * n_mass[:, c], 0.0)
                    jn_delta = torch.where(valid_c, jn_new - jn[:, c], torch.zeros_like(jn_new))
                    v = v + jn_delta[:, None] * n_c / BLOCK_MASS
                    w = w + jn_delta * rn[:, c] / BLOCK_INERTIA
                    jn[:, c] = torch.where(valid_c, jn_new, jn[:, c])

        jn = torch.where(valid, jn, self.cached_jn)
        return v, w, v_bias, w_bias, jn

    def _substep(self, action: torch.Tensor):
        acc = self.k_p * (action - self.agent_pos) - self.k_v * self.agent_vel
        self.agent_vel = self.agent_vel + acc * DT
        self.agent_pos = self.agent_pos + self.agent_vel * DT

        # Chipmunk integrates the CoG position; the observable body origin is
        # p = cog_world - R(theta) * local_cog.
        cog_world = self.block_p + rotate(self._const(BLOCK_COG).expand_as(self.block_p), self.block_theta)
        cog_world = cog_world + (self.block_v + self.v_bias) * DT
        theta = self.block_theta + (self.block_w + self.w_bias) * DT
        self.block_p = cog_world - rotate(self._const(BLOCK_COG).expand_as(cog_world), theta)
        self.block_theta = theta
        self.v_bias = torch.zeros_like(self.v_bias)
        self.w_bias = torch.zeros_like(self.w_bias)

        normal, point, pen, valid, v_other, feature = build_contacts(
            self.agent_pos, self.agent_vel, self.block_pose,
            walls_first=self.contact_order == 'walls')
        self.n_contacts = self.n_contacts + torch.sum(valid.to(torch.int64), dim=1)

        # space.damping = 0 in gym-pusht: Chipmunk zeroes body velocity after
        # position integration and before cached impulses / the solver.
        zero_v = torch.zeros_like(self.block_v)
        zero_w = torch.zeros_like(self.block_w)
        self.block_v, self.block_w, self.v_bias, self.w_bias, jn = self._solve(
            zero_v, zero_w, self.v_bias, self.w_bias, cog_world,
            normal, point, pen, valid, v_other, feature, warm_start=self.warm_start)
        self.cached_jn, self.cache_age, self.cached_feature = self._cache_step(valid, jn, feature)

    @torch.no_grad()
    def step(self, actions):
        actions = self._as_action(actions)

        self.n_contacts = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        for _ in range(N_SUBSTEPS):
            self._substep(actions)

        self.t = self.t + 1
        cov = self._coverage()
        reward = torch.clamp(cov / 0.95, 0.0, 1.0)
        terminated = cov > 0.95
        truncated = self.t >= self.max_episode_steps

        info = self._info(coverage=cov, terminated=terminated)
        info['is_success'] = terminated
        info['n_contacts'] = torch.ceil(self.n_contacts / N_SUBSTEPS).to(torch.int64)

        final_obs = None
        if self.auto_reset:
            done = terminated | truncated
            if bool(done.any()):
                final_obs = self._obs()
                self._reset_done(done)
        obs = self._obs()

        if final_obs is not None:
            info['final_observation'] = final_obs
        return obs, reward, terminated, truncated, info

    @torch.no_grad()
    def _reset_done(self, done: torch.Tensor):
        n = self.num_envs
        agent, block, angle = self._sample(n)
        block = self._place_block(torch.tensor(block, dtype=self.dtype, device=self.device),
                                  torch.tensor(angle, dtype=self.dtype, device=self.device))
        angle = torch.tensor(angle, dtype=self.dtype, device=self.device)

        m = done[:, None]
        self.agent_pos = torch.where(m, torch.tensor(agent, dtype=self.dtype, device=self.device), self.agent_pos)
        self.block_p = torch.where(m, block, self.block_p)
        self.block_theta = torch.where(done, angle, self.block_theta)
        self.agent_vel = torch.where(m, torch.zeros_like(self.agent_vel), self.agent_vel)
        self.block_v = torch.where(m, torch.zeros_like(self.block_v), self.block_v)
        self.block_w = torch.where(done, torch.zeros_like(self.block_w), self.block_w)
        self.v_bias = torch.where(m, torch.zeros_like(self.v_bias), self.v_bias)
        self.w_bias = torch.where(done, torch.zeros_like(self.w_bias), self.w_bias)
        self.cached_jn = torch.where(m, torch.zeros_like(self.cached_jn), self.cached_jn)
        self.cache_age = torch.where(m, torch.full_like(self.cache_age, self.cache_persistence),
                                     self.cache_age)
        self.cached_feature = torch.where(m, torch.full_like(self.cached_feature, -1),
                                          self.cached_feature)
        self.t = torch.where(done, torch.zeros_like(self.t), self.t)

        normal, point, pen, valid, v_other, feature = build_contacts(
            self.agent_pos, self.agent_vel, self.block_pose,
            walls_first=self.contact_order == 'walls')
        v, w, vb, wb, jn = self._solve(
            self.block_v, self.block_w, self.v_bias, self.w_bias, self._cog(),
            normal, point, pen, valid, v_other, feature, warm_start=False)
        new_cached, new_age, new_feature = self._cache_step(valid, jn, feature)
        self.v_bias = torch.where(m, vb, self.v_bias)
        self.w_bias = torch.where(done, wb, self.w_bias)
        self.cached_jn = torch.where(m, new_cached, self.cached_jn)
        self.cache_age = torch.where(m, new_age, self.cache_age)
        self.cached_feature = torch.where(m, new_feature, self.cached_feature)

    # ---------------------------------------------------------------- output

    def _coverage(self) -> torch.Tensor:
        return coverage(self.block_pose)

    def _obs(self):
        if self.obs_type == 'state':
            return self.state
        if self.obs_type == 'environment_state_agent_pos':
            return dict(environment_state=keypoints(self.block_pose).reshape(self.num_envs, 16),
                        agent_pos=self.agent_pos)
        if self.obs_type == 'pixels':
            return render(self.agent_pos, self.block_pose,
                          self.observation_height, self.observation_width,
                          antialias=self.observation_antialias)
        return dict(
            pixels=render(self.agent_pos, self.block_pose,
                          self.observation_height, self.observation_width,
                          antialias=self.observation_antialias),
            agent_pos=self.agent_pos,
        )

    def _info(self, coverage=None, terminated=None):
        cov = self._coverage() if coverage is None else coverage
        return dict(
            coverage=cov,
            block_pose=self.block_pose,
            goal_pose=self._const(GOAL_POSE).expand(self.num_envs, 3),
            agent_vel=self.agent_vel,
            is_success=cov > 0.95 if terminated is None else terminated,
        )

    @torch.no_grad()
    def render(self, height: int | None = None, width: int | None = None,
               antialias: int | None = None) -> torch.Tensor:
        return render(self.agent_pos, self.block_pose,
                      self.observation_height if height is None else height,
                      self.observation_width if width is None else width,
                      antialias=self.observation_antialias if antialias is None else antialias)

    # ---------------------------------------------------------------- utils

    def _as_action(self, actions) -> torch.Tensor:
        actions = torch.as_tensor(actions, dtype=self.dtype, device=self.device)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape != (self.num_envs, 2):
            raise ValueError(f'expected actions of shape ({self.num_envs}, 2), got {tuple(actions.shape)}')
        return torch.clamp(actions, 0.0, 512.0)

    def sample_actions(self) -> torch.Tensor:
        return torch.as_tensor(
            self._rng.uniform(0.0, 512.0, size=(self.num_envs, 2)).astype(self._np_dtype),
            dtype=self.dtype, device=self.device,
        )

    def close(self):
        pass
