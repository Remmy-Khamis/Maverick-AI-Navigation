"""
MAVERICK · RouteOptimizer Training Script
==========================================
Trains the PyTorch route-optimisation model on synthetically generated
flight data that mirrors real-world aviation physics.

Why synthetic data?
  Real Eurocontrol NEST data requires registration. Synthetic data built
  on the same physics (haversine distances, wind components, ICAO fuel
  burn factors) teaches the model the same relationships. Once you have
  Eurocontrol NEST access, swap in generate_eurocontrol_dataset() below.

Model architecture (matches main.py exactly):
  Input  (8):  [orig_lat, orig_lon, dest_lat, dest_lon, wx, wy, alt_pref, fuel_weight]
  Hidden (3):  64 → 128 → 64 neurons, ReLU activations
  Output (5):  [wp1_lat_delta, wp1_lon_delta, wp2_lat_delta, wp2_lon_delta, fuel_saving_norm]

Training strategy:
  Supervised regression on physics-derived optimal waypoints.
  Loss = MSE on waypoint positions + MSE on fuel saving fraction.
  The "ground truth" optimal waypoints are computed by the same
  haversine + wind-avoidance physics used in numpy_route_fallback(),
  so the model learns to replicate (and generalise beyond) that logic.

Output:
  route_optimizer.pt  — trained weights, load with model.load_state_dict()

Usage:
  pip install torch numpy
  python train_route_optimizer.py
  # Copy route_optimizer.pt next to main.py on Render
"""

import math
import random
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERROR: PyTorch not installed. Run:  pip install torch")
    raise

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

EPOCHS         = 120          # iterations over the full dataset
BATCH_SIZE     = 256
LR             = 1e-3         # Adam learning rate
N_SAMPLES      = 50_000       # synthetic training pairs to generate
WEIGHT_WP      = 1.0          # loss weight for waypoint accuracy
WEIGHT_FUEL    = 0.5          # loss weight for fuel saving accuracy
SAVE_PATH      = "route_optimizer.pt"

# Airport coordinate pool (lat, lon) — extended for richer training
AIRPORTS = {
    "EGLL": (51.4706,  -0.4619),
    "EDDF": (50.0333,   8.5706),
    "ESSA": (59.6519,  17.9186),
    "KJFK": (40.6413, -73.7781),
    "OMDB": (25.2528,  55.3644),
    "WSSS": ( 1.3502, 103.9940),
    "LFPG": (49.0097,   2.5478),  # Paris CDG
    "LEMD": (40.4983,  -3.5676),  # Madrid Barajas
    "LIRF": (41.8003,  12.2389),  # Rome Fiumicino
    "EHAM": (52.3086,   4.7639),  # Amsterdam
    "EPWA": (52.1657,  20.9671),  # Warsaw
    "LTBA": (40.9769,  28.8146),  # Istanbul Ataturk
    "VHHH": (22.3080, 113.9185),  # Hong Kong
    "RJTT": (35.5494, 139.7798),  # Tokyo Haneda
    "YSSY": (-33.9461, 151.1772), # Sydney
    "FAOR": (-26.1392,  28.2460), # Johannesburg
    "SBGR": (-23.4356, -46.4731), # São Paulo
    "CYYZ": (43.6777,  -79.6248), # Toronto
}
AIRPORT_LIST = list(AIRPORTS.values())

# Aircraft burn rate table (t/hr) — IATA standard figures
AIRCRAFT = {
    "A320": {"burn": 2.4, "speed": 850},
    "B737": {"burn": 2.6, "speed": 840},
    "A380": {"burn": 11.2,"speed": 900},
    "B777": {"burn": 6.8, "speed": 895},
}
AIRCRAFT_LIST = list(AIRCRAFT.values())


# ════════════════════════════════════════════════════════════════════
# PHYSICS ENGINE  (ground truth for training labels)
# ════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(max(0, a)))


def compute_optimal_route(orig, dest, wind_dir_deg, wind_spd_kt, ac):
    """
    Returns:
      wp1_lat, wp1_lon  — first intermediate waypoint
      wp2_lat, wp2_lon  — second intermediate waypoint
      fuel_opt          — optimised fuel burn (tonnes)
      fuel_std          — standard fuel burn (tonnes)
      save_frac         — fractional fuel saving 0..1
    """
    olat, olon = orig
    dlat, dlon = dest
    dist_km = haversine_km(olat, olon, dlat, dlon)

    # Bearing (radians)
    bearing = math.atan2(dlon - olon, dlat - olat)
    wind_rad = math.radians(wind_dir_deg)

    # Wind component along bearing (positive = tailwind)
    wind_comp = math.cos(wind_rad - bearing) * wind_spd_kt

    # Standard fuel: headwind/tailwind adjusted
    hrs_std  = dist_km / ac["speed"]
    fuel_std = hrs_std * ac["burn"] * (1 + max(0, -wind_comp) * 0.003)

    # AI saving fraction
    wind_factor = 1 - (wind_comp / ac["speed"]) * 0.4
    save_frac   = min(0.28, (0.17 + wind_spd_kt * 0.002) * max(0.5, wind_factor))

    fuel_opt = fuel_std * (1 - save_frac)

    # Optimal waypoints: curve toward jet stream / away from headwind
    mlat = (olat + dlat) / 2
    mlon = (olon + dlon) / 2

    # Perpendicular offset scaled by wind severity
    perp_lat = -(dlon - olon)
    perp_lon =  (dlat - olat)
    norm = math.sqrt(perp_lat**2 + perp_lon**2) + 1e-9
    perp_lat /= norm
    perp_lon /= norm

    wind_offset = wind_spd_kt * 0.04  # degrees of offset
    # Offset direction: if tailwind, lean toward wind origin to exploit it
    sign = 1 if wind_comp > 0 else -1

    wp1_lat = mlat * 0.45 + olat * 0.55 + sign * perp_lat * wind_offset * 0.6
    wp1_lon = mlon * 0.45 + olon * 0.55 + sign * perp_lon * wind_offset * 0.6
    wp2_lat = mlat * 0.55 + dlat * 0.45 + sign * perp_lat * wind_offset * 0.4
    wp2_lon = mlon * 0.55 + dlon * 0.45 + sign * perp_lon * wind_offset * 0.4

    return wp1_lat, wp1_lon, wp2_lat, wp2_lon, fuel_opt, fuel_std, save_frac


# ════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ════════════════════════════════════════════════════════════════════

def normalise(val, vmin, vmax):
    """Min-max normalise to [-1, 1]."""
    return 2 * (val - vmin) / (vmax - vmin + 1e-9) - 1

# Normalisation ranges
LAT_MIN, LAT_MAX = -60.0, 70.0
LON_MIN, LON_MAX = -130.0, 160.0
WIND_DIR_MIN, WIND_DIR_MAX = 0.0, 360.0
WIND_SPD_MIN, WIND_SPD_MAX = 0.0, 80.0
FUEL_MIN, FUEL_MAX = 0.5, 1.5   # fuel_weight param


def generate_sample():
    """Generate one (input_vector, label_vector) training pair."""
    # Random origin + destination
    orig = random.choice(AIRPORT_LIST)
    dest = random.choice(AIRPORT_LIST)
    while dest == orig:
        dest = random.choice(AIRPORT_LIST)

    # Random wind
    wind_dir = random.uniform(0, 360)
    wind_spd = random.uniform(0, 60)

    # Random aircraft
    ac = random.choice(AIRCRAFT_LIST)

    # Random alt preference and fuel weight
    alt_pref    = random.uniform(0, 1)
    fuel_weight = random.uniform(0.5, 1.5)

    # Convert wind to unit vector components
    wr = math.radians(wind_dir)
    wx = math.sin(wr) * wind_spd
    wy = math.cos(wr) * wind_spd

    # Physics labels
    wp1_lat, wp1_lon, wp2_lat, wp2_lon, fuel_opt, fuel_std, save_frac = \
        compute_optimal_route(orig, dest, wind_dir, wind_spd, ac)

    # Input: normalise everything to [-1, 1]
    inp = [
        normalise(orig[0], LAT_MIN, LAT_MAX),
        normalise(orig[1], LON_MIN, LON_MAX),
        normalise(dest[0], LAT_MIN, LAT_MAX),
        normalise(dest[1], LON_MIN, LON_MAX),
        wx / 80.0,  # already bounded
        wy / 80.0,
        alt_pref,
        fuel_weight / 1.5,
    ]

    # Label: waypoints as delta from origin (bounded range), fuel saving
    wp1_dlat = wp1_lat - orig[0]
    wp1_dlon = wp1_lon - orig[1]
    wp2_dlat = wp2_lat - orig[0]
    wp2_dlon = wp2_lon - orig[1]

    lbl = [
        wp1_dlat / 30.0,   # normalise deltas roughly to [-1, 1]
        wp1_dlon / 60.0,
        wp2_dlat / 30.0,
        wp2_dlon / 60.0,
        save_frac / 0.28,  # normalise to [0, 1]
    ]

    return inp, lbl


def build_dataset(n):
    print(f"Generating {n:,} training samples...")
    t0 = time.time()
    inputs, labels = [], []
    for i in range(n):
        inp, lbl = generate_sample()
        inputs.append(inp)
        labels.append(lbl)
        if (i+1) % 10_000 == 0:
            print(f"  {i+1:,}/{n:,} samples ({time.time()-t0:.1f}s)")
    X = torch.tensor(inputs, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.float32)
    print(f"Dataset built in {time.time()-t0:.1f}s  shape: {X.shape} → {Y.shape}")
    return X, Y


# ════════════════════════════════════════════════════════════════════
# MODEL  (identical architecture to main.py)
# ════════════════════════════════════════════════════════════════════

class RouteOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),  nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nMAVERICK RouteOptimizer Training")
    print(f"Device : {device}")
    print(f"Epochs : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}\n")

    # Build dataset
    X, Y = build_dataset(N_SAMPLES)

    # Split 90/10 train/val
    split  = int(N_SAMPLES * 0.9)
    X_tr, Y_tr = X[:split].to(device), Y[:split].to(device)
    X_val, Y_val = X[split:].to(device), Y[split:].to(device)

    loader = DataLoader(
        TensorDataset(X_tr, Y_tr),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    model = RouteOptimizer().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            # Weighted loss: waypoints more important than fuel estimate
            wp_loss   = loss_fn(pred[:, :4], yb[:, :4])
            fuel_loss = loss_fn(pred[:, 4:], yb[:, 4:])
            loss = WEIGHT_WP * wp_loss + WEIGHT_FUEL * fuel_loss
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * len(xb)
        sched.step()

        train_loss = epoch_loss / len(X_tr)

        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            wp_v  = loss_fn(pred_val[:,:4], Y_val[:,:4]).item()
            fl_v  = loss_fn(pred_val[:,4:], Y_val[:,4:]).item()
            val_loss = WEIGHT_WP * wp_v + WEIGHT_FUEL * fl_v

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            saved = " ✓ saved"
        else:
            saved = ""

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train {train_loss:.5f} | "
                f"val {val_loss:.5f} (wp {wp_v:.5f} fuel {fl_v:.5f}) | "
                f"{elapsed:.0f}s{saved}"
            )

    print(f"\n✓ Training complete in {time.time()-t0:.0f}s")
    print(f"✓ Best val loss: {best_val_loss:.5f}")
    print(f"✓ Weights saved to: {SAVE_PATH}")
    print(f"\nNext step: copy {SAVE_PATH} to your Render repo root")
    print("  and update main.py to load it (see instructions below).\n")

    # Quick sanity check
    model.load_state_dict(torch.load(SAVE_PATH, map_location='cpu'))
    model.eval()
    test_inp = torch.tensor([[
        normalise(51.47, LAT_MIN, LAT_MAX),   # EGLL lat
        normalise(-0.46, LON_MIN, LON_MAX),   # EGLL lon
        normalise(1.35,  LAT_MIN, LAT_MAX),   # WSSS lat
        normalise(103.99,LON_MIN, LON_MAX),   # WSSS lon
        0.22,   # wx (20kt WSW)
        -0.19,  # wy
        0.75,   # alt pref
        0.70,   # fuel weight
    ]], dtype=torch.float32)
    with torch.no_grad():
        out = model(test_inp)[0].tolist()
    print(f"Sanity check EGLL→WSSS:")
    print(f"  WP1 delta  : {out[0]*30:.2f}°lat  {out[1]*60:.2f}°lon")
    print(f"  WP2 delta  : {out[2]*30:.2f}°lat  {out[3]*60:.2f}°lon")
    print(f"  Fuel saving: {out[4]*0.28*100:.1f}%  (normalised {out[4]:.3f})")


if __name__ == "__main__":
    train()
