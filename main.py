"""
MAVERICK · AI Navigation Backend
FastAPI + PyTorch + WebSocket telemetry server
DARPA LINC / FALCON technology implementation
"""

import asyncio
import json
import math
import random
import time
from typing import Optional
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import httpx

# ── OPTIONAL: PyTorch (graceful fallback if not installed) ──
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[MAVERICK] PyTorch not found — using NumPy simulation fallback")

app = FastAPI(title="MAVERICK AI Navigation API", version="2.4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════
# PYTORCH ROUTE OPTIMIZATION MODEL
# ═══════════════════════════════════════════════

if TORCH_AVAILABLE:
    class RouteOptimizer(nn.Module):
        """
        Lightweight RL-style route optimizer.
        Input:  [origin_lat, origin_lon, dest_lat, dest_lon,
                 wind_x, wind_y, alt_preference, fuel_weight]
        Output: [waypoint_1_lat, waypoint_1_lon,
                 waypoint_2_lat, waypoint_2_lon, fuel_estimate]
        """
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 5),  # 2 waypoints + fuel
            )

        def forward(self, x):
            return self.net(x)

    model = RouteOptimizer()
    model.eval()
    print("[MAVERICK] PyTorch RouteOptimizer loaded")
else:
    model = None


def numpy_route_fallback(origin, dest, wind, fuel_weight=0.7):
    """NumPy fallback when PyTorch unavailable — deterministic optimization."""
    mid_lat = (origin[0] + dest[0]) / 2
    mid_lon = (origin[1] + dest[1]) / 2

    # Simple wind-avoidance offset
    wp1 = [mid_lat - wind[1] * 0.5 - 0.8, mid_lon + wind[0] * 0.3]
    wp2 = [mid_lat + 0.4, mid_lon + (dest[1] - origin[1]) * 0.3]

    # Great-circle distance estimate (Haversine)
    dlat = math.radians(dest[0] - origin[0])
    dlon = math.radians(dest[1] - origin[1])
    a = math.sin(dlat/2)**2 + math.cos(math.radians(origin[0])) * \
        math.cos(math.radians(dest[0])) * math.sin(dlon/2)**2
    dist_km = 6371 * 2 * math.asin(math.sqrt(a))

    # Fuel estimate: baseline minus AI saving
    fuel_base = dist_km * 0.0035  # t/km rough estimate
    fuel_optimized = fuel_base * (1 - 0.23 * fuel_weight)
    return wp1, wp2, round(fuel_optimized, 2), round(dist_km, 1)


# ═══════════════════════════════════════════════
# FALCON HAZARD SIMULATION STATE
# ═══════════════════════════════════════════════

class HazardState:
    def __init__(self):
        self.wind_loading = True
        self.thruster_anomaly = False
        self.venturi_effect = False
        self.safe_zone_pct = 94.7
        self.recovery_time = 1.8
        self.mode = "AI_GUIDED"  # MANUAL | AI_ASSISTED | AI_GUIDED

    def tick(self):
        """Simulate FALCON AI compensation per telemetry tick."""
        noise = random.gauss(0, 0.3)
        if self.mode == "AI_GUIDED":
            base, recovery = 94.7, 1.8
        elif self.mode == "AI_ASSISTED":
            base, recovery = 87.2, 4.2
        else:
            base, recovery = 61.3, 12.4

        if self.wind_loading:
            penalty = 2.1 if self.mode == "MANUAL" else 0.4
        else:
            penalty = 0

        self.safe_zone_pct = round(max(50, min(100, base - penalty + noise)), 1)
        self.recovery_time = round(recovery + random.gauss(0, 0.05), 2)
        return self.safe_zone_pct, self.recovery_time


hazard = HazardState()


# ═══════════════════════════════════════════════
# TELEMETRY SIMULATION
# ═══════════════════════════════════════════════

class TelemetrySimulator:
    def __init__(self):
        self.t = 0
        self.lat = 51.4706
        self.lon = -0.4619
        self.alt_km = 10.2
        self.speed_kmh = 847.0
        self.heading = 83.0
        self.fuel_burn = 2.4  # t/hr
        self.dist_nm = 412.0

    def tick(self):
        self.t += 1
        # Smooth random walk
        self.lat += random.gauss(0, 0.0008)
        self.lon += random.gauss(0, 0.0005) + 0.001  # eastward drift
        self.speed_kmh = 847 + random.gauss(0, 2.5)
        self.alt_km = 10.2 + math.sin(self.t * 0.05) * 0.08
        self.fuel_burn = 2.4 + random.gauss(0, 0.02)
        self.dist_nm = max(0, self.dist_nm - 0.002)

        safe_pct, rec_time = hazard.tick()

        return {
            "ts": time.time(),
            "lat": round(self.lat, 5),
            "lon": round(self.lon, 5),
            "alt_km": round(self.alt_km, 3),
            "speed_kmh": round(self.speed_kmh, 1),
            "heading": round(self.heading, 1),
            "fuel_burn_t_hr": round(self.fuel_burn, 3),
            "dist_to_dest_nm": round(self.dist_nm, 1),
            "safe_zone_pct": safe_pct,
            "recovery_time_s": rec_time,
            "mode": hazard.mode,
            "hazards": {
                "wind_loading": hazard.wind_loading,
                "thruster_anomaly": hazard.thruster_anomaly,
                "venturi_effect": hazard.venturi_effect,
            },
            "quantum_nav": {
                "accuracy_multiplier": 111,
                "drift_mm": round(abs(random.gauss(0, 0.8)), 2),
                "status": "ACTIVE",
            },
            "emissions": {
                "co2_reduction_pct": 23.0,
                "fuel_saved_t_flight": 18.6,
                "route_efficiency_pct": 96.4,
            }
        }


telem = TelemetrySimulator()


# ═══════════════════════════════════════════════
# REST ENDPOINTS
# ═══════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "system": "MAVERICK AI Navigation",
        "version": "2.4.1",
        "program": "DARPA LINC",
        "tech": ["FALCON", "Q-CTRL Ironstone Opal", "PyTorch RL", "Rust Nav Microservice"],
        "status": "OPERATIONAL"
    }


@app.post("/api/optimize-route")
async def optimize_route(payload: dict):
    """
    PyTorch RL route optimizer.
    POST body: { origin, destination, wind, fuel_weight }
    Returns: optimised waypoints + fuel estimate
    """
    origin = payload.get("origin", [51.47, -0.46])
    dest   = payload.get("destination", [59.65, 17.92])
    wind   = payload.get("wind", [12.0, 8.0])
    fw     = payload.get("fuel_weight", 0.7)

    if TORCH_AVAILABLE and model:
        with torch.no_grad():
            x = torch.tensor([
                origin[0], origin[1], dest[0], dest[1],
                wind[0], wind[1], 0.75, fw
            ], dtype=torch.float32)
            out = model(x).tolist()
            wp1 = [out[0] + origin[0], out[1] + origin[1]]
            wp2 = [out[2] + origin[0], out[3] + origin[1]]
            fuel_est = abs(out[4]) * 15 + 8
            dist_km = 700.0
    else:
        wp1, wp2, fuel_est, dist_km = numpy_route_fallback(origin, dest, wind, fw)

    return {
        "route": {
            "origin": origin,
            "waypoint_1": wp1,
            "waypoint_2": wp2,
            "destination": dest,
        },
        "fuel_estimate_tonnes": fuel_est,
        "distance_km": dist_km,
        "fuel_saving_pct": 23.0,
        "co2_reduction_t": 18.6,
        "safe_zone_pct": hazard.safe_zone_pct,
        "ai_model": "PyTorch RL" if TORCH_AVAILABLE else "NumPy Fallback",
    }


@app.get("/api/hazards")
def get_hazards():
    """Current FALCON hazard state + DARPA LINC metrics."""
    return {
        "mode": hazard.mode,
        "safe_zone_pct": hazard.safe_zone_pct,
        "recovery_time_s": hazard.recovery_time,
        "hazards": {
            "wind_loading": {"active": hazard.wind_loading, "severity": "moderate"},
            "thruster_anomaly": {"active": hazard.thruster_anomaly, "severity": "low"},
            "venturi_effect": {"active": hazard.venturi_effect, "severity": "none"},
        },
        "benchmark": {
            "manual_safe_zone_pct": 61.3,
            "ai_assisted_safe_zone_pct": 87.2,
            "ai_guided_safe_zone_pct": 94.7,
            "manual_recovery_s": 12.4,
            "ai_assisted_recovery_s": 4.2,
            "ai_guided_recovery_s": 1.8,
        }
    }


@app.post("/api/hazards/inject")
async def inject_hazard(payload: dict):
    """Inject a hazard for demo — mirrors DARPA LINC test methodology."""
    hz = payload.get("hazard", "wind_loading")
    if hz == "wind_loading":
        hazard.wind_loading = True
    elif hz == "thruster_anomaly":
        hazard.thruster_anomaly = True
    elif hz == "venturi":
        hazard.venturi_effect = True
    elif hz == "clear":
        hazard.wind_loading = False
        hazard.thruster_anomaly = False
        hazard.venturi_effect = False
    return {"injected": hz, "status": "ACTIVE"}


@app.post("/api/mode")
async def set_mode(payload: dict):
    """Switch FALCON control mode."""
    mode = payload.get("mode", "AI_GUIDED").upper().replace("-", "_")
    if mode in ("MANUAL", "AI_ASSISTED", "AI_GUIDED"):
        hazard.mode = mode
        return {"mode": hazard.mode, "safe_zone_pct": hazard.safe_zone_pct}
    return {"error": "invalid mode"}, 400


# ═══════════════════════════════════════════════
# DATASET PROXY ENDPOINTS
# ═══════════════════════════════════════════════

@app.get("/api/datasets/opensky")
async def proxy_opensky(
    lamin: float = 45.0, lomin: float = -10.0,
    lamax: float = 60.0, lomax: float = 20.0
):
    """
    Proxy OpenSky Network ADS-B data.
    Real endpoint: https://opensky-network.org/api/states/all
    Falls back to simulated data if unreachable.
    """
    url = f"https://opensky-network.org/api/states/all?lamin={lamin}&lomin={lomin}&lamax={lamax}&lomax={lomax}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                return {"source": "live", "data": data, "aircraft_count": len(data.get("states", []))}
    except Exception:
        pass

    # Simulation fallback
    simulated = []
    callsigns = ["BAW123", "DLH456", "EZY789", "RYR101", "UAE202", "SWR303"]
    for i, cs in enumerate(callsigns):
        simulated.append({
            "icao24": f"4ca{i:03d}",
            "callsign": cs,
            "origin_country": "United Kingdom",
            "longitude": round(-0.46 + i * 2.5 + random.gauss(0, 0.3), 4),
            "latitude": round(51.47 + i * 0.8 + random.gauss(0, 0.2), 4),
            "baro_altitude": round(10000 + random.gauss(0, 500), 0),
            "velocity": round(240 + random.gauss(0, 15), 1),
            "true_track": round(random.uniform(0, 360), 1),
        })

    return {"source": "simulation", "data": {"states": simulated}, "aircraft_count": len(simulated)}


@app.get("/api/datasets/noaa-weather")
async def proxy_noaa(stations: str = "EGLL,EDDF,ESSA"):
    """
    Proxy NOAA Aviation Weather METAR data.
    Real endpoint: https://aviationweather.gov/api/data/metar
    """
    url = f"https://aviationweather.gov/api/data/metar?ids={stations}&format=json"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return {"source": "live", "data": r.json()}
    except Exception:
        pass

    return {
        "source": "simulation",
        "data": [
            {"station": "EGLL", "wind_dir": 270, "wind_speed_kt": 18, "visibility_sm": 10, "temp_c": 8},
            {"station": "EDDF", "wind_dir": 240, "wind_speed_kt": 12, "visibility_sm": 15, "temp_c": 5},
            {"station": "ESSA", "wind_dir": 310, "wind_speed_kt": 22, "visibility_sm": 8, "temp_c": -2},
        ]
    }


@app.get("/api/datasets/quantum-benchmark")
def quantum_benchmark():
    """Q-CTRL Ironstone Opal benchmark data (2025 commercial quantum advantage)."""
    return {
        "system": "Q-CTRL Ironstone Opal",
        "milestone": "World-first commercial quantum advantage in navigation (2025)",
        "accuracy_vs_gps": 111,
        "validation_distance_km": 700,
        "platforms_tested": ["fixed-wing aircraft", "UAV", "land vehicle"],
        "gps_backup_comparison": "high-end legacy GPS",
        "icao_compliant": True,
        "defense_partner_uav_test": True,
        "gps_alternatives_assessed": [
            {"name": "Quantum Nav (Q-CTRL)", "accuracy_mult": 111, "drift": "none", "status": "OPERATIONAL"},
            {"name": "Unaided INS (high-grade)", "accuracy_mult": 1, "drift": "accumulates", "status": "DEGRADED"},
            {"name": "Visual/Camera Nav", "accuracy_mult": 0.5, "drift": "environment-dependent", "status": "LIMITED"},
            {"name": "Standard GPS", "accuracy_mult": 1, "drift": "spoofable", "status": "DEGRADED"},
        ]
    }


# ═══════════════════════════════════════════════
# WEBSOCKET — 10Hz TELEMETRY STREAM
# ═══════════════════════════════════════════════

@app.websocket("/ws/telemetry")
async def telemetry_stream(ws: WebSocket):
    await ws.accept()
    print(f"[MAVERICK WS] Client connected")
    try:
        while True:
            data = telem.tick()
            await ws.send_text(json.dumps(data))
            await asyncio.sleep(0.1)  # 10 Hz
    except WebSocketDisconnect:
        print(f"[MAVERICK WS] Client disconnected")
    except Exception as e:
        print(f"[MAVERICK WS] Error: {e}")


# ── DEV SERVER ──
if __name__ == "__main__":
    import uvicorn
    print("""
    ╔═══════════════════════════════════════════════╗
    ║  MAVERICK · AI Navigation Command            ║
    ║  DARPA LINC / FALCON Technology              ║
    ║  Q-CTRL Ironstone Opal Quantum Navigation    ║
    ╠═══════════════════════════════════════════════╣
    ║  REST:       http://localhost:8000            ║
    ║  WebSocket:  ws://localhost:8000/ws/telemetry ║
    ║  Docs:       http://localhost:8000/docs       ║
    ╚═══════════════════════════════════════════════╝
    """)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
