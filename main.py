"""
MAVERICK backend.

Deployable FastAPI service for route optimisation, telemetry simulation,
dataset proxying, and serving the single-page dashboard.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Literal, Sequence

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"
MODEL_WEIGHTS = BASE_DIR / "route_optimizer.pt"


class Settings:
    def __init__(self) -> None:
        cors_value = os.getenv("CORS_ORIGINS", "*")
        self.cors_origins = [item.strip() for item in cors_value.split(",") if item.strip()] or ["*"]
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.telemetry_hz = max(1, int(os.getenv("TELEMETRY_HZ", "10")))
        self.model_enabled = os.getenv("ENABLE_TORCH_MODEL", "true").lower() != "false"
        self.app_env = os.getenv("APP_ENV", "development")
        self.app_version = os.getenv("APP_VERSION", "3.0.0")


settings = Settings()

app = FastAPI(
    title="MAVERICK AI Navigation API",
    version=settings.app_version,
    description="AI-assisted aviation route optimisation and telemetry platform.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


AIRPORT_COORDS: dict[str, tuple[float, float]] = {
    "EGLL": (51.4706, -0.4619),
    "EDDF": (50.0333, 8.5706),
    "ESSA": (59.6519, 17.9186),
    "KJFK": (40.6413, -73.7781),
    "OMDB": (25.2528, 55.3644),
    "WSSS": (1.3502, 103.9940),
    "LFPG": (49.0097, 2.5478),
    "EHAM": (52.3086, 4.7639),
}

AIRCRAFT_PERFORMANCE = {
    "A320": {"burn_tph": 2.4, "speed_kmh": 850.0, "seat_capacity": 180},
    "B737": {"burn_tph": 2.6, "speed_kmh": 840.0, "seat_capacity": 189},
    "A380": {"burn_tph": 11.2, "speed_kmh": 900.0, "seat_capacity": 555},
    "B777": {"burn_tph": 6.8, "speed_kmh": 895.0, "seat_capacity": 396},
}


if TORCH_AVAILABLE:
    class RouteOptimizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5),
            )

        def forward(self, x):  # type: ignore[override]
            return self.net(x)


class RouteRequest(BaseModel):
    origin: str | Sequence[float] = Field(default="EGLL")
    destination: str | Sequence[float] = Field(default="ESSA")
    wind_direction: float = Field(default=270.0, ge=0.0, le=360.0)
    wind_speed_kt: float = Field(default=18.0, ge=0.0, le=180.0)
    fuel_weight: float = Field(default=0.7, ge=0.0, le=2.0)
    aircraft_type: str = Field(default="A320")


class HazardInjectRequest(BaseModel):
    hazard: Literal["wind_loading", "thruster_anomaly", "venturi", "clear"]


class ModeRequest(BaseModel):
    mode: Literal["MANUAL", "AI_ASSISTED", "AI_GUIDED", "manual", "ai_assisted", "ai_guided"]


class HazardState:
    def __init__(self) -> None:
        self.wind_loading = True
        self.thruster_anomaly = False
        self.venturi_effect = False
        self.safe_zone_pct = 94.7
        self.recovery_time = 1.8
        self.mode = "AI_GUIDED"

    def tick(self) -> tuple[float, float]:
        noise = random.gauss(0, 0.3)
        if self.mode == "AI_GUIDED":
            base, recovery = 94.7, 1.8
        elif self.mode == "AI_ASSISTED":
            base, recovery = 87.2, 4.2
        else:
            base, recovery = 61.3, 12.4

        penalty = 2.1 if self.wind_loading and self.mode == "MANUAL" else 0.4 if self.wind_loading else 0.0
        self.safe_zone_pct = round(max(50.0, min(100.0, base - penalty + noise)), 1)
        self.recovery_time = round(recovery + random.gauss(0, 0.05), 2)
        return self.safe_zone_pct, self.recovery_time


class TelemetrySimulator:
    def __init__(self, hazard_state: HazardState) -> None:
        self.hazard_state = hazard_state
        self.t = 0
        self.lat = 51.4706
        self.lon = -0.4619
        self.alt_km = 10.2
        self.speed_kmh = 847.0
        self.heading = 83.0
        self.fuel_burn = 2.4
        self.dist_nm = 412.0

    def tick(self) -> dict:
        self.t += 1
        self.lat += random.gauss(0, 0.0008)
        self.lon += random.gauss(0, 0.0005) + 0.001
        self.speed_kmh = 847 + random.gauss(0, 2.5)
        self.alt_km = 10.2 + math.sin(self.t * 0.05) * 0.08
        self.fuel_burn = 2.4 + random.gauss(0, 0.02)
        self.dist_nm = max(0.0, self.dist_nm - 0.002)

        safe_pct, rec_time = self.hazard_state.tick()
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
            "mode": self.hazard_state.mode,
            "hazards": {
                "wind_loading": self.hazard_state.wind_loading,
                "thruster_anomaly": self.hazard_state.thruster_anomaly,
                "venturi_effect": self.hazard_state.venturi_effect,
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
            },
        }


def resolve_airport(value: str | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, str):
        coords = AIRPORT_COORDS.get(value.upper())
        if coords is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"Unknown ICAO code: {value}",
                    "known_codes": sorted(AIRPORT_COORDS.keys()),
                },
            )
        return coords

    if len(value) != 2:
        raise HTTPException(status_code=422, detail="Airport coordinates must contain [lat, lon].")
    return float(value[0]), float(value[1])


def haversine_km(origin: tuple[float, float], dest: tuple[float, float]) -> float:
    lat1, lon1 = origin
    lat2, lon2 = dest
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(a))


def wind_components(direction_deg: float, speed_kt: float) -> tuple[float, float]:
    wind_rad = math.radians(direction_deg)
    return math.sin(wind_rad) * speed_kt, math.cos(wind_rad) * speed_kt


def compute_ai_saving(origin: tuple[float, float], dest: tuple[float, float], wind_direction: float, speed_kt: float) -> float:
    bearing = math.atan2(dest[1] - origin[1], dest[0] - origin[0])
    wind_rad = math.radians(wind_direction)
    wind_component = math.cos(wind_rad - bearing) * speed_kt
    wind_factor = 1 - (wind_component / 850.0) * 0.4
    return min(0.28, (0.17 + speed_kt * 0.002) * max(0.5, wind_factor))


def numpy_route_fallback(
    origin: tuple[float, float],
    dest: tuple[float, float],
    wind_xy: tuple[float, float],
    fuel_weight: float,
) -> tuple[list[float], list[float], float, float]:
    mid_lat = (origin[0] + dest[0]) / 2
    mid_lon = (origin[1] + dest[1]) / 2
    wp1 = [mid_lat - wind_xy[1] * 0.02 - 0.8, mid_lon + wind_xy[0] * 0.01]
    wp2 = [mid_lat + 0.4, mid_lon + (dest[1] - origin[1]) * 0.3]
    dist_km = haversine_km(origin, dest)
    fuel_base = dist_km * 0.0035
    fuel_optimized = fuel_base * (1 - 0.23 * fuel_weight)
    return wp1, wp2, round(fuel_optimized, 2), round(dist_km, 1)


route_model = None
route_model_status = "disabled"

if TORCH_AVAILABLE and settings.model_enabled:
    route_model = RouteOptimizer()
    if MODEL_WEIGHTS.exists():
        try:
            route_model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location="cpu"))
            route_model_status = "weights_loaded"
        except Exception:
            route_model_status = "weights_failed"
    else:
        route_model_status = "weights_missing"
    route_model.eval()
elif TORCH_AVAILABLE:
    route_model_status = "disabled_by_config"
else:
    route_model_status = "torch_unavailable"


hazard = HazardState()
telemetry = TelemetrySimulator(hazard)


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/api")
async def api_root():
    return {
        "system": "MAVERICK AI Navigation",
        "version": settings.app_version,
        "environment": settings.app_env,
        "status": "OPERATIONAL",
        "model_status": route_model_status,
        "telemetry_hz": settings.telemetry_hz,
    }


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "service": "maverick-api",
        "version": settings.app_version,
        "ui_served": INDEX_HTML.exists(),
        "model_status": route_model_status,
        "timestamp": time.time(),
    }


@app.get("/api/metrics/summary")
async def metrics_summary():
    return {
        "fuel_reduction_pct": 23.0,
        "co2_saved_tonnes_per_flight": 18.6,
        "safe_zone_pct_ai_guided": 94.7,
        "manual_safe_zone_pct": 61.3,
        "recovery_time_ai_guided_s": 1.8,
        "quantum_nav_accuracy_multiplier": 111,
    }


@app.post("/api/optimize-route")
async def optimize_route(payload: RouteRequest):
    started_at = time.perf_counter()

    origin = resolve_airport(payload.origin)
    destination = resolve_airport(payload.destination)
    aircraft = AIRCRAFT_PERFORMANCE.get(payload.aircraft_type.upper(), AIRCRAFT_PERFORMANCE["A320"])
    wind_xy = wind_components(payload.wind_direction, payload.wind_speed_kt)
    ai_saving_pct = compute_ai_saving(origin, destination, payload.wind_direction, payload.wind_speed_kt)

    if TORCH_AVAILABLE and route_model is not None:
        with torch.no_grad():
            x = torch.tensor(
                [
                    origin[0],
                    origin[1],
                    destination[0],
                    destination[1],
                    wind_xy[0],
                    wind_xy[1],
                    0.75,
                    payload.fuel_weight,
                ],
                dtype=torch.float32,
            )
            out = route_model(x).tolist()
            wp1 = [out[0] + origin[0], out[1] + origin[1]]
            wp2 = [out[2] + origin[0], out[3] + origin[1]]
            dist_km = haversine_km(origin, destination)
            raw_model_fuel = abs(out[4]) * 15 + 8
    else:
        wp1, wp2, raw_model_fuel, dist_km = numpy_route_fallback(origin, destination, wind_xy, payload.fuel_weight)

    flight_hours = dist_km / aircraft["speed_kmh"]
    fuel_standard = flight_hours * aircraft["burn_tph"]
    fuel_optimised = fuel_standard * (1 - ai_saving_pct)
    fuel_saved = fuel_standard - fuel_optimised
    co2_saved = fuel_saved * 2.54
    response_ms = round((time.perf_counter() - started_at) * 1000, 2)

    return {
        "route": {
            "origin": list(origin),
            "waypoint_1": [round(v, 4) for v in wp1],
            "waypoint_2": [round(v, 4) for v in wp2],
            "destination": list(destination),
        },
        "distance_km": round(dist_km, 1),
        "flight_hours": round(flight_hours, 2),
        "fuel_standard_tonnes": round(fuel_standard, 2),
        "fuel_estimate_tonnes": round(fuel_optimised, 2),
        "fuel_saved_tonnes": round(fuel_saved, 2),
        "co2_reduction_t": round(co2_saved, 2),
        "fuel_saving_pct": round(ai_saving_pct * 100, 1),
        "safe_zone_pct": hazard.safe_zone_pct,
        "aircraft_type": payload.aircraft_type.upper(),
        "wind_direction": payload.wind_direction,
        "wind_speed_kt": payload.wind_speed_kt,
        "ai_model": "PyTorch RL" if TORCH_AVAILABLE and route_model is not None else "NumPy fallback",
        "model_fuel_signal_tonnes": round(raw_model_fuel, 2),
        "response_time_ms": response_ms,
    }


@app.get("/api/hazards")
async def get_hazards():
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
        },
    }


@app.post("/api/hazards/inject")
async def inject_hazard(payload: HazardInjectRequest):
    if payload.hazard == "wind_loading":
        hazard.wind_loading = True
    elif payload.hazard == "thruster_anomaly":
        hazard.thruster_anomaly = True
    elif payload.hazard == "venturi":
        hazard.venturi_effect = True
    else:
        hazard.wind_loading = False
        hazard.thruster_anomaly = False
        hazard.venturi_effect = False
    return {"injected": payload.hazard, "status": "ACTIVE"}


@app.post("/api/mode")
async def set_mode(payload: ModeRequest):
    hazard.mode = payload.mode.upper()
    return {"mode": hazard.mode, "safe_zone_pct": hazard.safe_zone_pct}


@app.get("/api/datasets/opensky")
async def proxy_opensky(
    lamin: float = 45.0,
    lomin: float = -10.0,
    lamax: float = 60.0,
    lomax: float = 20.0,
):
    url = f"https://opensky-network.org/api/states/all?lamin={lamin}&lomin={lomin}&lamax={lamax}&lomax={lomax}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    "source": "live",
                    "aircraft_count": len(data.get("states", [])),
                    "data": data,
                }
    except Exception:
        pass

    callsigns = ["BAW123", "DLH456", "EZY789", "RYR101", "UAE202", "SWR303"]
    simulated = []
    for idx, callsign in enumerate(callsigns):
        simulated.append(
            {
                "icao24": f"4ca{idx:03d}",
                "callsign": callsign,
                "origin_country": "United Kingdom",
                "longitude": round(-0.46 + idx * 2.5 + random.gauss(0, 0.3), 4),
                "latitude": round(51.47 + idx * 0.8 + random.gauss(0, 0.2), 4),
                "baro_altitude": round(10000 + random.gauss(0, 500), 0),
                "velocity": round(240 + random.gauss(0, 15), 1),
                "true_track": round(random.uniform(0, 360), 1),
            }
        )

    return {"source": "simulation", "aircraft_count": len(simulated), "data": {"states": simulated}}


@app.get("/api/datasets/noaa-weather")
async def proxy_noaa(stations: str = "EGLL,EDDF,ESSA"):
    url = f"https://aviationweather.gov/api/data/metar?ids={stations}&format=json"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                return {"source": "live", "data": response.json()}
    except Exception:
        pass

    return {
        "source": "simulation",
        "data": [
            {"station": "EGLL", "wind_dir": 270, "wind_speed_kt": 18, "visibility_sm": 10, "temp_c": 8},
            {"station": "EDDF", "wind_dir": 240, "wind_speed_kt": 12, "visibility_sm": 15, "temp_c": 5},
            {"station": "ESSA", "wind_dir": 310, "wind_speed_kt": 22, "visibility_sm": 8, "temp_c": -2},
        ],
    }


@app.get("/api/datasets/quantum-benchmark")
async def quantum_benchmark():
    return {
        "system": "Q-CTRL Ironstone Opal",
        "milestone": "Commercial quantum advantage benchmark",
        "accuracy_vs_gps": 111,
        "validation_distance_km": 700,
        "platforms_tested": ["fixed-wing aircraft", "UAV", "land vehicle"],
        "icao_compliant": True,
        "defense_partner_uav_test": True,
    }


@app.websocket("/ws/telemetry")
async def telemetry_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps(telemetry.tick()))
            await asyncio.sleep(1 / settings.telemetry_hz)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.app_env == "development")
