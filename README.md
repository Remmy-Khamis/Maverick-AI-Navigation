# MAVERICK AI Navigation

MAVERICK is an AI-powered flight navigation platform that helps pilots and operators compare standard routes against more fuel-efficient alternatives. The app combines a FastAPI backend, simulated live telemetry, route optimisation logic, and a browser dashboard for fuel and emissions analysis.

## Improved Stack

- `FastAPI` backend serving both the API and the dashboard from one service
- `WebSocket` telemetry stream for near-real-time flight updates
- `PyTorch` route model with automatic NumPy fallback when weights are unavailable
- `HTML/CSS/JavaScript` dashboard with route, fuel, hazard, and dataset visualisation
- `Docker` and `docker-compose` support for local and cloud deployment
- Environment-based runtime configuration for host, port, CORS, and telemetry rate

## Project Files

- `main.py` - production-ready API and dashboard server
- `index.html` - browser dashboard
- `train_route_optimizer.py` - offline model training script
- `main.rs` - Rust prototype for navigation fusion concepts
- `requirements.txt` - Python dependencies
- `Dockerfile` - container build
- `docker-compose.yml` - local container orchestration
- `.env.example` - runtime environment template

## Local Development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Open [http://localhost:8000](http://localhost:8000).

## Training the Route Model

If you want to generate `route_optimizer.pt` locally:

```bash
pip install torch numpy
python train_route_optimizer.py
```

Place the generated `route_optimizer.pt` in the project root next to `main.py`.

## Docker

```bash
docker compose up --build
```

The app will be available at [http://localhost:8000](http://localhost:8000).

## API Endpoints

- `GET /api/health` - health check for deployment probes
- `GET /api` - API summary
- `POST /api/optimize-route` - route optimisation and fuel analysis
- `GET /api/hazards` - current hazard simulation state
- `POST /api/hazards/inject` - inject or clear demo hazards
- `POST /api/mode` - switch control modes
- `GET /api/datasets/opensky` - ADS-B proxy with simulation fallback
- `GET /api/datasets/noaa-weather` - weather proxy with simulation fallback
- `GET /api/datasets/quantum-benchmark` - benchmark metadata
- `GET /ws/telemetry` - telemetry WebSocket stream

## Deployment

Deployment guidance is in [DEPLOYMENT.md](/C:/Users/Remmy%20Khamis/Downloads/Maverick-AI-Navigation-main/Maverick-AI-Navigation-main/DEPLOYMENT.md).
