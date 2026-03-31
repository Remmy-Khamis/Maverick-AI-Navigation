# Deployment Guide

## Recommended Architecture

Use a single deployable service for the current project:

- FastAPI serves `index.html`
- REST APIs and WebSocket run from the same domain
- Optional `route_optimizer.pt` is mounted or bundled with the app
- Live external datasets degrade gracefully to simulation mode

This keeps the frontend and backend on one origin, avoids CORS headaches, and makes the dashboard WebSocket URL self-configuring.

## Environment Variables

Copy `.env.example` and set:

- `APP_ENV` - `production` in hosted environments
- `HOST` - usually `0.0.0.0`
- `PORT` - host-assigned port, commonly `8000`
- `TELEMETRY_HZ` - telemetry stream frequency
- `ENABLE_TORCH_MODEL` - `true` or `false`
- `CORS_ORIGINS` - comma-separated allowed origins if you split the frontend later

## Option 1: Docker

```bash
docker build -t maverick-ai-navigation .
docker run --env-file .env -p 8000:8000 maverick-ai-navigation
```

Health check:

```bash
curl http://localhost:8000/api/health
```

## Option 2: Render

Create a new Web Service and point it to this repository.

- Runtime: `Docker`
- Health check path: `/api/health`
- Auto deploy: enabled
- Instance type: start with a standard or starter instance if you want more predictable cold-start behavior

Environment variables:

- `APP_ENV=production`
- `PORT=8000`
- `CORS_ORIGINS=*`
- `ENABLE_TORCH_MODEL=true`

Start command if you do not use Docker:

```bash
pip install -r requirements.txt && python main.py
```

## Option 3: Railway or Fly.io

Either platform can use the included `Dockerfile`.

- Expose port `8000`
- Set the same environment variables as above
- Use `/api/health` for probes

## Reverse Proxy Notes

If you place MAVERICK behind Nginx, Caddy, or a cloud load balancer:

- enable WebSocket upgrades for `/ws/telemetry`
- preserve `Host` and `X-Forwarded-*` headers
- terminate TLS at the proxy or platform edge

## Production Checklist

- add `route_optimizer.pt` if you want the PyTorch path instead of fallback mode
- restrict `CORS_ORIGINS` to your production domain
- add platform logging and uptime monitoring on `/api/health`
- pin dependencies and scan the container image in CI
- consider moving the large `index.html` into modular frontend assets if the dashboard will keep growing

## Suggested Next Stack Upgrades

- Split the frontend into `React + Vite` or `Next.js` for maintainability
- Add typed request and response contracts with tests
- Persist optimisation runs and emissions reports in PostgreSQL
- Add Redis for cached weather and traffic datasets
- Put the Rust navigation prototype behind a real HTTP microservice boundary
- Add CI for linting, smoke tests, and container builds
