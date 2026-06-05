# Smart-Pricing-API

NANCY's Smart Pricing API runs a reverse-auction simulation to determine the winning service provider and final price from a set of competing bids.

## Overview

The API accepts a list of service providers, each with a minimum and maximum price. It configures and runs a multi-round reverse auction (via the `spm` simulation engine), then returns the winning provider and the agreed price. A safety mechanism ensures the final price never falls below the winner's minimum price.

## Prerequisites

- Python 3.11+ (recommended; matches the Docker image)
- pip

For Docker:

- Docker and Docker Compose

## Installation

### Local setup

1. Clone or download this repository and open a terminal in the project root (`Smart-Pricing-API/`).

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   ```

   **Windows (PowerShell):**

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   **Linux / macOS:**

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Docker setup

Build and run with Docker Compose:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## Running the API

### Local (development)

From the project root, start the server with hot reload:

```bash
uvicorn smart_pricing_api:smart_pricing_api --reload
```

The API listens on `http://127.0.0.1:8000` by default.

### Production-style (local)

```bash
uvicorn smart_pricing_api:smart_pricing_api --host 0.0.0.0 --port 8000
```

## Interactive API docs

Once the server is running, open:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

You can send test requests directly from the Swagger UI.

## API endpoints

### `GET /`

Health/welcome endpoint.

**Response:**

```json
{
  "message": "Welcome to NANCY's Smart Pricing"
}
```

### `POST /price_calculation`

Runs the reverse-auction simulation and returns the winning provider and price.

**Request body:**

| Field | Type | Description |
|-------|------|-------------|
| `services` | array | List of competing service offers |
| `services[].provider_id` | string | Unique identifier for the provider |
| `services[].service_id` | string | Identifier for the service being priced |
| `services[].minprice` | float | Minimum acceptable price for this provider |
| `services[].maxprice` | float | Starting/maximum bid price for this provider |

**Example request:**

```bash
curl -X POST "http://localhost:8000/price_calculation" \
  -H "Content-Type: application/json" \
  -d '{
    "services": [
      {
        "provider_id": "provider_a",
        "service_id": "cloud_storage",
        "minprice": 30.0,
        "maxprice": 120.0
      },
      {
        "provider_id": "provider_b",
        "service_id": "cloud_storage",
        "minprice": 35.0,
        "maxprice": 116.0
      },
      {
        "provider_id": "provider_c",
        "service_id": "cloud_storage",
        "minprice": 32.0,
        "maxprice": 117.0
      }
    ]
  }'
```

**PowerShell example:**

```powershell
$body = @{
  services = @(
    @{
      provider_id = "provider_a"
      service_id  = "cloud_storage"
      minprice    = 30.0
      maxprice    = 120.0
    },
    @{
      provider_id = "provider_b"
      service_id  = "cloud_storage"
      minprice    = 35.0
      maxprice    = 116.0
    }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://localhost:8000/price_calculation" -Method Post -Body $body -ContentType "application/json"
```

**Example response:**

```json
{
  "services": {
    "provider_id": "provider_a",
    "price": 45.36,
    "service_id": "cloud_storage"
  }
}
```

**Error response (invalid payload):**

```json
{
  "message": "Invalid payload format",
  "details": [ ... ]
}
```

## How it works

1. The API reads the submitted `services` list and extracts provider IDs, min/max prices, and the service ID.
2. It updates `spm/config.yml` with auction parameters (number of bidders, initial prices, average minimum price, etc.).
3. It runs the reverse-auction simulation: `python main.py --mode evaluate` inside the `spm/` directory.
4. Results are read from `spm/auction_results.json`.
5. If the winning bid is below that provider's `minprice`, the API walks back through round-by-round winners in `spm/outputs/winner_agents.json` and selects the most recent valid bid that respects the minimum price.
6. The winning `provider_id`, final `price`, and `service_id` are returned.

## Project structure

```
Smart-Pricing-API/
├── smart_pricing_api.py   # FastAPI application and /price_calculation endpoint
├── endpoint_classes.py    # Pydantic request models
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container image definition
├── docker-compose.yml     # Docker Compose service (port 8000)
└── spm/                   # Reverse-auction simulation engine
    ├── main.py            # Entry point (--mode evaluate)
    ├── config.yml         # Auction configuration (updated per request)
    ├── auction_results.json
    └── outputs/
        └── winner_agents.json
```

## Configuration

Auction settings in `spm/config.yml` are overwritten on each `/price_calculation` request. Defaults that apply unless overridden by the request:

| Setting | Default | Description |
|---------|---------|-------------|
| `max_rounds` | 10 | Number of bidding rounds |
| `evaluation.num_games` | 1 | Number of auction games to simulate |

To change persistent simulation behavior (e.g. render mode, training parameters), edit `spm/config.yml` directly.

## Troubleshooting

- **Port already in use:** Run on a different port, e.g. `uvicorn smart_pricing_api:smart_pricing_api --reload --port 8001`.
- **Module not found:** Ensure you are in the project root and the virtual environment is activated before starting the server.
- **Docker OpenCV errors:** The Dockerfile installs `libgl1-mesa-glx` for OpenCV; use `docker compose up --build` rather than running the image without rebuilding after changes.
- **Slow first request:** The first `/price_calculation` call loads ML/simulation dependencies and runs the full auction; subsequent calls follow the same path and may take a few seconds.
