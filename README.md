# Employee Attendance & Analytics API

FastAPI and MongoDB backend service for managing employee attendance lifecycle, manual regularization, concurrency-safe punch events, and analytical rollups via aggregation pipelines.

## Prerequisites

- Python 3.11+
- MongoDB 6.0+ (local instance or MongoDB Atlas)

## Environment Variables

The application reads configuration from environment variables (system environment variables take precedence over `.env`):

| Variable | Description | Default |
|---|---|---|
| `MONGO_URI` | MongoDB connection URI string | `mongodb://localhost:27017` |
| `MONGO_DB` | Database name | `attendance_db` |

## Installation

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Database Setup

Indexes are created automatically and idempotently on application startup. No manual index creation is required.

To seed the initial sample dataset (optional):
```bash
python sample_seed.py
```

## Running the Application

Start the service using Uvicorn:
```bash
uvicorn app.main:app --port 8000
```

The application becomes ready immediately and exposes readiness probes at `GET /health`.

## API Documentation

Interactive Swagger documentation is accessible at:
- http://localhost:8000/docs
- OpenAPI JSON: http://localhost:8000/openapi.json

## Running Tests

Run the automated test suite with pytest:
```bash
pytest -v
```

## Architecture & Implementation Notes

- **Data Models**: All records use customer identifier `emp_code` as their public natural identifier. MongoDB `_id` is never exposed.
- **Concurrency & Races**: Compound unique index `(emp_code, date)` guarantees atomic punch-in conflict handling (HTTP 409). Punch-out uses atomic conditional updates (`punch_out: None`). Regularization patches enforce history array length guards to prevent lost audit trails.
- **Analytics & Aggregation**: Monthly summaries, cross-collection department summaries (including zero-log headcount), leaderboard with competition tie ranking (`$rank`), and 7-day moving average department trends with gap-filling (`$densify`, `$setWindowFields`) run completely in MongoDB pipelines.
- **Index Optimization**: All query paths and aggregation match stages are backed by targeted compound indexes verifiable via `GET /admin/explain/{endpoint}` (`IXSCAN`).
