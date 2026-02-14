# LifeLedger Backend

Backend API for LifeLedger - a privacy-conscious app that turns screenshots and photos into a searchable knowledge base.

## Architecture

- FastAPI backend with PaddleOCR and LangGraph agent
- Azure GPT-4.1 for VLM (vision + extraction + reasoning)
- PostgreSQL with pgvector for semantic search
- AWS S3 for image storage
- AWS Cognito for authentication

## Quick Start

### Local Development

```bash
# Copy environment variables
cp .env.example .env
# Edit .env with your credentials

# Run with Docker
docker-compose up --build
```

### Deploy to EC2

```bash
# SSH into EC2
ssh -i lifeledger-key.pem ubuntu@<ec2-ip>

# Clone and run
git clone <repo-url>
cd LifeLedgerCapstone
cp .env.example .env
# Edit .env with production credentials
docker-compose up -d
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /upload | Upload images |
| POST | /search | Semantic + keyword search |
| GET | /documents | List user documents |
| GET | /radar | Upcoming deadlines |
| POST | /ask | Agent queries |

## Environment Variables

See `.env.example` for required configuration.

## Project Structure

```
app/
  main.py          # FastAPI application
  ocr_pipeline.py  # PaddleOCR + document classification
  vlm_client.py    # Azure GPT-4.1 integration
  extraction.py    # Field extraction for receipts
  agent.py         # LangGraph agent + tools
  auth.py          # Cognito JWT validation
  db.py            # PostgreSQL + pgvector
  s3.py            # S3 operations
```

## Team

UC Berkeley MIDS Capstone - Spring 2025
