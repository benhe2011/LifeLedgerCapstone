# LifeLedger Backend

Backend API for LifeLedger - document OCR, semantic search, spending analytics, and AI-powered analysis.

## Quick Reference

- Production API: https://llapi.click
- Health Check: https://llapi.click/health
- EC2 Instance: 54.214.55.52 (internal, behind ALB)

## Architecture

Vercel (Frontend) -> AWS ALB (HTTPS) -> EC2 (FastAPI) -> Aurora PostgreSQL + S3 + Azure GPT-4.1

Components:
- ALB with ACM certificate for SSL termination
- EC2 t3.large running FastAPI + PaddleOCR
- Aurora Serverless v2 with pgvector for embeddings
- S3 for image storage with presigned URLs
- Cognito for JWT authentication
- Azure OpenAI GPT-4.1 for extraction and agent reasoning

## Deploying Code Updates

```bash
# SSH into EC2
ssh -i LLKey.pem ubuntu@54.214.55.52

# Pull and rebuild
cd LifeLedgerCapstone
git pull origin main
docker-compose down
docker-compose up --build -d

# Verify startup (watch for PaddleOCR pre-warm completion)
docker logs -f lifeledgercapstone_app_1
```

## API Endpoints

Core endpoints:

- GET /health - Health check
- POST /upload - Upload images (multipart form)
- GET /documents - List user documents
- GET /documents/{id} - Get document details with OCR blocks
- GET /documents/{id}/related - Get similar documents via pgvector
- POST /documents/{id}/review - Submit manual review for failed OCR
- DELETE /documents - Batch delete documents
- POST /search - Semantic search over documents
- POST /ask - AI agent for analytical questions
- GET /radar - Get upcoming events and deadlines

Example requests:

```bash
# Health check
curl https://llapi.click/health

# Ask the AI agent
curl -X POST "https://llapi.click/ask" \
  -H "Authorization: Bearer dev_testuser" \
  -H "Content-Type: application/json" \
  -d '{"question":"How much did I spend at Target?"}'

# Get related documents
curl "https://llapi.click/documents/1/related?limit=4" \
  -H "Authorization: Bearer dev_testuser"

# Get upcoming deadlines
curl "https://llapi.click/radar" \
  -H "Authorization: Bearer dev_testuser"
```

## AI Agent Tools

The agent uses OpenAI function calling to answer questions:

- search_documents - "Find my Amazon receipts"
- get_total_spending - "How much did I spend last month?"
- get_spending_by_merchant - "Where do I spend the most?"
- get_receipts_by_merchant - "Show me all Target receipts"
- get_receipts_by_date_range - "What did I buy in January?"

The agent uses extracted dates when available, falling back to upload timestamps for photos without dates.

## Project Structure

```
app/
  main.py          - FastAPI endpoints and CORS
  agent.py         - AI agent with OpenAI function calling
  ocr_pipeline.py  - PaddleOCR and document classification
  vlm_client.py    - Azure GPT-4.1 for extraction and radar
  radar_crawler.py - Background event extraction
  extraction.py    - Field extraction and spending queries
  db.py            - PostgreSQL, pgvector, and embeddings
  s3.py            - S3 operations and presigned URLs
  auth.py          - Cognito JWT validation
```

## Processing Pipeline

1. Upload: Images saved to S3, document record created
2. OCR: PaddleOCR extracts text and bounding boxes
3. Classification: Keyword heuristics determine doc type
4. Embedding: SentenceTransformer generates 384-dim vectors
5. Extraction: VLM extracts structured fields from receipts
6. Radar: Background crawler extracts future dates and deadlines

Uses FastAPI BackgroundTasks and asyncio.gather (no Celery/Redis needed).

## Environment Variables

- DATABASE_URL - Aurora PostgreSQL connection string
- AZURE_OPENAI_ENDPOINT - Azure OpenAI endpoint
- AZURE_OPENAI_KEY - Azure OpenAI API key
- AZURE_OPENAI_DEPLOYMENT - Deployment name (default: gpt-4.1)
- AWS_S3_BUCKET - S3 bucket for images
- COGNITO_POOL_ID - Cognito user pool ID
- COGNITO_REGION - Cognito region (default: us-west-2)
- DEV_MODE - Set true to allow dev_ token bypass
- MAX_CONCURRENT_OCR - Parallel OCR limit (default: 2)

## Troubleshooting

Container won't start:
```bash
docker logs lifeledgercapstone_app_1
```

OCR timeout on first request:
- PaddleOCR pre-warms on startup - wait for "PaddleOCR model loaded" in logs

Database connection issues:
- Verify DATABASE_URL in .env
- Check EC2 security group allows Aurora access on port 5432

## Team

UC Berkeley MIDS Capstone - Spring 2025

Benjamin He, Daniel Wang, Jiayi Ding, Viola Qiu, Umesh Kant
