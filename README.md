# LifeLedger Backend

Backend API for document OCR, semantic search, and AI-powered analysis.

## Quick Reference

**EC2 IP**: `18.236.74.130`
**API Base URL**: `http://18.236.74.130:8000`

## Deploying Code Updates

After pushing changes to GitHub:

```bash
# SSH into EC2
ssh -i LLKey.pem ubuntu@18.236.74.130

# Pull latest code and rebuild
cd LifeLedgerCapstone
git pull origin main
docker-compose down
docker-compose up --build -d

# Watch logs to verify startup
docker logs -f lifeledgercapstone-app-1
```

## Checking Logs

```bash
# View recent logs
docker logs lifeledgercapstone-app-1

# Follow logs in real-time
docker logs -f lifeledgercapstone-app-1

# View last 100 lines
docker logs --tail 100 lifeledgercapstone-app-1
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload images (multipart form) |
| `/process` | POST | Trigger OCR for uploaded doc |
| `/documents` | GET | List user's documents |
| `/documents/{id}` | GET | Get document details + bounding boxes |
| `/search` | POST | Semantic search + AI answer |
| `/ask` | POST | AI agent for analytical questions |

### Example Requests

```bash
# Health check
curl http://18.236.74.130:8000/health

# Ask a question (with dev auth)
curl -X POST "http://18.236.74.130:8000/ask" \
  -H "Authorization: Bearer dev_testuser" \
  -H "Content-Type: application/json" \
  -d '{"question":"How much did I spend?"}'

# Get document with bounding boxes
curl "http://18.236.74.130:8000/documents/1" \
  -H "Authorization: Bearer dev_testuser"
```

## Agent Tools

The AI agent can answer analytical questions using these tools:

| Tool | Example Question |
|------|------------------|
| `search_documents` | "Find my Amazon receipts" |
| `get_total_spending` | "How much did I spend last month?" |
| `get_spending_by_merchant` | "Where do I spend the most?" |
| `get_receipts_by_merchant` | "Show me all Target receipts" |
| `get_receipts_by_date_range` | "What did I buy in January?" |

## Project Structure

```
app/
  main.py          # FastAPI endpoints
  agent.py         # AI agent with OpenAI function calling
  ocr_pipeline.py  # PaddleOCR + VLM refinement
  vlm_client.py    # Azure GPT-4.1 integration
  extraction.py    # Field extraction + query functions
  db.py            # PostgreSQL + pgvector
  s3.py            # S3 operations
  auth.py          # Cognito JWT validation
```

## Environment Variables

Edit `.env` on EC2 (`nano .env`) - key variables:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Aurora PostgreSQL connection string |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key |
| `AWS_S3_BUCKET` | S3 bucket for images |
| `DEV_MODE` | Set `true` to bypass auth for testing |

## Troubleshooting

**Container won't start:**
```bash
docker logs lifeledgercapstone-app-1
```

**Database connection issues:**
- Check `DATABASE_URL` in `.env`
- Verify EC2 security group allows Aurora access

**OCR not processing:**
- Check background task logs
- Verify S3 permissions

## Team

UC Berkeley MIDS Capstone - Spring 2025
Benjamin He, Daniel Wang, Jiayi Ding, Viola Qiu, Umesh Kant
