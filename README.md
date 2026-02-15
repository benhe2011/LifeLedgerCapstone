# LifeLedger Backend

Backend API for LifeLedger - a privacy-conscious app that turns screenshots and photos into a searchable knowledge base with OCR, field extraction, semantic search, and AI-powered analysis.

## Features

- **OCR Pipeline**: PaddleOCR for text extraction with VLM refinement for low-confidence results
- **Document Classification**: Auto-detect receipts, subscriptions, warranties from text
- **Field Extraction**: Extract merchant, date, total amount from receipts via GPT-4.1 vision
- **Semantic Search**: pgvector embeddings for similarity search
- **AI Agent**: OpenAI function calling for analytical questions ("How much did I spend at Target?")
- **Privacy-First**: Azure OpenAI enterprise guarantees (data not used for training)

## Architecture

```
Frontend (Vercel)  →  FastAPI Backend (EC2)  →  Azure GPT-4.1
                            ↓
              Aurora PostgreSQL + pgvector
                      + AWS S3
```

| Component | Technology |
|-----------|------------|
| API Server | FastAPI |
| OCR | PaddleOCR (CPU) |
| VLM | Azure OpenAI GPT-4.1 |
| Database | Aurora PostgreSQL + pgvector |
| Storage | AWS S3 |
| Auth | AWS Cognito |

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
# {"status":"healthy","service":"lifeledger-api"}
```

### Upload Images
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Authorization: Bearer <token>" \
  -F "files=@receipt.jpg"
# {"uploaded":[{"doc_id":1,"s3_key":"user123/1.jpg"}],"count":1}
```

### Process Document (trigger OCR)
```bash
curl -X POST "http://localhost:8000/process" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"s3_key":"user123/1.jpg","row_id":"1"}'
# {"status":"processing","message":"OCR processing started in background"}
```

### List Documents
```bash
curl "http://localhost:8000/documents" \
  -H "Authorization: Bearer <token>"
# [{"id":"1","type":"Receipt","status":"Done","primaryEntity":"Target",...}]
```

### Get Document Details (includes bounding boxes)
```bash
curl "http://localhost:8000/documents/1" \
  -H "Authorization: Bearer <token>"
# {"id":"1","fileUrl":"...","ocr_blocks":[{"text":"Target","bbox":[[10,20],...]}],...}
```

### Search Documents
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query":"groceries","limit":10}'
# {"answer":"Found 3 receipts...","documents":[...],"query":"groceries"}
```

### Ask Agent (analytical questions)
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question":"How much did I spend last month?"}'
# {"answer":"You spent $245.50 across 8 receipts...","sources":["1","3","5"]}
```

## Agent Capabilities

The AI agent uses OpenAI function calling with 5 tools:

| Tool | Description | Example Question |
|------|-------------|------------------|
| `search_documents` | Semantic search | "Find my Amazon receipts" |
| `get_total_spending` | Sum spending | "How much did I spend last month?" |
| `get_spending_by_merchant` | Breakdown by store | "Where do I spend the most?" |
| `get_receipts_by_merchant` | List receipts from store | "Show me all Target receipts" |
| `get_receipts_by_date_range` | List by timeframe | "What did I buy in January?" |

## Environment Variables

```bash
# Database (direct connection for backend)
DATABASE_URL=postgresql://user:pass@aurora-endpoint:5432/lifeledger

# Azure OpenAI (GPT-4.1 VLM)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4.1

# AWS
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-west-2
AWS_S3_BUCKET=lifeledger-images

# Auth
COGNITO_POOL_ID=us-west-2_xxxxxxxxx
COGNITO_REGION=us-west-2

# CORS
CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000

# Development (bypasses auth for testing)
DEV_MODE=false
```

## Local Development

```bash
# Clone repo
git clone <repo-url>
cd LifeLedgerCapstone

# Setup environment
cp .env.example .env
# Edit .env with your credentials

# Run with Docker
docker-compose up --build

# Test
curl http://localhost:8000/health
```

## Deploy to EC2

### Prerequisites
- EC2 instance (t3.large recommended for PaddleOCR)
- Aurora PostgreSQL cluster with pgvector extension
- S3 bucket for images
- Azure OpenAI resource with GPT-4.1 deployment

### Step 1: Setup EC2
```bash
ssh -i lifeledger-key.pem ubuntu@<EC2_IP>

# Install Docker
sudo apt update && sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
# Log out and back in
```

### Step 2: Clone and Configure
```bash
git clone <repo-url>
cd LifeLedgerCapstone
cp .env.example .env
nano .env  # Fill in production credentials
```

### Step 3: Database Setup

Create tables in Aurora PostgreSQL:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  user_id TEXT NOT NULL,
  s3_key TEXT NOT NULL,
  doc_text TEXT,
  text_vector VECTOR(384),
  doc_type TEXT DEFAULT 'unknown',
  ocr_blocks JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE extractions (
  id SERIAL PRIMARY KEY,
  doc_id INTEGER REFERENCES documents(id),
  doc_type TEXT,
  merchant TEXT,
  date DATE,
  total_amount DECIMAL,
  address TEXT
);

CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_vector ON documents USING ivfflat (text_vector vector_cosine_ops);
```

### Step 4: Build and Run
```bash
docker-compose up --build -d
docker logs -f lifeledgercapstone-app-1
```

### Step 5: Verify
```bash
# Health check
curl http://<EC2_IP>:8000/health

# Test with dev auth (if DEV_MODE=true)
curl -X POST "http://<EC2_IP>:8000/ask" \
  -H "Authorization: Bearer dev_testuser" \
  -H "Content-Type: application/json" \
  -d '{"question":"How much did I spend?"}'
```

## Project Structure

```
app/
  main.py          # FastAPI endpoints
  ocr_pipeline.py  # PaddleOCR + VLM refinement
  vlm_client.py    # Azure GPT-4.1 integration
  extraction.py    # Field extraction + query functions
  agent.py         # OpenAI function calling agent
  auth.py          # Cognito JWT validation
  db.py            # PostgreSQL + pgvector + embeddings
  s3.py            # S3 upload/presigned URLs
```

## Privacy & Security

- **Azure OpenAI Enterprise**: Data not used for training, not shared with OpenAI
- **S3 SSE**: Encryption at rest
- **HTTPS**: All traffic encrypted in transit
- **Row-level isolation**: All queries filter by user_id
- **JWT validation**: Cognito tokens required for all endpoints

## Team

UC Berkeley MIDS Capstone - Spring 2025
- Benjamin, Daniel, Viola, Jiayi, Umesh
