# Price Is Right Agent 🧠💰

A production-style multi-agent AI system that autonomously discovers online deals, estimates fair market value using an ensemble of pricing models, and surfaces statistically significant discount opportunities.

## 🎯 Problem Motivation

Online deals are noisy, inconsistent, and often misleading.

The challenge is not discovering deals — it's determining whether a deal is genuinely underpriced relative to market value.

This project explores how multi-agent AI systems, RAG, and ensemble modeling can be combined to build a more reliable pricing intelligence pipeline.

This system combines:

- 🔎 RSS-based deal discovery
- 🧠 LLM-based structured filtering
- 📚 RAG over a product vector database
- 🧮 Fine-tuned pricing model (remote, Modal)
- 🧠 Deep neural network regression model (optional)
- 🤝 Ensemble aggregation
- 💾 Persistent memory tracking
- 📊 Embedding visualization via TSNE

## 🏗 System Architecture

The system is built using a coordinated multi-agent architecture:

```
PlanningAgent
├── ScannerAgent → Structured LLM filtering of RSS deals
├── EnsembleAgent
│   ├── Preprocessor (Local LLM via Ollama)
│   ├── SpecialistAgent (Fine-tuned model deployed on Modal)
│   ├── FrontierAgent (RAG + GPT reasoning over similar products)
│   └── NeuralNetworkAgent (Local regression model, optional)
└── MessagingAgent → Alert when discount exceeds threshold
```

**Vector Store:** ChromaDB  
**Embeddings:** SentenceTransformers (all-MiniLM-L6-v2)  
**Frontier LLM:** OpenAI GPT-4o-mini  
**Local LLM:** LLaMA3 via Ollama

## 🧩 Why This Tech Stack?

**ChromaDB** – Lightweight local vector database enabling fast similarity search and persistent embeddings without external infrastructure.

**SentenceTransformers (all-MiniLM-L6-v2)** – Efficient embedding model optimized for semantic similarity tasks with low latency.

**OpenAI GPT-4o-mini** – Used for frontier reasoning with contextual grounding from retrieved products.

**Modal** – Enables remote deployment of fine-tuned models without managing dedicated GPU infrastructure.

**Ollama (LLaMA3)** – Provides cost-efficient local LLM preprocessing to reduce downstream token usage.

**PyTorch Neural Network** – Optional lightweight regression model to introduce statistical diversity into ensemble predictions.

## 🔄 Workflow

1. ScannerAgent fetches RSS deals.
2. A structured LLM filtering stage selects the top 5 high-quality product descriptions.
3. For each product:
   - Preprocess description via local LLM.
   - Query vector DB for similar products (RAG).
   - Estimate price using:
     - Fine-tuned LLM (remote)
     - GPT reasoning over similar products
     - Neural network regression (optional)
4. EnsembleAgent aggregates predictions.
5. PlanningAgent calculates discount.
6. MessagingAgent triggers alert if discount exceeds threshold.

## 🧮 Ensemble Strategy

The current aggregation strategy uses a simple averaging mechanism, but the architecture allows for weighted or confidence-based aggregation.

Why ensemble?

- Reduces variance from LLM hallucinations
- Mitigates bias from any single model
- Improves robustness on noisy product descriptions
- Allows graceful degradation if one model fails

If the neural network weights are missing, the system automatically skips that model.

### 🔍 Example Pricing Decision

For a sample product:

- **Specialist Model (fine-tuned LLM):** $700
- **Frontier Model (RAG + GPT reasoning):** $1200
- **Neural Network (regression):** $285

**Final Ensemble Estimate (simple average):**

(700 + 1200 + 285) / 3 ≈ 1057

If the deal price is $350, the system computes:

**Discount ≈ $707**

This illustrates how ensemble aggregation mitigates extreme predictions from any single model while maintaining responsiveness to contextual pricing signals.

## 📚 RAG Design

FrontierAgent performs:

1. Embedding of product description
2. Vector similarity search in ChromaDB
3. Retrieval of top 5 similar products
4. GPT reasoning with contextual grounding

This constrains LLM price estimation by grounding predictions in semantically similar historical product data.

## 🧠 Fine-Tuned Model

SpecialistAgent calls a fine-tuned LLM deployed remotely on Modal.

Benefits:

- Domain-specific pricing behavior
- Lower variance vs generic LLM
- Decoupled deployment architecture

## 💾 Memory System

The system persists surfaced opportunities in `memory.json`.

This prevents re-alerting on the same deal and enables historical tracking.

## 📊 Embedding Visualization

TSNE projection of product embeddings is available for exploration and debugging.

## 🚀 Running the System

### Install dependencies

```
pip install -r requirements.txt
```

### Set environment variables

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

### Run

```
python deal_agent_framework.py
```

## 🛠 Engineering Tradeoffs

- ChromaDB chosen for local vector persistence
- Structured Outputs used to enforce JSON schema compliance
- Neural network optional to avoid large model distribution
- Local LLM preprocessing reduces token cost for frontier reasoning
- Threshold-based alerting avoids noisy notifications

## ⚠ Failure Handling & Robustness

- Neural network model is optional and automatically skipped if weights are unavailable.
- Ensemble aggregation mitigates single-model instability.
- Structured Outputs enforce schema validation.
- Discount threshold prevents noisy alerts.
- Persistent memory avoids duplicate deal notifications.

## 💰 Cost-Aware Design

- Local LLM preprocessing reduces downstream token usage.
- RAG constrains GPT reasoning to relevant product context.
- Optional neural network model runs locally with zero API cost.
- Threshold-based filtering prevents unnecessary remote model calls.

## 🎯 Key Design Principles

- Modular agents with single responsibility
- Graceful failure handling
- Cost-aware LLM usage
- Production-style logging
- Reproducible local setup
- Optional heavy components

## 📌 Future Improvements

- Async execution for parallel model calls
- Weighted ensemble learning
- Model confidence scoring
- Deal category-specific pricing models
- Automated retraining pipeline
