# 6-Month Intense Plan 

**Goal:** Become a _research-inclined applied AI engineer_ + _AI infra/systems engineer_ at the top 0.001% skill percentile.  
**Mindset:** Zero fluff. This is a _daily execution plan_ requiring 3–5 hours/day weekdays, 6–8 hours/day weekends.

The plan is structured into **phases**, each with **output-driven milestones** that prove competence.

---

# Month 1: Mathematical Foundations + Core DL Engineering

**Objective:** Build the foundation to understand, modify and debug transformer internals.

### Skills

1. **Math for ML**  
    • Linear algebra: SVD, eigen decomposition, projections  
    • Probability: random variables, KL-divergence, cross-entropy  
    • Optimization: SGD, momentum, Adam, LR schedules
    
2. **Deep Learning Fundamentals**  
    • Backpropagation: derive gradients manually  
    • Initialization, normalization, regularization  
    • Vanishing/exploding gradients
    
3. **PyTorch Engineering**  
    • Autograd mechanics  
    • Writing custom nn.Modules  
    • Torch.compile basics  
    • vectorization (einsum, broadcasting)
    

### Milestone Projects

• Implement a **2-layer neural network** from scratch (NumPy only)  
• Implement **manual backprop** for that network  
• Replicate MNIST training with PyTorch (no cheats)  
• Write a blog: “Understanding Backprop Without Autograd”

---

# Month 2: Transformers From Scratch + GPU Awareness

**Objective:** Be capable of implementing transformer components by hand.

### Skills

1. **Attention & Transformer Internals**  
    • Scaled dot-product attention  
    • Multi-head attention  
    • Position encodings  
    • LayerNorm & residuals  
    • Causal masking
    
2. **Implement Core Components**  
    • Complete transformer block in PyTorch  
    • Training loop with mixed precision  
    • Basic logging/profiling (torch.profiler)
    
3. **GPU Fundamentals**  
    • fp16 vs bf16  
    • Memory bottlenecks  
    • Using CUDA kernels (high-level understanding)
    

### Milestone Projects

• Implement **scaled dot-product attention using only matmul & softmax**  
• Build a **mini GPT-2** (1–5M params) and train on character-level data  
• Implement a tokenizer from scratch (BPE or WordPiece)  
• Blog: “Building a Minimal Transformer That Actually Works”

---

# Month 3: Model Serving, Optimization, vLLM, Inference Systems

**Objective:** Become elite in **serving** and **optimizing** LLMs at scale.

### Skills

1. **LLM Serving Foundations**  
    • KV cache  
    • Batch scheduling  
    • Token parallelism  
    • Streaming architectures  
    • Speculative decoding
    
2. **Serving Tools**  
    • vLLM internals  
    • TensorRT-LLM  
    • HF Text Generation Inference  
    • ONNX Runtime
    
3. **Performance Profiling**  
    • Throughput vs latency  
    • Measuring token/s  
    • Profiling per-request overhead
    

### Milestone Projects

• Build a **mini vLLM-like serving engine**:  
– KV cache reuse  
– batching  
– streaming tokens  
• Benchmark it vs naive PyTorch inference  
• Blog: “How KV Cache Reuse Reduces Latency by X%”

---

# Month 4: Retrieval, Vector DB Internals, Distributed Systems

**Objective:** Go beyond RAG consumers and become a RAG **engineer**.

### Skills

1. **Vector DB Internals**  
    • HNSW  
    • IVF  
    • Product quantization  
    • Re-ranking (ColBERT, SPLADE basics)
    
2. **Distributed Retrieval Systems**  
    • Sharding  
    • Replication  
    • Consistency  
    • Hybrid lexical + dense search
    
3. **Systems Design**  
    • Message queues (Kafka/SQS)  
    • Worker pools  
    • Backpressure & retries  
    • Observability (traces, logs, metrics)
    

### Milestone Projects

• Implement a **tiny HNSW index** in Python  
• Build your own **retriever + reranker** pipeline  
• Build a **distributed retrieval service** using:  
– FastAPI  
– Celery/RabbitMQ or Kafka  
– Redis caching  
• Blog: “Building and Scaling a Custom RAG Engine”

---

# Month 5: Fine-Tuning, RLHF, Evals (Research + Applied Fusion)

**Objective:** Gain the ability to **train**, **fine-tune**, **evaluate**, and **improve** models.

### Skills

1. **Fine-tuning**  
    • LoRA & QLoRA  
    • Dataset curation  
    • Tokenizer alignment  
    • Grad accumulation  
    • Mixed precision training
    
2. **RLHF / RLAIF**  
    • Reward modeling  
    • Pairwise preference learning  
    • Sampling & scoring  
    • Reinforcement learning loops
    
3. **Evals Engineering**  
    • Designing golden datasets  
    • Functional evals (SQL, coding tasks)  
    • Behavioral evals (toxicity, jailbreaks)  
    • Regression testing
    

### Milestone Projects

• Fine-tune a 3B model for a **domain-specific task**  
• Build a **micro RLHF loop** (using preference pairs)  
• Build an **automatic eval suite**  
• Blog: “How I Fine-Tuned and Evaluated a Domain Model End-to-End”

---

# Month 6: Advanced Systems + Thesis Project (Portfolio Definer)

**Objective:** Create a **flagship, publishable project** demonstrating research + scaling skill.

### Choose **one** of these capstone paths:

## Option 1: **Build a Fully Custom Model Serving Stack**

Features:  
• custom batching  
• speculative decoding  
• KV cache reuse  
• Triton kernel optimization (bonus)  
• retrieval + caching layer  
• multi-model routing  
Outcome:  
This is Devin/Cursor/Perplexity-tier infra work.

## Option 2: **Train a Specialist Model End-to-End**

• Train a 1–3B model on a niche domain (e.g., SQL, code, medical, finance)  
• Add LoRA adapters  
• Add RLHF loop  
• Build full eval pipeline  
Outcome:  
This is the profile of elite applied researchers.

## Option 3: **Design a Search + Retrieval System at Scale**

• custom vector DB backend  
• multi-stage retrieval  
• re-ranking  
• streaming generation  
• caching layers  
Outcome:  
This is Perplexity-tier retrieval engineering.

### Final Deliverables (Non-negotiable)

• 2–3 technical blog posts  
• 1 large GitHub repository (high depth)  
• Latency charts, eval charts, diagrams  
• A complete demo (video + README)  
• A research-style writeup describing architecture decisions

This becomes your signature “A+B portfolio”.

---

# Weekly Time Allocation

**Weekdays (3–5 hours/day)**  
• 1 hour: math + theory  
• 2 hours: code / implementation  
• 1 hour: papers / notes  
• 30 mins: documentation & logging progress

**Weekends (6–8 hours/day)**  
• 3–4 hours: project building  
• 2 hours: profiling & debugging  
• 1–2 hours: paper reading + blog writing