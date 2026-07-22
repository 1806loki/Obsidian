Insights till now
- OTel and @trace_graph are complementary

Implementation ideas
 Framework (standard metrics)+ Custom Logic = end-to-end Agent testing
 
 Input query
      ↓
  Agent runs (via @trace_graph — captures tool calls, agent hops, params, order)
      ↓
  Two evaluation planes:
      ├── Output quality → framework standard metrics (LLM-as-judge)
      └── Behavioral trace → custom TraceComparator logic
       ↓
   Single unified score + diff report

Questions


## What “industry-standard metrics” usually means in practice

[Inference] For agentic AI systems, teams usually evaluate across these buckets rather than relying on one single score:

**Task success**

- final answer correctness
    
- goal completion
    
- exact match / semantic match against expected outcome
    

**Reasoning quality**

- factual correctness
    
- faithfulness to provided evidence
    
- instruction adherence
    

**Tool-use quality**

- correct tool chosen
    
- correct parameter extraction
    
- correct tool sequence
    
- unnecessary tool-call rate
    

**Retrieval quality**  
for RAG or agent + KB systems:

- context precision
    
- context recall
    
- answer relevancy
    
- faithfulness
    

**Operational quality**

- latency
    
- token usage
    
- cost
    
- error rate
    
- retry rate
    
- timeout rate
    

**Safety / governance**

- toxicity
    
- prompt-injection resistance
    
- hallucination / unsupported claims
    
- policy violations
    
- data leakage / secrets exposure  
    DeepEval explicitly advertises red teaming and security/safety scanning capabilities.'

Build your evaluation suite in 5 groups:

1. **Golden task tests**
    
    - expected answer / action known
        
    - exact or semantic correctness
        
2. **Workflow tests**
    
    - tool order
        
    - tool arguments
        
    - branch logic
        
    - retries / fallback behavior
        
3. **RAG quality tests**
    
    - faithfulness
        
    - context precision / recall
        
    - answer relevancy
        
4. **Safety tests**
    
    - prompt injection
        
    - jailbreak attempts
        
    - sensitive data leakage
        
    - forbidden actions
        
5. **Operational tests**
    
    - latency budget
        
    - token budget
        
    - cost budget
        
    - flaky test detection