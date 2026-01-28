**Target outcome:** You can _design, justify, and build_ reliable agentic systems—not just assemble demos.

---

## Week 1 — Mental model reset: agents are distributed systems

### Reading (primary)

**Designing Data-Intensive Applications (DDIA)**  
Focus chapters:

- Ch 1: Reliable, Scalable, Maintainable Applications
    
- Ch 2: Data Models & Query Languages
    
- Ch 8: The Trouble with Distributed Systems
    

**Key concepts to extract**

- State ≠ memory prompt
    
- Side effects must be controlled
    
- Failure is normal, not exceptional
    

### Implementation

Build a **single-agent tool runner** with:

- Explicit state object (JSON)
    
- Idempotent tool calls
    
- Retry-safe execution
    

**Rules**

- No memory embeddings yet
    
- No multi-agent
    
- Persist state between runs
    

**Deliverable**

- A repo with:
    
    - `state.json`
        
    - tool execution log
        
    - retry handling
        

---

## Week 2 — Determinism, retries, and idempotency

### Reading

**DDIA**

- Ch 7: Transactions
    
- Ch 11: Stream Processing (selective)
    

**Key concepts**

- Exactly-once is mostly a lie
    
- Idempotency is the real goal
    
- Ordering matters more than intelligence
    

### Implementation

Upgrade last week’s agent:

- Add:
    
    - execution IDs
        
    - deduplication logic
        
    - retry counters
        
- Simulate failures (timeouts, partial tool success)
    

**Deliverable**

- Demonstrate:
    
    - Same request run twice → same outcome
        
    - Tool failure → safe retry
        

---

## Week 3 — Architectural tradeoffs in agent systems

### Reading

**Software Architecture: The Hard Parts**  
Focus:

- Architecture tradeoffs
    
- Fitness functions
    
- Distributed workflows
    

**Key concepts**

- “More agents” is usually worse
    
- Architecture is about constraints, not cleverness
    

### Implementation

Refactor into **Planner → Executor**

- Planner: produces structured plan
    
- Executor: executes step-by-step
    
- Add a “fitness function”:
    
    - max steps
        
    - max cost
        
    - max retries
        

**Deliverable**

- Architecture diagram
    
- Written justification of:
    
    - Why planner/executor split exists
        
    - Failure boundaries
        

---

## Week 4 — Multi-agent ≠ many LLMs

### Reading

**Distributed Systems – Tanenbaum**  
Focus:

- Coordination
    
- Partial failure
    
- Consensus (conceptual only)
    

Optional:  
**Patterns of Enterprise Application Architecture**

- Process Manager pattern
    

### Key concepts

- Coordination cost dominates
    
- Shared memory causes implicit coupling
    
- Most “multi-agent” systems are workflow engines
    

### Implementation

Add **one more agent only**:

- Roles must be asymmetric
    
    - e.g., Planner + Verifier
        
- Explicit communication protocol
    
- No shared mutable memory
    

**Deliverable**

- Protocol definition
    
- Failure case demo (one agent fails, system survives)
    

---

## Week 5 — Decision-making under uncertainty

### Reading

**Thinking in Bets**

- Focus on probabilistic thinking
    
- Decision quality vs outcome quality
    

### Key concepts

- LLM output is a probability distribution
    
- Agents must reason about confidence
    

### Implementation

Add:

- Confidence scores to plans
    
- “Abort / Continue” decision gates
    
- Fallback strategy if confidence < threshold
    

**Deliverable**

- Agent that can _decide not to act_
    
- Logged reasoning for decisions
    

---

## Week 6 — Stopping, looping, and cost control

### Reading

**Algorithms to Live By**

- Optimal stopping
    
- Explore vs exploit
    

### Key concepts

- Infinite loops are a design failure
    
- Stopping conditions are architecture, not prompts
    

### Implementation

Add:

- Hard stop rules
    
- Cost tracking
    
- Step budget enforcement
    

**Deliverable**

- Agent that:
    
    - Stops early when sufficient
        
    - Does not over-plan
        
    - Logs cost vs outcome
        

---

## Week 7 (optional) — Read the AI agent book _properly_

### Reading

**Building Applications with AI Agents (O’Reilly)**  
Now read it **critically**, not passively.

Map:

- Their concepts → what you already built
    
- Identify oversimplifications
    

### Implementation

Refactor documentation only:

- Rewrite your system as a design doc
    
- Explicitly list:
    
    - Tradeoffs
        
    - Failure modes
        
    - Non-goals
        

---

## Week 8 (optional) — Production hardening

### Implementation only

Add:

- Observability (logs, traces)
    
- Evaluation harness
    
- Deterministic replay of runs
    

**Deliverable**

- “Why this agent is safe to run in prod” document
    

---

## Final outcome (what you will truly gain)

After this plan, you will:

- Stop thinking in “agents” and start thinking in **systems**
    
- Know when _not_ to use multi-agent
    
- Be able to defend architecture decisions
    
- Understand why most agent demos collapse in production
    

---

## Important honesty note

There is **no book** that will give you this understanding alone.  
Books give _models_. **Implementation reveals truth**.

---

If you want next, I can:

- Convert this into a **daily schedule**
    
- Tailor it to **LangGraph specifically**
    
- Design a **capstone agent project** aligned with interviews or enterprise use
    

Tell me which direction to take.