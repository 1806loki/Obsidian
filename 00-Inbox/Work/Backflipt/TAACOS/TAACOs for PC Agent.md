- [1. Objective](https://backflipt1.atlassian.net/wiki/spaces/TransferIQ/pages/1942028306/PC+Agent+Evaluation#1.-Objective)
- [2. Data Requirements](https://backflipt1.atlassian.net/wiki/spaces/TransferIQ/pages/1942028306/PC+Agent+Evaluation#2.-Data-Requirements)
- [3. Methods to Retrieve Tool Call Details](https://backflipt1.atlassian.net/wiki/spaces/TransferIQ/pages/1942028306/PC+Agent+Evaluation#3.-Methods-to-Retrieve-Tool-Call-Details)
    - [3.1 Langfuse](https://backflipt1.atlassian.net/wiki/spaces/TransferIQ/pages/1942028306/PC+Agent+Evaluation#3.1-Langfuse)
    - [3.2 Database (LangGraph Checkpoints)](https://backflipt1.atlassian.net/wiki/spaces/TransferIQ/pages/1942028306/PC+Agent+Evaluation#3.2-Database-\(LangGraph-Checkpoints\))
    - [3.3 Custom Function](https://backflipt1.atlassian.net/wiki/spaces/TransferIQ/pages/1942028306/PC+Agent+Evaluation#3.3-Custom-Function)
- [4. Pros & Cons Comparison](https://backflipt1.atlassian.net/wiki/spaces/TransferIQ/pages/1942028306/PC+Agent+Evaluation#4.-Pros-%26-Cons-Comparison)
- [5. Recommended Approach](https://backflipt1.atlassian.net/wiki/spaces/TransferIQ/pages/1942028306/PC+Agent+Evaluation#5.-Recommended-Approach)
- [6. Required Changes in TAACOs](https://backflipt1.atlassian.net/wiki/spaces/TransferIQ/pages/1942028306/PC+Agent+Evaluation#6.-Required-Changes-in-TAACOs)
- [7. References](https://backflipt1.atlassian.net/wiki/spaces/TransferIQ/pages/1942028306/PC+Agent+Evaluation#7.-References)

## **1. Objective**

Evaluate the accuracy of the PC Agent by analyzing:

- Tools invoked during execution
    
- Order in which tools were executed
    

---

## **2. Data Requirements**

- Full list of tools invoked
    
- Execution sequence of tool calls
    

---

## **3. Methods to Retrieve Tool Call Details**

### **3.1 Langfuse**

Retrieve execution traces from Langfuse by filtering with a **question_id**. Langfuse logs granular spans for each tool invocation, allowing precise inspection of tool activity.

**Use Cases**

- Reviewing tool behavior for specific questions
    
- Visualizing tool execution timelines
    
- Debugging unexpected routing or multi-tool flows
    

---

### **3.2 Database (LangGraph Checkpoints)**

Query and decrypt LangGraph checkpoint documents associated with a **question_id**. These checkpoints contain the complete execution snapshot, including tool call history.

**Use Cases**

- Extracting end-to-end tool call sequences
    
- Automated evaluation and scoring flows
    
- Maintaining reproducible internal execution records
    

---

### **3.3 Custom Function**

Capture tool calls programmatically during execution and attach them to the final output. Implemented directly in the agent logic.

**Use Cases**

- Quick diagnostic checks during development
    
- Lightweight, local validation
    
- Real-time visibility during agent iteration
    

---

## **4. Pros & Cons Comparison**

|   |   |   |
|---|---|---|
|**Method**|**Pros**|**Cons**|
|**Langfuse**|- Centralized structured tracing.<br>    <br>- Visual inspection of execution flow.<br>    <br>- Filterable using `question_id`.<br>    <br>- Minimal code changes after integration|- External dependency (service uptime, latency).<br>    <br>- Not widely adopted across teams.<br>    <br>- Requires explicit instrumentation.<br>    <br>- Retention and data format managed externally.|
|**Database (Checkpoints)**|- Native to LangGraph → automatically generated.<br>    <br>- High usage across teams.<br>    <br>- Fully internal and secure.<br>    <br>- Deterministic, consistent execution metadata.<br>    <br>- No external APIs or services required.|- Checkpoint payloads require decryption.<br>    <br>- Requires knowledge of checkpoint schema.<br>    <br>- Data accuracy depends on what the graph logs.|
|**Custom Function**|- Maximum flexibility.<br>    <br>- Real-time capture.<br>    <br>- No external services needed.|- No persistence unless stored manually.<br>    <br>- High maintenance overhead.<br>    <br>- Easy to miss unwrapped tool calls.<br>    <br>- No visualization.<br>    <br>- Not suitable for production-scale evals.|

---

## **5. Recommended Approach**

Use LangGraph checkpoint documents as the primary mechanism to extract tool call data and evaluate PC Agent accuracy.

**Key Reasons**

- High adoption across teams; no additional setup required
    
- Automatically generated for every LangGraph execution
    
- Reliable, internal source of execution metadata
    
- No external dependencies or service availability considerations
    
- Best suited for automated evaluation and scoring workflows
    

**Notes on Other Methods**

- **Langfuse** → Useful for debugging and visualization; not consistently adopted across teams
    
- **Custom Function** → Lacks persistence and not suitable for scalable, production-level evaluation
    

---

## **6. Required Changes in TAACOs**

- Migrate from **Motor** to **PyMongo Async (4.x)**
    
- Add or configure connection pooling (`maxPoolSize`)
    
- Assign a unique **question_id** for each execution
    
- Retrieve, decode, and extract `tool_calls` from checkpoints
    
- Compute evaluation scores using extracted data
    

---

## **7. References**

- **PyMongo Migration Guide:**  
    [Migrate to PyMongo Async - PyMongo Driver - MongoDB Docs](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/reference/migration/)
    
- **PyMongo Connection Pooling:**  
    [pool – Pool module for use with a MongoDB client. - PyMongo 4.15.4 documentation](https://pymongo.readthedocs.io/en/stable/api/pymongo/pool.html)
    
- **LangGraph Persistence (Checkpoints):**  
    [Persistence - Docs by LangChain](https://docs.langchain.com/oss/javascript/langgraph/persistence#checkpoints)