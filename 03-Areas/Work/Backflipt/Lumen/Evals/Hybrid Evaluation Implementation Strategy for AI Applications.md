## 1. Objective

Define the implementation approach for evaluating AI applications in a way that is extensible, scalable, and compatible with complex agentic workflows, including multi-step orchestration and custom business logic.

## 2. Chosen Implementation

We are choosing to build an **internal evaluation SDK layer** that acts as a unifying evaluation interface across AI applications.

This SDK will:

- standardize how evaluation inputs, traces, outputs, and metadata are captured,
    
- support reusable evaluation components,
    
- allow custom logic to be added without changing the core system,
    
- support both offline and online evaluation workflows,
    
- integrate with tracing, experimentation, and reporting systems.
    

## 3. Why This Implementation

A standalone SDK-based approach is the most suitable option because our evaluation needs go beyond simple response scoring.

The implementation must support:

- multi-step workflows,
    
- tool usage validation,
    
- graph/state transition correctness,
    
- business-rule-specific evaluation,
    
- reusable evaluation logic across multiple applications.
    

A lightweight SDK gives us control over these requirements while keeping the system extensible and maintainable.

## 4. Industry Standard Approach

The current industry direction is not to build a monolithic evaluation system from scratch. Instead, teams typically adopt a **hybrid evaluation architecture** with the following characteristics:

- a common internal evaluation layer,
    
- pluggable evaluators,
    
- support for model-based and rule-based scoring,
    
- custom metrics for domain logic,
    
- separation of evaluation execution from observability and reporting.
    

This pattern is considered industry-aligned because it avoids lock-in, supports fast experimentation, and allows teams to evolve evaluation methods without redesigning the full platform.

## 6. Proposed Implementation Model

### 6.1 Core SDK Responsibilities

The SDK will act as the central orchestration layer for evaluation.

Its main responsibilities will include:

- defining standard evaluation schemas,
    
- accepting application traces and outputs,
    
- routing evaluation cases to the appropriate metric handlers,
    
- collecting scores and evidence,
    
- aggregating results,
    
- publishing results to downstream systems.
    

### 6.2 Canonical Evaluation Objects

The SDK will standardize the following internal objects:

- **Evaluation Case** – the unit being evaluated,
    
- **Metric Definition** – the scoring logic contract,
    
- **Metric Result** – the score, explanation, and evidence,
    
- **Evaluation Run** – the overall execution context,
    
- **Evaluation Pack** – a grouped set of metrics for a use case,
    
- **Gate Result** – pass/fail decision for release or monitoring.
    

### 6.3 Plugin-Based Extension Model

The SDK will provide extension points for:

- custom metrics,
    
- scoring strategies,
    
- result aggregation,
    
- trace adapters,
    
- dataset loaders,
    
- output sinks.
    

This allows new logic to be introduced as plugins rather than changes to the core runtime.

## 7. Evaluation Levels

To support modern AI applications, the implementation will evaluate across multiple layers.

### 7.1 Response-Level Evaluation

Measures response quality, correctness, relevance, completeness, and safety.

### 7.2 Tool-Level Evaluation

Checks whether the correct tool was selected, whether it was invoked at the right time, and whether the arguments were valid.

### 7.3 Workflow-Level Evaluation

Evaluates whether the application followed the correct execution path and whether routing decisions were appropriate.

### 7.4 Trace-Level Evaluation

Assesses the full execution chain from input to final output, including intermediate reasoning artifacts and system actions where available.

### 7.5 Session-Level Evaluation

Measures consistency, memory handling, recovery across turns, and long-running task completion.

## 8. Custom Evaluation Logic

Custom logic is a primary requirement of this implementation.

Examples of custom evaluation areas include:

- correctness of workflow transitions,
    
- validation of state updates,
    
- tenant or policy compliance,
    
- business-rule adherence,
    
- tool parameter validation,
    
- fallback and retry behavior,
    
- domain-specific output structure checks.
    

These metrics will be implemented as independent SDK extensions.

## 9. How We Are Thinking to Implement It

## 9.1 Phase 1 – Core SDK Foundation

Build the base SDK with:

- common schemas,
    
- evaluator interfaces,
    
- metric registry,
    
- execution runner,
    
- result model,
    
- dataset support.
    

## 9.2 Phase 2 – Workflow Integration

Integrate application traces and workflow execution data into the SDK so that evaluations can run on complete execution units rather than only final responses.

## 9.3 Phase 3 – Built-In Metric Packs

Create reusable evaluation packs for:

- response quality,
    
- workflow correctness,
    
- tool accuracy,
    
- retrieval quality,
    
- safety and policy adherence.
    

## 9.4 Phase 4 – Result Aggregation and Gating

Add score aggregation, thresholding, weighted scoring, and pass/fail gating to support regression testing and release validation.

## 9.5 Phase 5 – Reporting and Observability

Send evaluation results to dashboards, experiment tracking systems, or observability platforms for comparison, auditability, and trend analysis.

## 10. Target Operating Model

The SDK will support the following operating modes:

### 10.1 Offline Evaluation

Used for:

- regression testing,
    
- dataset-based benchmark runs,
    
- pre-release validation,
    
- comparison of prompt or workflow changes.
    

### 10.2 Online Evaluation

Used for:

- production monitoring,
    
- sampling-based quality review,
    
- drift detection,
    
- live experiment scoring.
    

### 10.3 Human-in-the-Loop Review

Used for:

- validating edge cases,
    
- adjudicating ambiguous results,
    
- calibrating scoring rubrics,
    
- improving custom metrics over time.
    

## 11. Benefits of This Approach

This implementation provides:

- extensibility,
    
- maintainability,
    
- support for complex workflows,
    
- reduced vendor lock-in,
    
- reusable evaluation assets,
    
- compatibility with future evaluation methods,
    
- alignment with current industry architecture patterns.
    

## 12. Risks and Mitigations

### Risk: Over-engineering too early

**Mitigation:** Start with a thin SDK and expand only where reusable patterns emerge.

### Risk: Metric inconsistency across teams

**Mitigation:** Use a centralized metric registry and shared evaluation contracts.

### Risk: Tight coupling with application internals

**Mitigation:** Introduce adapters that translate application traces into a standard SDK schema.

### Risk: Hard-to-interpret results

**Mitigation:** Require every metric result to include explanation and evidence where possible.

## 13. Final Recommendation

We should implement a **hybrid evaluation SDK** as the internal standard for AI application evaluation.

This approach is aligned with current industry practice, supports extensibility through clear interfaces, and is well suited for complex workflow-driven applications where standard response-level metrics alone are insufficient.

## References

- [https://www.confident-ai.com/docs/llm-evaluation/quickstart](https://www.confident-ai.com/docs/llm-evaluation/quickstart?utm_source=chatgpt.com)
    
- [https://docs.ragas.io/en/stable/](https://docs.ragas.io/en/stable/?utm_source=chatgpt.com)
    
- [https://docs.ragas.io/en/v0.3.8/howtos/customizations/metrics/_write_your_own_metric/](https://docs.ragas.io/en/v0.3.8/howtos/customizations/metrics/_write_your_own_metric/?utm_source=chatgpt.com)
    
- [https://arize.com/docs/phoenix](https://arize.com/docs/phoenix?utm_source=chatgpt.com)
    
- [https://arize.com/docs/phoenix/evaluation/llm-evals](https://arize.com/docs/phoenix/evaluation/llm-evals?utm_source=chatgpt.com)
    
- [https://inspect.aisi.org.uk/extensions.html](https://inspect.aisi.org.uk/extensions.html?utm_source=chatgpt.com)
    
- [https://inspect.aisi.org.uk/scorers.html](https://inspect.aisi.org.uk/scorers.html?utm_source=chatgpt.com)
    
- [https://mlflow.org/docs/latest/genai/eval-monitor/](https://mlflow.org/docs/latest/genai/eval-monitor/?utm_source=chatgpt.com)
    
- [https://www.trulens.org/getting_started/core_concepts/feedback_functions/](https://www.trulens.org/getting_started/core_concepts/feedback_functions/?utm_source=chatgpt.com)
    
- [https://langfuse.com/docs](https://langfuse.com/docs?utm_source=chatgpt.com)