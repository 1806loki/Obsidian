## Runtime evaluation of agents: a full landscape view

The industry has converged on a clear mental model here. "Runtime eval" splits into two distinct problems that are often conflated:

**Online eval** — scoring/evaluating agent behavior on live production traffic, asynchronously or in near-real-time. This is what most people mean when they say runtime eval.

**Hooks/middleware** — intercepting the agent execution loop to capture intermediate state (tool picks, params, model responses) _without_ persisting to a DB first. This is the data-capture layer that feeds online eval.


A well-structured trace includes spans for tool calls, retrievals, LLM generations, and session-level multi-turn context. Distributed tracing is the backbone — it captures the complete lifecycle of a request as it traverses model calls, external tools, and microservices. [Maxim Articles](https://www.getmaxim.ai/articles/llm-observability-best-practices-for-2025/) The standard pattern looks like:

1. **Instrument** — decorate agent components to emit spans (OpenTelemetry or proprietary)
2. **Capture** — intercept tool calls, model I/O, and parameters mid-flight
3. **Score** — run LLM-as-judge or deterministic metrics asynchronously on the captured trace
4. **Alert/route** — flag regressions, surface failures to a review queue


![[Pasted image 20260318153445.png |500]]
### Recommended architecture for your situation

Given that you're on LangGraph and want runtime eval without DB round-trips:

**Option A (recommended if you stay LangGraph-native):** Use `after_model` / `after_agent` hooks to capture tool calls and model responses → push to a lightweight async queue → run DeepEval metrics (ToolCorrectnessMetric, ArgumentCorrectnessMetric) in a worker process → pipe scores to Langfuse or LangSmith for dashboards. This keeps your agent loop fast and your eval pipeline decoupled.

**Option B (if you want the least integration work):** DeepEval integrates with LangGraph via a callback handler, supports both single-turn and multi-turn conversational evaluation for agentic systems, and provides detailed tracing from API calls through to retrieval to help pinpoint where things went wrong. [Comet](https://www.comet.com/site/blog/llm-observability-tools/) Drop in the LangGraph callback handler, add `@observe` decorators on your tool functions, and let Confident AI handle scoring async in the cloud.

**Option C (if you want self-hosted and vendor-neutral):** Langfuse is fully open-source under MIT, self-hostable without restrictions, and OpenTelemetry-compatible for piping traces into existing infrastructure. [Braintrust](https://www.braintrust.dev/articles/best-ai-observability-platforms-2025) Instrument with OTel, self-host Langfuse, run LLM-as-judge scoring via the Langfuse SDK's `score_current_trace` after each agent run.

---

### Bottom line

The LangChain hooks (`before_model`, `after_model`) are a **solid foundation for data capture** — use them to intercept tool picks and model outputs in-flight. But they're not a substitute for an eval framework. Pair them with DeepEval (if you want rich agent-specific metrics and a managed platform) or Langfuse (if you want open-source and self-hosted). The pattern that works at scale is: hooks → async queue → eval worker → observability dashboard, not hooks → synchronous eval → block the response.