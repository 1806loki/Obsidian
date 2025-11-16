#FileView 

Sentinel AI Analytics is your AI assistant that monitors file transfers, identifies failures, provides troubleshooting solutions, and sends alerts to keep your Axway Secure Transport operations running smoothly.

It is a sophisticated FastAPI-based microservices architecture that implements a multi-agent LLM orchestration system for file transfer analytics and troubleshooting. The application leverages LangGraph's conversation orchestration graph pattern with specialized AI agents.

 
1. **Router** - Intelligently routes user queries to the most appropriate specialized agent based on query intent and content analysis.

2. **DefaultAgent** - Handles greetings, casual interactions, system capability questions, and provides general information about the application.

3. **SQLQueryGenerationAgent** - Converts natural language questions into optimized SQL queries for Sentinel database analytics and executes them.

4. **SQLDataSummarizationAgent** - Summarizes and presents SQL query results in user-friendly formats with insights and visual recommendations.

5. **AxwayCommunityAgent** - Provides troubleshooting solutions and resolutions for file transfer issues using Axway Community knowledge base.

6. **NotificationConfigurationGenerator** - Creates and configures alert notifications, scheduled reports, and monitoring rules for file transfer events.

7. **OutlierHandlerAgent** - Handles out-of-scope queries, future date requests, and unrelated questions by politely redirecting users.

8. **SemanticAgent** - Analyzes user queries to extract semantic components and business context for improved query processing.

9. **ChartRecommender** - Suggests appropriate chart types and visualizations for presenting file transfer analytics data.

10. **SentinelAnalytics** - Provides general file transfer analytics, reporting, and data insights from the Sentinel database.

11. **SentinelFailedTransferResolution** - Analyzes specific transfer failures and provides detailed failure reasons and resolution steps.


Input → Process → Output

#### Input
- Prompt
- Few Shots
- Question
- Semantic Analysis


#### Output 
- Response


Claude Sonnet 4.0 - us.anthropic.claude-sonnet-4-20250514-v1:0