#Tech/AI/PromptEngineering 
#### Definition
 Divide the prompt into sections and build a dynamic prompt using the required sections (we get this from Semantic Layer).

##### Implementation

- [ ] Normalization of the prompt
- [x] Divide it into sections 
- [x] Dynamic selection of Question type in Semantic Layer

#### 1. Core Framework Sections

These establish the foundation and can operate independently.

- Role & Objective – Defines AI’s fundamental purpose (SQL generator for file transfer analytics).
- RuleGuidedReasoningProcess – Provides structured reasoning for query building.
- Input – Specifies inputs (examples, schema, date, semantic mappings).
- StructuredOutputRequirements – Ensures all responses follow a consistent format.

#### 2. Input Processing

Guides how incoming questions are interpreted.

- SemanticAnalysisUsage – Maps semantic meaning to columns/filters, overrides general rules when analysis data exists.

#### 3. Question Type Classification

Handles specific query patterns.

- CountQuestions – Counting & aggregation (`how many`, transfer counts, success/failure).
- SuperlativeQuestions – Ranking & comparison (`top`, `most`, `least`).
- TrafficPeriodQuestions – Time-based traffic analysis (peak, busiest periods, usage trends).
- FileNameQuestions – File-specific queries (names, extensions, verification).

#### 4. Specialized Query Handling

Focused on transfer outcomes and date/time rules.

- FailedTransfers – Error-focused queries (failure reasons, error analysis).
- SuccessfulTransfers – Queries specific to completed transfers.
- DateTimeQuestions – Full date/time filtering and calculations.


#### 5. Technical Implementation Rules

Control how queries are actually built.

- TimePeriodFormatting – Database-specific formatting for time-based queries.
- AggregationAndDetailLogic – Decides between summary vs file-level detail queries.
- General – Universal rules and constraints applied across all queries.


#### 6. Coupling & Dependencies

##### A. Independent Sections

- `<Role>`, `<Objective>`, `<RuleGuidedReasoningProcess>`, `<Input>`, `<StructuredOutputRequirements>`  → Standalone, no dependencies.


##### B. Loosely Coupled

- General ↔ SemanticAnalysisUsage, TimePeriodFormatting
- SemanticAnalysisUsage ↔ General, TimePeriodFormatting
- FailedTransfers → General
- SuccessfulTransfers → General
- SuperlativeQuestions → CountQuestions

→ Mostly reference-based, easy to separate.

##### C. Tightly Coupled (High Dependency)

1. Core Query Logic Cluster

- CountQuestions ↔ TimePeriodFormatting ↔ TrafficPeriodQuestions

2. Time Processing Cluster

- DateTimeQuestions ↔ TimePeriodFormatting

3. Question Type Cluster

- FileNameQuestions ↔ CountQuestions ↔ TimePeriodFormatting

4. AggregationAndDetailLogic

- Strongly dependent on CountQuestions


#### 7. Coupling Summary

- High Coupling Clusters:

- Core Query Logic, Time Processing, Question Type dependencies.

- Medium Coupling:

- SuperlativeQuestions → CountQuestions

- AggregationAndDetailLogic → CountQuestions

- Low Coupling:

- General ↔ SemanticAnalysisUsage

- FailedTransfers → General

- SuccessfulTransfers → General

|                 | Time Taken (secs) | Prompt Size |
| --------------- | ----------------- | ----------- |
| Full Prompt     | 25.67             | 55470       |
| Composed Prompt | 25.72             | 14625       |

#### Dynamic Assembly Process
1. **Semantic Analysis**: Analyze user query to determine question types
2. **Section Selection**: Select core sections + question-type specific sections
3. **Prompt Assembly**: Combine selected sections in correct order
4. **Execution**: Process the assembled prompt through Lumen

#### 4. Benefits of This Architecture

- **Efficiency**: Only relevant prompt content is processed
- **Flexibility**: Easy to add new question types and sections
- **Performance**: Reduced token usage and processing time
- **Maintainability**: Clear separation of concerns
- **Scalability**: Can handle complex query patterns without overwhelming the system

This decomposition strategy ensures optimal performance within Lumen's constraints while maintaining the full functionality required for complex SQL query generation tasks.

#### Upgrade Prompts

- [ ] Make Semantic section handle the semantic analysis output
- [ ] Reduce the cross referencing of sections as much as possible 
- [ ] Remove redundant lines
- [ ] Remove Critical/Important/Mandatory keywords
- [ ] Merge and Split sections as necessary 



Todo List : 

- [x] State Ordering Issues
- [x] Column Selection and Aggregation Logic
- [x] Column Alias Inconsistencies
- [x] Time Grouping Granularity Issues
- [x] Case-Insensitive Matching Issues
- [ ] Question Classification Ambiguity

Question Type Token Counts:

|Question Type|Sections|Tokens|
|---|---|---|
|count|COUNT_LOGIC_SECTION + QUERY_FORMAT_DECISION_SECTION|1,662|
|filename|FILE_NAME_QUESTIONS_SECTION|703|
|traffic|TRAFFIC_PERIOD_QUESTIONS_SECTION|595|
|superlative|SUPERLATIVE_QUESTIONS_SECTION + COUNT_LOGIC_SECTION|1,288|

Core Sections Analysis:

| Section                              | Tokens |
| ------------------------------------ | ------ |
| CONTEXT_OPENING_SECTION              | 116    |
| SEMANTIC_ANALYSIS_SECTION            | 593    |
| GENERAL_GUIDELINES_SECTION           | 1,099  |
| AGGREGATION_AND_DETAIL_LOGIC_SECTION | 430    |
| TIME_HANDLING_SECTION                | 1,213  |
| CONTEXT_CLOSING_SECTION              | 0      |
| Core sections total                  | 3,451  |

### Question Type Token Counts (With Core Sections):

| Question Type | Core Sections | Question-Specific | Total Tokens | Reduction | Reduction % |
| ------------- | ------------- | ----------------- | ------------ | --------- | ----------- |
| traffic       | 3,451         | 595               | 4,046        | 2,829     | 41.1%       |
| filename      | 3,451         | 703               | 4,154        | 2,721     | 39.6%       |
| superlative   | 3,451         | 1,288             | 4,739        | 2,136     | 31.1%       |
| count         | 3,451         | 1,662             | 5,113        | 1,762     | 25.6%       |
Total Tokens for Complete Prompt: 6,875
Average Prompt Size after Prompt Decomposition : 4513
Average Reduction Percentage: 34.4%

![[Claude Monthly Cost.png| 300]]