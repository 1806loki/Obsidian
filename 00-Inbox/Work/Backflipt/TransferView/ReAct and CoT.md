Explore CoT and []()ReAct implemenatation
See whether the LLM can check the query it generated and create another query during a single LLM call

	Sources 

https://react-lm.github.io/
https://www.promptingguide.ai/


How many transfers failed yesterday from Partner Account ProdAccount001 to Adobe

```  
"semantic_analysis": {
            "question_type": "count",
            "grouping_type": [
                {
                    "value": "time",
                    "queryColumn": "EVENTTIMESTAMP"
                }
            ],
            "grouping_interval": {
                "value": "hour",
                "queryColumn": "EVENTTIMESTAMP",
                "expression": "TO_CHAR(DATE '1970-01-01' + (TO_NUMBER(EVENTTIMESTAMP) / 1000 / 86400), 'HH24') "
            },
            "transfer_type": {
                "value": "failed-transfer",
                "queryColumn": "STATE"
            },
            "filter": {
                "state": {
                    "value": [
                        "FAILED"
                    ],
                    "queryColumn": "STATE"
                },
                "partner_account": {
                    "value": "ProdAccount001",
                    "queryColumn": "SOURCEACCOUNT"
                },
                "receiver": {
                    "value": "Adobe",
                    "queryColumn": "RECEIVERID"
                },
                "LOCATION": {
                    "value": "axway-st-1",
                    "queryColumn": "LOCATION"
                }
            },
            "order_by": "time-ascending",
            "time": {
                "value": "yesterday",
                "queryColumn": "EVENTTIMESTAMP",
                "expression": "TO_NUMBER(EVENTTIMESTAMP) >= ((TRUNC(SYSDATE - 1) - DATE '1970-01-01') * 86400 * 1000) AND TO_NUMBER(EVENTTIMESTAMP) < ((TRUNC(SYSDATE) - DATE '1970-01-01') * 86400 * 1000)"
            },
            "time_interval": "1",
            "businessType": "NYL",
            "tableType": "ORACLE"
        },
```


|                       | Latency   | Accuracy  |
| --------------------- | --------- | --------- |
| Rule-Guided Reasoning | Increases | Increases |
| Prompts Decomposition | Decreases | Increases |


#### Rule-Guided Reasoning

**Pros :**
1. Structured decision-making with confidence scoring
2. Consistency in predictable behavior
3. Easier debugging and traceability

**Cons :**
1. Latency overhead from extra processing (10 - 15 seconds)



#### Prompt Deconstruction

**Pros :**
1. Performance gains from smaller, focused prompts
2. Maintainability through modular components

**Cons :**
1. Risk of context loss when breaking apart prompts

