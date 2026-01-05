#### Optimizations
- Optimize the Fileview to run sub-graphs for testing 


- Update all the error response formats to below
```
{
"isSuccess": Boolean,
"message": String
}
  ```

- Need to update the observation to Plan and the content from "The query retrieves" to "Retrieve"
- Format the router prompt
- Update the `questions_to_cache.json`
- Complete the TODOs
- Cleanup the env files
- Update the few shots retrieval methods
- Automate the data population script
- Update env variables

#### Priority Changes :
-  Make the env variables dynamic in UI, so that they can be changed while deploying a docker image.
-  Add KMS_VPC_ENDPOINT_DNS and use it to connect to the KMS key (ID will also be provided via a variable: AWS_KEY_ID).


Alerts
- Handle current user email and email mentioned in the query


Questions 
- How are deciding if a question should be fileName or Count
- Why 