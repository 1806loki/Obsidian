#Tech/AI/LLM/RAG


![[Advanced_RAG.png]]


RAG is a [[NLP (Natural Language Processing)]] technique  that enhances text generation by retrieving relevant external data and passing it to the [[Vector Embeddings]] before generating the response

![[Basic_RAG.png]]

RAG consists of three Basic Steps
1. Indexing
2. Retrieval 
3. Generation

**Indexing** - This is the process of storing and organizing external data in a structured way i.e. using vector embeddings in a database.

**Retrieval** - This step involves searching the indexed data to find the most relevant information based on a given query. Typically, this is done using similarity search techniques like vector search.

**Generation** - The retrieved information is then combined with the user's query and passed to a [[Vector Embeddings]] , which generates a response based on the retrieved data and it's internal knowledge.

The Advance Steps are :
1. Query Translation
   1. Multi-Query
   2. RAG Fusion
   3. Decomposition
   4. Step Back
   5. HyDE (Hypothetical Documentation Embeddings)
2. Routing
3. Indexing
   1. Multi Representation
   2. RAPTOR (Retrieval-Augmented Pretrained Transformer with Optimal Recall)
   3. colBERT (Contextualized Late Interaction over BERT)
4. CRAG (Context-Rich Augmented Generation)
5. Adaptive RAG

#### 1. Query Translation
   Query translation enhances how a query is processed to improve retrieval effectiveness, it includes :
   - **Multi-Query** - Generates multiple variations of a query to improve the chances of retrieving relevant information.
   - **RAG Fusion** - Merges results from multiple queries to improve response accuracy.
   - **Decomposition** - Breaks a complex query into smaller, simpler sub-queries to retrieve more precise information.
   - **Step Back** - Reformulates a query to retrieve more general or contextual information before narrowing down.
   - **HyDE (Hypothetical Documentation Embeddings)** - Generates a hypothetical answer using an LLM and retrieves a document similar to this generated answer
#### 2.Routing
   Dynamically decides whether to retrieve external knowledge, use the LLM's internal knowledge or apply specialized retrieval techniques based on the query type.
   
#### 3. Indexing
   Advanced methods for improving how data is stored and retrieved efficiently, it includes :
   - **Multi Representation** - Stores multiple representations(e.g., different embeddings or formats) of the same data for better retrieval accuracy.
   - **RAPTOR (Retrieval-Augmented Pretrained Transformer with Optimal Recall)** - A retrieval technique that combines different search methods(e.g., keyword-based and vector search) to find the most relevant information more accurately and efficiently
   - **colBERT (Contextualized Late Interaction over BERT)** - Uses contextualized embeddings at a token level (rather than whole-document embeddings) for more precise retrieval. 
#### 4. CRAG
  Enhances retrieval by incorporating additional contextual layers, such as structured metadata or semantic relationships, to improve relevance.

#### 5. Adaptive RAG
  Dynamically adjusts retrieval and generation strategies based on the query type, ensuring optimal performance for different kinds of questions.



