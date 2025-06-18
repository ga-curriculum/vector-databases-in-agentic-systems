<h1>
  <span class="headline">Vector Databases in Agentic Systems</span>
  <span class="subhead">RAG in Practice</span>
</h1>

## Learning Objective
By the end of this lesson, learners will be able to:
- Implement the three core steps of RAG systems: Embed, Index, and Search
- Integrate vector retrieval into an agent's ReAct framework for grounded reasoning
- Evaluate retrieval quality and identify optimization strategies

## From Theory to Implementation: Building RAG

Let's see how this works with a few lines of code. We'll use **Weaviate**, an open-source vector database, to store a few internal memos and then ask a question that a base LLM couldn't possibly answer.

### Step 1: Setting Up the Environment and Ingesting Documents

First, we install and set up our environment:

```bash
# Install the required dependencies
pip install weaviate-client
pip install openai
```

Then we ingest our documents:

```python
# Ingesting internal documents into a Weaviate vector database.
import weaviate
import os

# Connect to Weaviate and configure it to use OpenAI for embeddings
client = weaviate.Client(
    url="http://localhost:8080",
    additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

# Define a simple "class" or schema for our documents
class_obj = {
    "class": "InternalMemo",
    "vectorizer": "text2vec-openai"
}
# client.schema.create_class(class_obj) # Run once to create the class

# Ingest our internal documents
memos = [
    {
        "title": "Project Phoenix Update",
        "content": "The Project Phoenix timeline has been updated. The new target launch is Q3 2025, pushed back from the original Q1 target due to technical challenges in the authentication module."
    },
    {
        "title": "Remote Work Policy", 
        "content": "Effective immediately, all employees are required to be in office at least 3 days per week. Flexible arrangements can be discussed with direct managers."
    },
    {
        "title": "Q4 Budget Allocation",
        "content": "Engineering department has been allocated an additional $2M for cloud infrastructure costs. This budget must be utilized by December 31st."
    }
]

# Note: The "Project Phoenix Update" memo contains the answer to our test query

client.batch.configure(batch_size=10)
with client.batch as batch:
    for memo in memos:
        batch.add_data_object(data_object=memo, class_name="InternalMemo")
```

### Step 2: Querying the Vector Database

Now that our "librarian" has read the memos, our agent can ask it questions:

```python
# Querying the vector database to retrieve context for the agent.
user_query = "What is the new deadline for Project Phoenix?"

# Perform a vector search to find relevant context
response = (
    client.query
    .get("InternalMemo", ["title", "content"])
    .with_near_text({"concepts": [user_query]})
    .with_limit(1)
    .do()
)

# Extract the retrieved context
retrieved_context = response["data"]["Get"]["InternalMemo"][0]["content"]

# The vector search should return the "Project Phoenix Update" memo since it's 
# semantically closest to our query about "Project Phoenix deadline"
print(f"Retrieved context: {retrieved_context}")
# Expected output: "The Project Phoenix timeline has been updated. The new target launch is Q3 2025..."

# This context is then passed to the LLM to generate the final answer.
final_prompt = f"""
Use the following context to answer the user's question.
Context: {retrieved_context}
Question: {user_query}
Answer:
"""
# The agent's LLM brain would then process this prompt.
```

## RAG in the ReAct Loop: Grounding Agent Reasoning

This retrieval step slots directly into the ReAct loop we learned about in Module 1:

<div class="mermaid">
graph TD
    A[User Query: "What's the Project Phoenix deadline?"] --> B[Thought: "This requires internal knowledge"]
    B --> C[Action: vector_search("Project Phoenix deadline")]
    C --> D[Observation: "Q3 2025 target, delayed from Q1"]
    D --> E[Thought: "I have factual context"]
    E --> F[Final Answer: "Project Phoenix deadline moved to Q3 2025"]
    
    style B fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style C fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style D fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
</div>

### The Enhanced ReAct Process with RAG

- **Thought:** "The user is asking about Project Phoenix's deadline. This is internal information. I must search my knowledge base."

- **Action:** `vector_search("Project Phoenix deadline")`

- **Observation:** The system retrieves the memo: "The Project Phoenix timeline has been updated. The new target launch is Q3 2025..."

- **Thought:** "I have found the answer. I can now respond to the user with factual information."

- **Final Answer:** "Based on the latest internal update, Project Phoenix has been moved to Q3 2025, delayed from the original Q1 target due to technical challenges in the authentication module."

## Measuring Retrieval Quality: Precision@K

Not all retrieval is created equal. We need metrics to evaluate how well our vector database finds relevant documents:

```python
def evaluate_retrieval_quality(query, expected_docs, k=3):
    """
    Evaluate retrieval quality using Precision@K metric
    """
    # Perform vector search
    response = (
        client.query
        .get("InternalMemo", ["title", "content"])
        .with_near_text({"concepts": [query]})
        .with_limit(k)
        .do()
    )
    
    retrieved_docs = response["data"]["Get"]["InternalMemo"]
    relevant_retrieved = 0
    
    for doc in retrieved_docs:
        if doc["title"] in expected_docs:
            relevant_retrieved += 1
    
    precision_at_k = relevant_retrieved / k
    return precision_at_k

# Test with a tricky query
test_query = "What are the new office requirements?"
expected = ["Remote Work Policy"]
precision = evaluate_retrieval_quality(test_query, expected, k=3)
print(f"Precision@3: {precision:.2f}")
```

**Precision@K** measures what percentage of the top K retrieved documents are actually relevant. This helps you tune your chunking strategy and embedding models.

## Common RAG Optimization Strategies

### 1. Chunk Size Optimization

```python
# Experiment with different chunk sizes
chunk_strategies = {
    "small": {"max_tokens": 256, "overlap": 50},
    "medium": {"max_tokens": 512, "overlap": 100}, 
    "large": {"max_tokens": 1024, "overlap": 200}
}

# Test each strategy and measure retrieval quality
for strategy, params in chunk_strategies.items():
    # Re-chunk documents with new parameters
    # Measure precision@3 for standard test queries
    # Compare results
    pass
```

### 2. Hybrid Search: Vector + Keyword

Many production systems combine vector search with traditional keyword search:

```python
# Hybrid approach: combine semantic and keyword search
def hybrid_search(query, alpha=0.7):
    """
    alpha: weight for vector search (1-alpha for keyword search)
    """
    vector_results = client.query.get("InternalMemo", ["title", "content"]) \
        .with_near_text({"concepts": [query]}) \
        .do()
    
    keyword_results = client.query.get("InternalMemo", ["title", "content"]) \
        .with_bm25(query=query) \
        .do()
    
    # Combine and rerank results based on alpha weighting
    return combine_results(vector_results, keyword_results, alpha)
```

### 3. Re-ranking with Cross-Encoders

After initial retrieval, re-rank results for better precision:

```python
from sentence_transformers import CrossEncoder

# Initialize a cross-encoder for re-ranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def rerank_results(query, initial_results, top_k=3):
    """
    Re-rank initial retrieval results using cross-encoder
    """
    query_doc_pairs = [(query, doc["content"]) for doc in initial_results]
    scores = reranker.predict(query_doc_pairs)
    
    # Sort by cross-encoder scores
    ranked_results = sorted(zip(initial_results, scores), 
                          key=lambda x: x[1], reverse=True)
    
    return [result[0] for result in ranked_results[:top_k]]
```

## Exercise: Document Chunking Strategy Design

**Instructions**: Work in triads for 10 minutes. Each group gets a different document type:

### Group A: Technical Specification (50-page PDF)
Design a chunking strategy for a dense technical document with code examples, diagrams, and cross-references.

### Group B: Slack Thread Conversation
Design a chunking strategy for a long Slack thread with multiple participants and various topics.

### Group C: Database Schema Documentation  
Design a chunking strategy for SQL table definitions, relationships, and business rules.

**For each scenario, justify:**
1. Optimal chunk size (tokens)
2. Overlap strategy
3. Metadata to preserve
4. Expected retrieval challenges

## Production Considerations

### Keeping Memory Fresh

```python
# Automated re-embedding pipeline
import schedule
import time

def update_vector_database():
    """
    Nightly job to re-embed updated documents
    """
    # 1. Check for modified documents since last update
    # 2. Delete old embeddings for changed docs
    # 3. Generate new embeddings
    # 4. Update vector database
    # 5. Log metrics and performance
    pass

# Schedule nightly updates
schedule.every().day.at("02:00").do(update_vector_database)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

### Cost Management Strategies

- **Incremental Updates**: Only re-embed changed documents
- **Document Versioning**: Track changes to avoid unnecessary re-processing
- **Batch Processing**: Group updates to minimize API calls
- **Caching**: Store frequently accessed embeddings locally

## Module Checklist

Can you...

✅ **Explain the "amnesia problem" of base LLMs using an analogy?**
- Base LLMs are like consultants with outdated knowledge—they can't access your current company information

✅ **Describe the difference between a traditional database and a vector database?**
- Traditional = Phone book (exact matches), Vector = Librarian (semantic understanding)

✅ **Name the three core steps of a RAG system?**
- 1. Embed (convert text to vectors), 2. Index (store in vector DB), 3. Search (find similar vectors)

✅ **Identify where the retrieval step fits into an agent's ReAct loop?**
- Action phase: agent calls vector_search() tool to retrieve relevant context before reasoning

## Key Takeaways

- RAG transforms agents from hallucination-prone to fact-grounded systems
- Vector databases enable semantic search beyond keyword matching
- Retrieval quality depends on chunking strategy, embedding models, and re-ranking
- Production RAG requires balancing freshness, accuracy, and operational costs
- Integration with ReAct loops makes agent reasoning transparent and auditable

You now have the foundation to build memory-enabled agents that can access dynamic, private information. In the upcoming lab, you'll experiment with a dataset of 1,000 support tickets and optimize retrieval performance for real-world scenarios.
