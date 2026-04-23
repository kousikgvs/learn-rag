# FAISS RAG Pipeline Report

## Overview

This notebook builds a compact end-to-end retrieval-augmented generation workflow on top of a local FAISS vector store. The flow starts with raw text, converts it into searchable embeddings, stores those embeddings in FAISS, retrieves relevant chunks for a user query, reranks them with a cross-encoder, and finally sends the best context to an LLM with a strict instruction to answer only from the retrieved scope.

The source document used here is a long biographical text about A. P. J. Abdul Kalam. That makes this notebook a good practical example of how to go from messy raw text to grounded question answering.

## What This Notebook Uses

This pipeline is built from a mix of lightweight retrieval tooling and LLM orchestration utilities already present in the notebook:

- `faiss` for fast vector similarity search
- `langchain_community.vectorstores.FAISS` as the vector store wrapper
- `langchain_community.docstore.in_memory.InMemoryDocstore` for document mapping
- `sentence_transformers.SentenceTransformer` for dense embeddings
- `sentence_transformers.CrossEncoder` for reranking
- `transformers.AutoTokenizer` and `transformers.AutoModelForSequenceClassification` for the reranking model path
- `langchain_text_splitters.RecursiveCharacterTextSplitter` for baseline chunking
- `langchain_text_splitters.SpacyTextSplitter` for sentence-based chunking
- `langchain_groq.ChatGroq` for answer generation
- `langchain_core.messages.SystemMessage` and `HumanMessage` for structured LLM prompting
- `dotenv` for loading environment variables
- `uuid_utils.uuid4` for document IDs

The notebook also includes a few environment safeguards before model loading:

- disabling Hugging Face Xet storage with `HF_HUB_DISABLE_XET`
- disabling urllib warnings
- a small `httpx` verification patch for model download issues

## Pipeline Walkthrough

### 1. Source Loading

The notebook first searches for `data.txt` in the current folder and then in the parent folder. Once found, the file is read as UTF-8 text.

This is a simple but useful touch because it keeps the notebook flexible even if the working directory changes.

### 2. Text Cleaning

Before chunking, the text is normalized to remove obvious noise from the raw source. The cleaning logic does the following:

- removes citation markers like `[2]` or `[a]`
- normalizes Windows line endings to standard newlines
- replaces tab characters
- compresses repeated spaces
- compresses excessive blank lines
- strips line-by-line whitespace
- removes empty lines

This step matters because retrieval quality often drops when the source contains reference markers, irregular spacing, or formatting debris.

### 3. Chunking Strategy

The notebook experiments with two chunking approaches:

#### Recursive character chunking

A `RecursiveCharacterTextSplitter` is created with:

- `chunk_size = 500`
- `chunk_overlap = 100`
- separators ordered as paragraph, newline, sentence, space, then fallback

This is the simple baseline.

#### Sentence-based chunking

A `SpacyTextSplitter` is also used with:

- `pipeline = "sentencizer"`
- `chunk_size = 500`
- `chunk_overlap = 80`

The notebook ultimately embeds the `spacy_chunks`, so the effective retrieval path is sentence-aware rather than purely character-based.

### 4. Embedding Model

The active embedding model in the notebook is:

- `sentence-transformers/all-MiniLM-L6-v2`

There is also a commented alternative:

- `BAAI/bge-base-en`

The notebook wraps the embedder in a small custom adapter class, `SentenceTransformerEmbeddings`, so it can be used cleanly with the LangChain FAISS store. The adapter supports both:

- `embed_documents()` for chunk embeddings
- `embed_query()` for query embeddings

The query path also prefixes the input with:

`Represent this sentence for searching relevant passages:`

That prompt-style prefix helps retrieval by aligning query embedding behavior with search intent.

### 5. Vector Store Construction

Once chunk embeddings are created, the notebook builds a FAISS index using:

- `faiss.IndexFlatL2` for similarity search
- `InMemoryDocstore` for document storage
- UUID-based IDs for each chunk

The notebook then adds precomputed embeddings and text into the FAISS wrapper and persists the store to disk inside:

- `vectorDB/faiss_store/`

That makes the pipeline reusable without recomputing everything on every run.

### 6. Retrieval

For inference, the notebook reloads the saved FAISS store using `FAISS.load_local()` and performs similarity search with:

- `k = 20`

This returns a pool of candidate text chunks for the question.

### 7. Reranking

The retrieval output is then refined using a cross-encoder reranker based on:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`

The reranking function tokenizes question-answer pairs, runs the sequence classification model, applies a sigmoid to the logits, and sorts the candidates by relevance score. The top reranked chunks are then passed forward as the grounding context.

This is a strong design choice because the first-stage FAISS retrieval is fast, while the second-stage reranker improves precision.

### 8. LLM Answer Generation

The answering stage uses:

- `qwen/qwen3-32b` through `ChatGroq`

The prompt is intentionally strict. It tells the model:

- use only the provided context
- do not use prior knowledge
- do not make assumptions
- if the answer is not explicit in context, say so
- return exactly three lines in the required format

The final response format is:

- `answer: ...`
- `did you answer from the scope : yes/no`
- `did it make any assumptions : yes/no`

That is a good grounding pattern because it forces the answering step to behave more like a verifier than a free-form generator.

### 9. Streaming Inference

The notebook’s end-to-end inference cell streams the LLM response token by token using `llm.stream(messages)` instead of waiting for the entire response at once.

That makes the notebook feel more interactive and closer to how a production assistant would behave in a real app.

## Timed End-to-End Inference

The notebook includes a timed inference pipeline covering the full path:

1. query input
2. vector DB load
3. similarity search
4. reranking
5. LLM call
6. final streamed output

### Current timing snapshot

Using the query:

`where did APJ abdul kalam die ?`

The latest measured timings were:

- Vector DB load time: `0.0020 seconds`
- Retrieval time: `0.0258 seconds`
- Reranking time: `0.8691 seconds`
- LLM call time: `0.8355 seconds`
- Total pipeline time: `1.7333 seconds`

### Final grounded output

The streamed answer produced by the notebook was:

```text
answer: A.P.J. Abdul Kalam died in Shillong while delivering a lecture at IIM Shillong.
did you answer from the scope : yes
did it make any assumptions : no
```

## Diagnostic Thinking Included in the Notebook

One of the strongest parts of this notebook is that it does not stop at implementation. It also includes a diagnostic section that asks whether poor retrieval quality is caused by:

- chunk size being too small
- chunk boundaries breaking semantic meaning
- structured text being split awkwardly
- embedding choice not matching the data well
- lack of context awareness in the splitter

The notebook then proposes an improved chunking path with:

- `chunk_size = 1200`
- `chunk_overlap = 300`

This is a practical improvement path because larger chunks often preserve meaning better when the source contains semi-structured or densely factual text.

## What Works Well Here

This notebook already demonstrates several strong RAG design decisions:

- raw text is cleaned before indexing
- chunking is compared rather than assumed
- embeddings are normalized
- FAISS storage is persisted locally
- reranking is added instead of relying only on nearest-neighbor retrieval
- the LLM is prompt-constrained to reduce hallucination
- output schema is enforced
- streaming and timing are both included
- diagnostics are part of the workflow, not an afterthought

## Practical Takeaway

In plain terms, this notebook is not just a FAISS demo. It is a compact retrieval pipeline that shows the full lifecycle of a grounded QA system:

- prepare noisy text
- split it intelligently
- embed it
- store it in FAISS
- retrieve relevant context
- rerank for precision
- answer with a constrained LLM
- measure latency end to end

That makes it a solid learning artifact for understanding how local vector search, reranking, and strict answer generation fit together in a realistic RAG workflow.

## Suggested Next Step

If this notebook is extended further, the most valuable next improvement would be to make the reportable metrics repeatable across multiple queries so you can compare:

- chunking strategies
- embedding models
- reranker depth
- average latency
- answer accuracy

That would turn this from a good prototype into a stronger evaluation notebook.
