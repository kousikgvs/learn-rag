# FAISS Notebook Report

## Overview

This notebook shows a simple FAISS-based retrieval pipeline over multiple company text files.

It loads text files from the `Datasets` folder, splits them into chunks, turns the chunks into vectors, stores them in FAISS, searches with metadata filters, and reranks the retrieved chunks with a cross-encoder model.

## Files Used In This Notebook

The notebook reads `.txt` files from the `Datasets` folder.

At the time of the latest run, it found these four files:

- `amazon_2023.txt`
- `amazon_2024.txt`
- `microsoft_2023.txt`
- `microsoft_2024.txt`

The notebook also saves the vector store inside `vectorDB/faiss_store/`.

## Main Tools Used

This notebook uses these main libraries and models:

- `faiss` for vector similarity search
- `langchain_community.vectorstores.FAISS` for the vector store wrapper
- `langchain_community.docstore.in_memory.InMemoryDocstore` for storing chunk documents
- `langchain_text_splitters.RecursiveCharacterTextSplitter` for chunking text
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking
- `transformers` for loading the reranker model and tokenizer
- `uuid_utils.uuid4` for unique chunk IDs
- `langchain_groq.ChatGroq` with `qwen/qwen3-32b` as the loaded LLM

## What You Did In This Notebook

### 1. Loaded the libraries and environment values

The notebook imports the libraries needed for FAISS, embeddings, reranking, and the LLM.

It also loads values from the root `.env` file.

### 2. Loaded the LLM setup

The notebook creates a `ChatGroq` model with:

- model: `qwen/qwen3-32b`
- temperature: `0`

In the current notebook, the LLM is initialized, but the main completed workflow in this file is retrieval and reranking.

### 3. Collected dataset files

The notebook points to the `Datasets` folder and lists all `.txt` files.

It prints the file names so you can confirm what data is available before building the vector store.

### 4. Previewed one file

The notebook loads the first text file and shows:

- the selected file name
- total character count
- a short preview of the text

This helps confirm that the input data is being read correctly.

### 5. Checked all files

The notebook loops through all dataset files and prints the size of each file.

This gives a quick view of the corpus before chunking.

### 6. Split the files into chunks

The notebook uses `RecursiveCharacterTextSplitter.from_tiktoken_encoder()` with:

- `encoding_name = "cl100k_base"`
- `chunk_size = 250`
- `chunk_overlap = 50`

It first shows a single-file example, then rebuilds the final `docs` list from all dataset files.

Each chunk keeps metadata such as:

- source path
- document name
- year
- company

This metadata is important because you later use it to filter search results.

### 7. Loaded the embedding model

The notebook loads the embedding model:

- `sentence-transformers/all-MiniLM-L6-v2`

It also reads the optional Hugging Face token from the environment.

### 8. Converted chunks into vectors

The notebook defines a custom adapter class so the sentence-transformer model can work with the LangChain FAISS wrapper.

It creates vectors for all chunk texts and prints the output shape.

At the latest successful run, the notebook showed:

- `20` chunk vectors
- embedding size `384`

So the vector shape was `20 x 384`.

### 9. Built the FAISS store

The notebook creates a FAISS index with `faiss.IndexFlatL2`.

Then it creates a LangChain `FAISS` vector store using:

- the embedding adapter
- the FAISS index
- an in-memory docstore
- an empty index-to-docstore map

### 10. Added embeddings to FAISS

The notebook creates a UUID for each chunk and adds the text, vectors, and metadata to the FAISS store.

At the latest run, it printed:

- `Added 20 embeddings to FAISS`

### 11. Saved and reloaded the vector store

The notebook saves the local FAISS store to the `faiss_store` folder.

Then it reloads the same store with `FAISS.load_local()`.

This makes the workflow reusable without rebuilding the store every time.

### 12. Compared filtered and unfiltered search

The notebook runs a similarity search for `"microsoft 2023"` in two ways:

- without metadata filtering
- with metadata filtering for company and year

At the latest run, the output was:

- docs before metadata filter: `20`
- docs after metadata filter: `10`

This shows that the metadata filter is working correctly.

### 13. Retrieved chunks for a business question

The notebook searches the vector store with this query:

`what is revenue of microsoft in 2023 ?`

It applies this metadata filter:

- company: `microsoft`
- year: `2023`

Then it prints the top retrieved chunks and stores their text in the `answers` list.

### 14. Loaded the reranker model

The notebook loads:

- `AutoModelForSequenceClassification`
- `AutoTokenizer`

for the model:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`

It also creates a `CrossEncoder` instance for the same model.

### 15. Reranked the retrieved chunks

The notebook defines a function called `top_k_rerank()`.

This function:

- pairs the question with each retrieved chunk
- tokenizes the pairs
- runs the reranker model
- converts logits to scores with `sigmoid`
- sorts the chunks by score
- returns the top results

The notebook tests reranking with the same Microsoft 2023 revenue question.

## Current Notebook Output Summary

Based on the latest executed cells, this notebook currently shows these confirmed results:

- `4` dataset text files found
- `20` chunk vectors created
- vector size `384`
- `20` embeddings added to FAISS
- `20` documents before metadata filtering
- `10` documents after metadata filtering

## Important Notes

This report matches the current notebook content.

The older report was describing a different workflow with `data.txt`, APJ Abdul Kalam content, text cleaning, sentence chunking, streaming answers, and timing sections. Those steps are not the main flow in this notebook now, so they were removed from this report.

The notebook currently goes up to retrieval and reranking. The LLM is loaded, but there is no completed final answer-generation section in the current visible workflow.

## Simple Takeaway

In simple terms, this notebook teaches how to:

- read multiple text files
- split them into chunks
- convert chunks into embeddings
- store them in FAISS
- search with metadata filters
- rerank the results with a cross-encoder

It is a clean learning notebook for understanding the retrieval side of a RAG pipeline.
