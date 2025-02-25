# Semantic Search

An **open-source** Python library for semantic search, featuring:

- **FAISS** for rapid vector similarity.
- **SentenceTransformers** for high-quality embeddings.
- **Pluggable database backends** (MongoDB, SQLite, Redis, PostgreSQL, MySQL).

With **this library**, you can efficiently store, index, and retrieve documents based on **semantic similarity**.

---

## Table of Contents

1. [Features](#features)  
2. [Databases & FAISS Table](#databases--faiss-table)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Testing](#testing)  
6. [FAQ](#faq)

---

## Features

- **Multi-DB Support** – Choose MongoDB, SQLite, Redis, PostgreSQL, or MySQL.  
- **FAISS Index** – Build a fast, in-memory index for vector searches.  
- **Easy API** – `add_document()`, `build_faiss_index()`, `retrieve()`.  
- **Scalable** – Handle millions of embeddings with FAISS.  
- **Open Source** – Contributions are welcome!

---

## Databases & FAISS Table

| **Database**   | **Best For**                       | **Advantages**                                     | **Considerations**                                 |
|---------------|------------------------------------|----------------------------------------------------|----------------------------------------------------|
| **MongoDB**    | JSON-like docs, horizontal scaling | - Great for big data<br>- Flexible schema          | - No built-in vector search<br>- Use external index|
| **SQLite**     | Lightweight local storage          | - Easy setup<br>- Single file DB                   | - Not ideal for concurrent writes                 |
| **Redis**      | Fast in-memory caching             | - Extremely fast read<br>- Good for ephemeral data | - Data stored in RAM<br>- Must handle persistence |
| **PostgreSQL** | Traditional SQL, robust & reliable | - Potential pgvector extension<br>- ACID-compliant | - Requires indexing optimization for large data    |
| **MySQL**      | Widely used SQL store              | - Scalable<br>- Familiar to many devs              | - No native vector index; manual approach needed   |

**FAISS**:
- **IndexFlatL2** used for demonstration (simple & effective for smaller data).
- For **large data**: consider IVF, HNSW, or GPU-based indexing.

---

## Installation

1. **pypi install** this repo:
   ```bash
    pip install pysemantic-search
   ```

2. **Install Dependencies for Development**:
```bash
    git clone https://github.com/username/semantic-search.git
    cd semantic-search
    pip install -r requirements.txt
    
    Make sure to adjust dependencies (faiss-cpu vs. faiss-gpu) depending on your environment.
    If you plan to use PyTorch on GPU, install torch with CUDA support.
```

## Usage
Below is a basic example using MongoDB as the database backend. You can switch to other backends by changing db_type.
```python
from semantic_search import SemanticSearch, DatabaseFactory

# 1. Create a database connection (MongoDB example)
db = DatabaseFactory.create_database(
    db_type="mongodb",
    mongo_uri="mongodb://localhost:27017/",
    db_name="semantic_db",
    collection_name="documents"
)

# 2. Initialize SemanticSearch with the DB
search_engine = SemanticSearch(database=db)

# 3. If documents already exist, build FAISS index
try:
    search_engine.build_faiss_index()
except ValueError:
    print("No existing documents found. The FAISS index was not built.")

# 4. Add a new document
search_engine.add_document("Deep learning for NLP is a powerful tool.")

# 5. Retrieve similar documents
query = "Best techniques for NLP deep learning?"
results = search_engine.retrieve(query, top_k=3)
print("Results:", results)

```


## Testing
This library includes pytest tests in the tests/ directory. To run them locally: <br>
1. Install dev dependencies (pytest, pylint, etc.) from requirements.txt. <br>
2. Run 
```bash
python -m pytest tests/
```

## FAQ

### 1. **Why do I see a type error about `faiss_index.add(x)`?**
FAISS’s Python bindings are generated via SWIG, so static type checkers (like Pyright) think the signature is `add(n, x)`. We add `# type: ignore` to bypass this false positive. **Runtime** usage works fine with `add(x)`.

### 2. **Is `.cpu()` needed for embeddings?**
- By default, SentenceTransformers returns **NumPy arrays** if you pass `convert_to_tensor=False`, so `.cpu()` is **not** needed.  
- If you use `convert_to_tensor=True`, you get a **PyTorch tensor**. Convert it with:
  ```python
  embedding = embedding.cpu().numpy().astype(np.float32)
   ```
### 3. **Which database is best?**
- MongoDB or PostgreSQL for large data.
- Redis for fast in-memory lookups.
- SQLite for small local apps.

### 4. **Can I store embeddings in the DB & FAISS on disk?**
Currently, IndexFlatL2 is in-memory only. To store FAISS on disk, use other FAISS indexes with I/O support or HDF5-based approach.</br>

### 5. **How do I contribute?**
Fork this repo, create a new branch, and submit a PR with changes.</br>
Add tests in tests/.</br>

## Reference
https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/README.md</br>
