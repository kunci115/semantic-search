"""
Semantic Search implementation using FAISS and SentenceTransformer.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    """
    Handles document storage, indexing, and retrieval using semantic embeddings.
    """

    def __init__(self, database, model_name='all-MiniLM-L6-v2'):
        """
        Initialize a SemanticSearch instance.

        Args:
            database: A database instance created by DatabaseFactory.
            model_name (str): Name of the SentenceTransformer model to load.
        """
        self.model = SentenceTransformer(model_name)
        self.db = database
        self.faiss_index = None

    def build_faiss_index(self):
        """
        Builds a FAISS index from stored embeddings.
        Raises:
            ValueError: If no embeddings exist in the database.
        """
        docs = self.db.get_all_documents()
        if not docs:
            raise ValueError("No embeddings found in the database.")

        # Gather embeddings from the DB and convert to float32
        embeddings = np.array([doc[1] for doc in docs], dtype=np.float32)

        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        # type checker complains because SWIG sees add(n, x). We ignore this mismatch:
        self.faiss_index.add(embeddings)  # type: ignore

    def add_document(self, text):
        """
        Encodes and adds a document embedding to the database, then updates the FAISS index if built.

        Args:
            text (str): The text of the document to add.
        """
        # Ensure that SentenceTransformers returns a NumPy array (convert_to_tensor=False)
        embedding = self.model.encode([text], convert_to_tensor=False)[0]
        embedding = embedding.astype(np.float32)

        # Store in DB as a list (JSON-serializable if needed)
        self.db.add_document(text, embedding.tolist())

        # Update FAISS index if it exists
        if self.faiss_index is not None:
            self.faiss_index.add(embedding[np.newaxis, :])  # type: ignore

    def retrieve(self, query, top_k=5):
        """
        Retrieves the most similar documents for the given query.

        Args:
            query (str): The search query.
            top_k (int): Number of documents to retrieve.

        Returns:
            list of str: Document texts most similar to the query.
        """
        # Encode query
        query_embedding = self.model.encode(query)
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.cpu().numpy()
        query_embedding = query_embedding.astype(np.float32)

        # Reshape for FAISS: (1, embedding_dim)
        query_embedding = np.expand_dims(query_embedding, axis=0)

        docs = self.db.get_all_documents()

        # If FAISS index built, use the index
        if self.faiss_index is not None:
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            results = [
                docs[idx][0] for idx in indices[0] if idx < len(docs)
            ]
        else:
            # Fallback: do a manual cosine similarity
            doc_embeddings = np.array(
                [np.array(doc[1]) for doc in docs],
                dtype=np.float32
            )
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            sorted_indices = np.argsort(similarities)[::-1][:top_k]
            results = [
                docs[idx][0]
                for idx in sorted_indices
                if similarities[idx] > 0.3
            ]
        return results
