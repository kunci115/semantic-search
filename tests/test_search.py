"""
Unit tests for SemanticSearch using a mock or simplified in-memory database.
"""

import pytest
from semantic_search.search import SemanticSearch


class MockDatabase:
    """
    A simple in-memory database replacement
    that replicates get_all_documents() and add_document().
    """
    def __init__(self):
        self.docs = []

    def add_document(self, text, embedding):
        self.docs.append((text, embedding))

    def get_all_documents(self):
        return self.docs

@pytest.fixture
def mock_search_engine():
    """
    Provides a SemanticSearch instance with a mock DB.
    """
    db = MockDatabase()
    engine = SemanticSearch(database=db)
    return engine

def test_add_document(mock_search_engine):
    """
    Ensure a document is added properly to the mock DB.
    """
    mock_search_engine.add_document("Test Document")
    all_docs = mock_search_engine.db.get_all_documents()
    assert len(all_docs) == 1
    assert all_docs[0][0] == "Test Document"
    assert isinstance(all_docs[0][1], list)  # embedding as list

def test_build_faiss_index(mock_search_engine):
    """
    Test building a FAISS index with mock DB docs.
    """
    mock_search_engine.add_document("Doc 1")
    mock_search_engine.add_document("Doc 2")
    mock_search_engine.build_faiss_index()
    assert mock_search_engine.faiss_index is not None

def test_retrieve(mock_search_engine):
    """
    Test retrieving documents from the mock DB after adding them.
    """
    mock_search_engine.add_document("NLP is great")
    mock_search_engine.add_document("Deep Learning transforms NLP")
    mock_search_engine.build_faiss_index()

    results = mock_search_engine.retrieve("NLP", top_k=2)
    assert len(results) <= 2
    # We expect at least 1 match
    assert any("NLP" in doc for doc in results)
