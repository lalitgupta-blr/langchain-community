"""
LangChain Retriever for VRX Studio Versioned Relational eXchange.
Provides deterministic, version-grounded Document retrieval for RAG pipelines.
"""

from langchain_community.retrievers.vrx import VRXPolicyRetriever

__all__ = ["VRXPolicyRetriever"]
