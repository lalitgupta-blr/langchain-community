"""
LangChain Retriever for VRX Studio Versioned Relational eXchange.
Provides deterministic, version-grounded Document retrieval for RAG pipelines.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from vrx_integrations.client import VRXClient
except ImportError:
    try:
        from vrx_studio.client import VRXClient
    except ImportError:
        VRXClient = None  # type: ignore

try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
except ImportError:
    # Graceful fallback Document structure if langchain_core is not installed
    class Document(BaseModel):  # type: ignore
        page_content: str
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class BaseRetriever(BaseModel):  # type: ignore
        def get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
            return self._get_relevant_documents(query, **kwargs)

    CallbackManagerForRetrieverRun = Any  # type: ignore


class VRXPolicyRetriever(BaseRetriever):
    """
    Retriever that fetches deterministic, audit-provenanced policy records
    from VRX Studio rather than approximate vector embeddings.
    """

    api_token: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    domain: Optional[str] = Field(default=None)
    version: str = Field(default="G1", description="Graph revision state ('G0' or 'G1').")
    include_provenance: bool = Field(default=True)
    _client: Optional[Any] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        client = data.get("_client")
        if client:
            object.__setattr__(self, "_client", client)
        else:
            if VRXClient is None:
                raise ImportError(
                    "Could not import vrx_integrations python package. "
                    "Please install it with `pip install vrx-integrations`."
                )
            token = self.api_token or os.environ.get("VRX_API_TOKEN") or os.environ.get("VRX_JWT_TOKEN")
            url = self.base_url or os.environ.get("VRX_BASE_URL") or "https://api.vrxstudio.ai"
            object.__setattr__(self, "_client", VRXClient(api_token=token, base_url=url, default_domain=self.domain))

    def _get_relevant_documents(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Synchronously query VRX Studio and wrap the result in a LangChain Document."""
        client = getattr(self, "_client", None)
        if not client:
            if VRXClient is None:
                raise ImportError(
                    "Could not import vrx_integrations python package. "
                    "Please install it with `pip install vrx-integrations`."
                )
            token = self.api_token or os.environ.get("VRX_API_TOKEN") or os.environ.get("VRX_JWT_TOKEN")
            url = self.base_url or os.environ.get("VRX_BASE_URL") or "https://api.vrxstudio.ai"
            client = VRXClient(api_token=token, base_url=url, default_domain=self.domain)
            object.__setattr__(self, "_client", client)

        resp = client.query(
            query=query,
            version=self.version,
            domain=self.domain,
            include_provenance=self.include_provenance
        )

        metadata: Dict[str, Any] = {
            "source": "vrx_studio",
            "domain": resp.domain,
            "model_id": resp.model_id,
            "version": self.version,
            "prediction": resp.prediction,
            "unit": resp.unit,
            "latency_ms": resp.latency_ms,
            "serving_engine": resp.serving_engine,
            "retraining_updates": resp.retraining_updates,
            "acid_adaptation_passed": resp.acid_adaptation_passed
        }

        if resp.provenance:
            metadata["node_id"] = resp.provenance.node_id
            metadata["graph_snapshot"] = resp.provenance.graph_snapshot
            metadata["active_revision"] = resp.provenance.active_revision
            metadata["deterministic_rule"] = resp.provenance.deterministic_rule

        if resp.merkle_audit_hash:
            metadata["merkle_audit_hash"] = resp.merkle_audit_hash

        doc = Document(
            page_content=resp.format_text(),
            metadata=metadata
        )
        return [doc]
