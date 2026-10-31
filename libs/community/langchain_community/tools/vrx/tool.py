"""
LangChain Tool for VRX Studio Neuro-Symbolic Governance Engine.
Enables LangChain ReAct agents and toolkits to query deterministic domain rules.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field

try:
    from vrx_integrations.client import VRXClient
except ImportError:
    try:
        from vrx_studio.client import VRXClient
    except ImportError:
        VRXClient = None  # type: ignore

try:
    from langchain_core.tools import BaseTool
    from langchain_core.callbacks import CallbackManagerForToolRun
except ImportError:
    # Graceful fallback if langchain_core is not installed in the current environment
    class BaseTool(BaseModel):  # type: ignore
        name: str
        description: str
        return_direct: bool = False

        def run(self, tool_input: Any, **kwargs: Any) -> Any:
            if isinstance(tool_input, dict):
                return self._run(**tool_input)
            return self._run(str(tool_input))

    CallbackManagerForToolRun = Any  # type: ignore


class VRXToolInput(BaseModel):
    """Input schema for VRX Studio Governance Tool."""
    query: str = Field(
        ...,
        description="The domain question, policy condition, or compliance check to evaluate (e.g. 'What is the prior-auth window for CPT 70553 under UHC?')"
    )
    version: str = Field(
        default="G1",
        description="The knowledge graph revision state. 'G1' queries the mutated live policy; 'G0' queries the baseline policy."
    )


class VRXGovernanceTool(BaseTool):
    """
    Tool that queries the VRX Studio Versioned Relational eXchange engine.
    
    Use this tool whenever you need deterministic, audit-provenanced answers
    over volatile business policies, tariffs, healthcare prior-authorization rules,
    or compliance constraints where traditional Vector RAG might hallucinate.
    """

    name: str = "vrx_policy_governance"
    description: str = (
        "Authoritative neuro-symbolic engine for versioned enterprise business rules. "
        "Input should be a specific policy question (e.g., return window, SLA, prior-auth code, tariff duty). "
        "Guarantees 100% discrete mathematical recall and audit provenance."
    )
    args_schema: Type[BaseModel] = VRXToolInput
    return_direct: bool = False

    api_token: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    domain: Optional[str] = Field(default=None)
    default_version: str = Field(default="G1")
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

    def _run(
        self,
        query: str,
        version: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool synchronously."""
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

        target_ver = version or self.default_version
        response = client.query(
            query=query,
            version=target_ver,
            domain=self.domain,
            include_provenance=self.include_provenance
        )
        return response.format_text()
