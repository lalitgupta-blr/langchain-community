import unittest
from unittest.mock import patch
from langchain_community.retrievers.vrx import VRXPolicyRetriever
from vrx_integrations.models import VRXInferenceResponse, VRXProvenance


class TestVRXRetriever(unittest.TestCase):
    @patch("vrx_integrations.client.VRXClient.query")
    def test_vrx_retriever_get_documents(self, mock_query):
        mock_query.return_value = VRXInferenceResponse(
            status="success",
            domain="Healthcare Prior-Authorization",
            query="What is the window for CPT 70553?",
            prediction="3",
            unit="business days",
            latency_ms=85.0,
            serving_engine="vLLM Multi-Task Dynamic Multiplexer",
            retraining_updates=0,
            provenance=VRXProvenance(
                node_id="Rule_UHC_70553_v2",
                graph_snapshot="G1 (Mutated Live State)",
                active_revision=2,
                deterministic_rule="Decision window within 3 business days for UHC 70553"
            ),
            merkle_audit_hash="test_merkle_hash_123"
        )

        retriever = VRXPolicyRetriever(api_token="test_token", version="G1")
        docs = retriever.get_relevant_documents("What is the window for CPT 70553?")

        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertIn("Result: 3 business days", doc.page_content)
        self.assertEqual(doc.metadata["source"], "vrx_studio")
        self.assertEqual(doc.metadata["node_id"], "Rule_UHC_70553_v2")
        self.assertEqual(doc.metadata["merkle_audit_hash"], "test_merkle_hash_123")
        self.assertEqual(doc.metadata["retraining_updates"], 0)


if __name__ == "__main__":
    unittest.main()
