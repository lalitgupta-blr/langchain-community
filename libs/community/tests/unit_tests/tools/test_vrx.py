import unittest
from unittest.mock import patch
from langchain_community.tools.vrx import VRXGovernanceTool
from vrx_integrations.models import VRXInferenceResponse, VRXProvenance


class TestVRXTool(unittest.TestCase):
    @patch("vrx_integrations.client.VRXClient.query")
    def test_vrx_tool_run(self, mock_query):
        mock_query.return_value = VRXInferenceResponse(
            status="success",
            domain="Healthcare Prior-Authorization",
            query="What is the window for CPT 70553?",
            prediction="3",
            unit="business days",
            provenance=VRXProvenance(
                node_id="Rule_UHC_70553_v2",
                graph_snapshot="G1 (Mutated Live State)",
                active_revision=2,
                deterministic_rule="Decision window within 3 business days for UHC 70553"
            )
        )

        tool = VRXGovernanceTool(api_token="test_token", domain="healthcare_prior_auth")
        res = tool.run("What is the window for CPT 70553?")

        self.assertIn("Result: 3 business days", res)
        self.assertIn("Rule_UHC_70553_v2", res)
        mock_query.assert_called_once()


if __name__ == "__main__":
    unittest.main()
