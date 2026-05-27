import sys
import unittest
from pathlib import Path

# Add the project root to sys.path so we can import from utility
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utility.snowflake_client import SnowflakeClient


class TestSnowflakeConnection(unittest.TestCase):
    """Integration tests for SnowflakeClient connectivity."""

    def setUp(self) -> None:
        self.client = SnowflakeClient()
        self.client.connect()

    def tearDown(self) -> None:
        self.client.close()

    def test_get_connection_info_returns_expected_keys(self) -> None:
        """Verify that get_connection_info() returns all expected session metadata keys."""
        info = self.client.get_connection_info()

        expected_keys = {"ACCOUNT", "REGION", "USER", "ROLE", "WAREHOUSE", "DATABASE", "SCHEMA", "VERSION"}
        self.assertEqual(set(info.keys()), expected_keys)

    def test_get_connection_info_has_required_values(self) -> None:
        """Verify that critical session fields are populated (not None)."""
        info = self.client.get_connection_info()

        self.assertIsNotNone(info["ACCOUNT"], "ACCOUNT must not be None")
        self.assertIsNotNone(info["USER"], "USER must not be None")
        self.assertIsNotNone(info["VERSION"], "VERSION must not be None")

    def test_read_sql_returns_dataframe(self) -> None:
        """Verify that read_sql() returns a non-empty DataFrame for a trivial query."""
        import pandas as pd

        df = self.client.read_sql("SELECT 1 AS test_col")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIn("TEST_COL", df.columns)


if __name__ == "__main__":
    unittest.main()
