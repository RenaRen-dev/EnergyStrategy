"""
Snowflake key-pair authenticated client.

Loads an unencrypted (or passphrase-protected) RSA private key from disk,
authenticates to Snowflake, and exposes simple helpers for the common
data-loading workflow:

    from utility.snowflake_client import SnowflakeClient

    sf = SnowflakeClient()
    df = sf.read_sql("SELECT TOP 10 * FROM MY_DB.MY_SCHEMA.MY_TABLE")

    # context-managed for short-lived scripts
    with SnowflakeClient() as sf:
        df = sf.read_table("MY_DB.MY_SCHEMA.MY_TABLE")

All connection settings come from environment variables (see .env.example),
so no secrets are committed.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv

import snowflake.connector
from snowflake.connector import SnowflakeConnection

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def _required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing required env var {name}. Copy .env.example to .env and "
            f"fill it in."
        )
    return val


def _load_private_key_der(key_path: Path, passphrase: str | None) -> bytes:
    """Read a PKCS8 .p8 private key from disk and return DER bytes that
    snowflake-connector-python expects in its `private_key` kwarg."""
    if not key_path.exists():
        raise FileNotFoundError(f"Private key not found at: {key_path}")

    pwd: bytes | None = passphrase.encode() if passphrase else None
    with open(key_path, "rb") as f:
        pem = f.read()

    pkey = serialization.load_pem_private_key(
        pem, password=pwd, backend=default_backend()
    )
    return pkey.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


class SnowflakeClient:
    """Thin wrapper around `snowflake.connector` with key-pair auth and
    pandas helpers."""

    def __init__(
        self,
        *,
        account: str | None = None,
        user: str | None = None,
        private_key_path: str | Path | None = None,
        private_key_passphrase: str | None = None,
        role: str | None = None,
        warehouse: str | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        self.account = account or _required("SNOWFLAKE_ACCOUNT")
        self.user = user or _required("SNOWFLAKE_USER")

        key_path = private_key_path or os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
        if not key_path:
            raise RuntimeError(
                "SNOWFLAKE_PRIVATE_KEY_PATH not set in env and no private_key_path arg given."
            )
        key_path = Path(key_path)
        if not key_path.is_absolute():
            key_path = ROOT / key_path

        passphrase = (
            private_key_passphrase
            if private_key_passphrase is not None
            else os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE") or None
        )
        self._private_key_der = _load_private_key_der(key_path, passphrase)

        self.role = role or os.getenv("SNOWFLAKE_ROLE") or None
        self.warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE") or None
        self.database = database or os.getenv("SNOWFLAKE_DATABASE") or None
        self.schema = schema or os.getenv("SNOWFLAKE_SCHEMA") or None

        self._conn: SnowflakeConnection | None = None

    def connect(self) -> SnowflakeConnection:
        if self._conn is None or self._conn.is_closed():
            kwargs = {
                "account": self.account,
                "user": self.user,
                "private_key": self._private_key_der,
                "client_session_keep_alive": True,
            }
            for k, v in [
                ("role", self.role),
                ("warehouse", self.warehouse),
                ("database", self.database),
                ("schema", self.schema),
            ]:
                if v:
                    kwargs[k] = v
            self._conn = snowflake.connector.connect(**kwargs)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.is_closed():
            self._conn.close()
        self._conn = None

    def __enter__(self) -> "SnowflakeClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def cursor(self) -> Iterator:
        cur = self.connect().cursor()
        try:
            yield cur
        finally:
            cur.close()

    def execute(self, sql: str, params: Iterable | None = None) -> None:
        with self.cursor() as cur:
            cur.execute(sql, params or ())

    def read_sql(self, sql: str, params: Iterable | None = None) -> pd.DataFrame:
        """Execute a query and return a pandas DataFrame.

        Uses fetch_pandas_all() (Arrow-based) when possible -- significantly
        faster than tuple iteration for large pulls.
        """
        with self.cursor() as cur:
            cur.execute(sql, params or ())
            try:
                return cur.fetch_pandas_all()
            except snowflake.connector.errors.NotSupportedError:
                rows = cur.fetchall()
                cols = [c[0] for c in cur.description]
                return pd.DataFrame(rows, columns=cols)

    def read_table(self, fully_qualified_name: str, *, limit: int | None = None) -> pd.DataFrame:
        sql = f"SELECT * FROM {fully_qualified_name}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.read_sql(sql)

    def get_connection_info(self) -> dict:
        """Return current session metadata (account, region, user, role,
        warehouse, database, schema, version). Useful for verifying the setup."""
        df = self.read_sql(
            "SELECT "
            "  CURRENT_ACCOUNT()    AS account, "
            "  CURRENT_REGION()     AS region, "
            "  CURRENT_USER()       AS user, "
            "  CURRENT_ROLE()       AS role, "
            "  CURRENT_WAREHOUSE()  AS warehouse, "
            "  CURRENT_DATABASE()   AS database, "
            "  CURRENT_SCHEMA()     AS schema, "
            "  CURRENT_VERSION()    AS version"
        )
        return df.iloc[0].to_dict()


if __name__ == "__main__":
    with SnowflakeClient() as sf:
        info = sf.get_connection_info()
        print("Connected to Snowflake:")
        for k, v in info.items():
            print(f"  {k:>10}: {v}")