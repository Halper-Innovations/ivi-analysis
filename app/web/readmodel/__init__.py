"""Read models for the IVI web UI.

Everything in this package reads the books of record through a read-only
SQLite connection (URI ``mode=ro``) or indexes run artifacts from disk into
the UI's own store under ``data/outputs/webui/``. Nothing here may write
``engine.db``.
"""
