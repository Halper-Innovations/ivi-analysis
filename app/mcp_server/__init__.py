"""MCP server exposing read-only SEC EDGAR data (filings, XBRL financials) to AI assistants.

The tool logic lives in plain modules (``companies``, ``financials``,
``filing_text``) that return JSON-serializable dicts and never import the MCP
SDK, so they are usable and testable without the optional ``mcp`` extra.
``server`` wires them into an MCP server; ``ivi-mcp`` runs it over stdio.
"""
