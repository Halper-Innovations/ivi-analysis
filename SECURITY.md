# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/Halper-Innovations/ivi-analysis/security/advisories/new)
rather than a public issue. Include steps to reproduce and the version or
commit you tested. You can expect an acknowledgement within a few days.

## Scope notes

IVI Analysis runs locally and stores API keys only in environment variables or
a local `.env` file that is git-ignored. The application:

- refuses to load a `.env` file that is readable or writable by other users;
- redacts credentials from URLs, logs and stored provenance before writing them;
- contacts only the SEC, the configured price provider and, when enabled, the
  configured AI provider and research sources on an allowlist.

Reports about any path that leaks a credential, writes outside the configured
data directory, or reaches a host outside those lists are especially welcome.
