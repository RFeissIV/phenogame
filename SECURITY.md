# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | ✓         |
| < 0.3   | ✗         |

## Reporting a Vulnerability

This is a scientific research package. It does not handle authentication,
network traffic, or sensitive personal data.

If you discover a security issue (e.g., code injection via crafted CSV
input), please email the maintainer directly rather than opening a
public issue.

## Scope

phenogame processes CSV data files and performs numerical computations.
It does not:
- Access the network
- Execute arbitrary user code
- Store credentials
- Handle PII
