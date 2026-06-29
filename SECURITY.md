# Security Policy

## Supported versions

The latest released minor version receives security fixes. As a pre-1.0 project,
older versions are not separately maintained — upgrade to the latest release.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately to the maintainers (see `CONTRIBUTING.md` for contact). Include:

- A description of the vulnerability and its impact.
- Steps to reproduce (a minimal proof of concept).
- Affected version(s) and environment.

We aim to acknowledge reports within **72 hours** and to provide a remediation
timeline after triage. Please allow us a reasonable period to release a fix
before any public disclosure.

## Scope & hardening

The threat model, controls, OWASP API Top-10 mapping, and a deployment hardening
checklist are documented in **[docs/security.md](docs/security.md)** and
**[docs/deployment.md](docs/deployment.md#hardening-checklist)**.
