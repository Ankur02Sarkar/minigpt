# Security Policy

## Supported Versions
| Version | Supported | Security Policy |
|---------|-----------|-----------------|
| 0.1.0   | :white_check_mark: | See below |

## Reporting a Vulnerability
Please report security issues via email at [security@minigpt.io](mailto:security@minigpt.io).

We will:
1. Acknowledge receipt within 48 hours
2. Triage and assess severity within 5 business days
3. Provide a patch or mitigation within 30 days (budget-dependent)
4. Credit reporters in the `CHANGELOG.md` (unless requested otherwise)

## What to Report
- Authentication bypasses or token leaks
- Prompt injection vectors that surface PII
- Denial-of-service via token generation
- Any discovery of training data memorization

## Security Policy
- We follow [CVE](https://cve.mitre.org/) naming for tracked issues
- No pay-for-play; all reports treated equally
- Credits: reporters may request to be named in `CHANGELOG.md` or stay anonymous
- If the issue involves Azure resource exposure, we will deallocate affected VMs immediately

## Preferred Languages
- English (for issue description)
- YAML (for config-related issues)
- Cypher (for graph/query issues)
