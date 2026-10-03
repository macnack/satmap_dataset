# Security

## Reporting

If you find a vulnerability in this repository (credential handling, unsafe
path writes, etc.), open a private security advisory on GitHub or contact the
maintainer via the repository’s contact channels. Please do not file a public
issue for credential leaks.

## Secrets

- Copy `.secret.template` → `.secret` (gitignored). Never commit `.secret` or `.env`.
- Prefer environment variables (`SATMAP_NLS_API_KEY`, `SATMAP_LANTMATERIET_*`).
- NLS may append `api-key` as a query parameter; avoid logging full request URLs.
- satmap-studio can optionally write an NLS key to `.secret` — use only on
  trusted local machines.

## Downloads

Assets should be written via temporary `.part` files and renamed on success.
Any non-empty file may be treated as complete by reuse checks — delete truncated
files before re-running.
