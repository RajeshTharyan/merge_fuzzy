# Security policy

This tool reads uploaded (Streamlit) or locally chosen (Tkinter) CSV, Excel,
and Stata files and keeps them in memory for the session. It does not add
authentication, sandboxing, or a size cap.

## Supported versions

Only the latest commit on `main` is considered current. Older snapshots are
not patched.

## Reporting a vulnerability

Please use [GitHub Security Advisories](https://github.com/RajeshTharyan/merge_fuzzy/security/advisories/new)
rather than a public issue. Include the affected file path and a way to
reproduce the problem.

## What to assume as a user

- Treat any **hosted** copy (Streamlit Community Cloud, a shared Codespace,
  a public URL) as untrusted for confidential data. This repository does not
  currently publish a hosted demo.
- A huge or malformed spreadsheet can stall or crash the process (denial of
  service against whoever is running the app).
- Excel/CSV parsers have a history of security bugs; do not open files from
  people you do not trust.
- There are no application secrets in this repo. If you deploy your own copy
  and later add Streamlit secrets, keep them in `.streamlit/secrets.toml`
  (gitignored) and never commit them.
