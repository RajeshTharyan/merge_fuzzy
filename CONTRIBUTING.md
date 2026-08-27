# Contributing

This is a small personal project. Useful changes are welcome, but there is no
SLA and no promise that every idea will be merged.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Checks to run locally

```bash
pytest
streamlit run fuzzy_streamlit.py
```

Keep matching and file I/O in `fuzzy_core/`. `fuzzy_streamlit.py` and
`fuzzy.py` should stay thin UI layers.

## Pull requests

- Prefer a short description of the behavior change, not only the code change.
- Add or update pytest coverage for parsers, key building, or scoring when
  the logic changes.
- Do not add live-network tests or GUI click-tests unless they are opt-in.
