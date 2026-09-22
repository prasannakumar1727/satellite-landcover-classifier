# Website UI (Flask)

Serves the glass-style dashboard UI and calls the existing `src/` pipeline
(no changes made to any DIP/classification logic).

## Run
```bash
pip install -r ../requirements.txt
python -m webapp.server
```
Then open **http://localhost:5000** in your browser.

Files:
- `server.py` — Flask API (`/api/config`, `/api/analyze`) wrapping `src/pipeline.run_pipeline`
- `static/index.html`, `static/style.css`, `static/script.js` — the UI
