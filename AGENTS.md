# AGENTS.md

Compact guidance for OpenCode sessions. Full detail lives in `CLAUDE.md` — read it for architecture; this file is only the "don't get burned" checklist.

## Commands (Windows PowerShell, NOT Git Bash for .ps1)

- Use the venv interpreter **always**: `./.venv/Scripts/python.exe`. The system `python` is a Microsoft Store stub that silently exits. It does not accept `/c/...` paths and prints GBK — wrap stdout in UTF-8 if printing Chinese.
- Install deps in order: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` (or `cpu`) **first**, then `pip install -r requirements.txt`. `torch`/`torchvision` are intentionally absent from `requirements.txt`.
- Run server: `./start.ps1` (PowerShell only) / stop: `./stop.ps1`. Foreground: `./.venv/Scripts/python.exe main.py`.
- Tests: stdlib `unittest` only (no pytest). `./.venv/Scripts/python.exe -m unittest discover -s tests -t .`. Single case: `./.venv/Scripts/python.exe -m unittest tests.test_core_logic` (dot-path, not file path). Frontend: `node tests/test_stream_manager_frontend.mjs`.

## API / write requests

- **Every write endpoint (POST/PUT/PATCH/DELETE) requires header `X-Lab-Monitor-Request: 1`** or a guard middleware returns 403. When `Origin` is present it must be same-origin and `Host` whitelisted (else 421). Frontend injects this in `static/js/utils/api.js`. `curl` writes must also send Chinese as UTF-8 (`--data-binary @file`), not `-d` (Git Bash encodes GBK → 400).
- `POST /api/admin/shutdown` only accepts loopback — don't trigger during debugging.
- When `LAB_MONITOR_USERNAME`/`PASSWORD` are both empty, all endpoints are unauthenticated (and non-loopback bind is refused).

## Code landmines (verified in CLAUDE.md)

- **ReID match threshold `0.75` is duplicated in 5 places** — `src/reid.py:171`, `src/reid.py:208`, `src/reid_validator.py:31`, `src/pipeline.py:110`, `src/pipeline.py:210`, plus `src/identity_store.py:371`. Changing one is a no-op.
- **Never rebuild `PersonTracker` inside `_reset_stream_state()`** (`src/pipeline.py:222-229`): `BaseTrack._count` is class-level, so rebuilding zeroes every camera's track ids → identity collisions. Reset tracker state on stream reopen/file-loop instead by clearing absent streaks, not by reconstructing.
- **`import src/db.py` runs `Database()` at module load and connects to production `outputs/lab_monitor.db`.** Tests touch production data — back it up first.
- `LAB_MONITOR_MODEL_POOL=1` reverts only pool size, **not** the `cv2.setNumThreads(1)` / `torch.set_num_threads(2)` clamping.
- ReID features are throttled to disk (50 updates / 30s). New shutdown paths must call `flush_identity_features()` or lose recent averages.

## Startup degradations (no crash, just silent feature loss)

- `topology.json` references a camera not in `sources.json` → `TopologyValidationError` caught in `main.py` → empty topology → **MISSING_PERSON silently dead**. Only visible in startup logs.
- A source file with `isOpened()==True` but failing first `read()` is dropped with a warning; that camera gets no card.
- Edit `sources.json` ⇄ `topology.json` together (topology validates against configured camera ids).

## Frontend

- Native ES modules, no build step. After editing any `static/css/*.css` or `static/js/**`, **bump `?v=` in `static/index.html`** (currently CSS `?v=11.5`, `app.js?v=11.6`; `main.css` 的 `@import` 子路径也带 `?v=11.5`，bump 时要一起改). `app.js` 里的 `import './modules/*.js'` 不带版本号 — hard refresh (`Ctrl+F5`) to see changes.
- `static/js/modules/stream_manager.js` caps concurrent MJPEG to ~4 and uses IntersectionObserver; respect `resetStreamRegistry()`/`registerStreamImage()` timing or streams leak / never open.

## Other

- `INTRUSION` fences come only from `config/roi.json` (normalized `[0,1]` coords); only `rnd_16` has one today. `POST /api/roi` hot-reloads; hand-edits need a restart.
- Git: branch `main`, semantic prefixes (`feat:`/`fix:`/`docs:`/`style:`/`refactor:`). Never commit `*.pt`, `videos/`, `*.7z`, `docs/*.xlsx` (gitignored).

See `CLAUDE.md` for full architecture, perf tiers, and the `P0-1`/`F5` audit-ticket dictionary in `docs/CODE_AUDIT_2026-07-28.md`.
