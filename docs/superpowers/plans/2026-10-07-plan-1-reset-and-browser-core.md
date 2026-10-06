# Plan 1 — Repo Reset and Browser Inference Core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reset the repo to the v2 layout and build a tested in-browser detection core (worker + ONNX Runtime Web + tracker + threat engine) proven end-to-end against the stock YOLO26n COCO model.

**Architecture:** A Web Worker owns the ONNX session (WebGPU, falling back to WASM) and turns `ImageBitmap` frames into `Detection[]`. The main thread runs a framework-free pipeline (`web/src/core`) — ByteTrack-lite tracker, then the threat engine — and paints overlay primitives onto a canvas. A temporary Lab page exercises image and webcam input and is covered by Playwright.

**Tech Stack:** Vite 8, React 19, TypeScript 6.0, Tailwind CSS 4, onnxruntime-web 1.30.0, onnxruntime-node 1.30.0 (tests only), Vitest 5, Playwright 1.63; Python 3.12 + Ultralytics 8.4.174 for model export and golden fixtures.

**Spec:** `docs/superpowers/specs/2026-10-07-weaponshield-v2-design.md`

## Global Constraints

- All work on branch `v2`. Never commit to or push `main` in this plan.
- Commits are authored as `Priyanshu Kumar Tiwari <143282195+Halok600@users.noreply.github.com>` (already the repo's git config). **No `Co-Authored-By` trailers, no "Generated with" lines, in any commit or PR.** Before every push run `git log origin/v2..HEAD --format=%B | grep -i -E "co-authored|claude|generated with"` and confirm it prints nothing.
- TypeScript pinned to `~6.0.3` (typescript-eslint requires `<6.1.0`). `onnxruntime-web` and `onnxruntime-node` pinned to exactly `1.30.0`.
- Node 24, npm 11, Python 3.12.
- Model output contract: ONNX exported with `nms=False` gives one output shaped `[1, 300, 6]` = `[x1, y1, x2, y2, conf, class_id]` in letterboxed input-pixel coordinates. Letterbox pad colour is `rgb(114,114,114)`, input is RGB float32 in `[0,1]`, NCHW.
- App class keys are exactly `'person' | 'pistol' | 'rifle' | 'knife'`. Overlay colours: person `#22c55e`, weapon `#ef4444`, armed `#f59e0b`.
- No model binaries in git: `*.onnx` and `*.pt` are git-ignored; models are downloaded from GitHub Releases by `web/scripts/prepare-assets.mjs` and verified by SHA-256.
- No paid services. Repo licence: AGPL-3.0.
- COOP `same-origin` + COEP `require-corp` headers on dev, preview and production (needed for WASM threads).

---

## File Structure

```
.gitignore                               rewritten for v2
LICENSE                                  AGPL-3.0 text
README.md                                short v2 work-in-progress readme
docs/decisions.md                        lessons carried over from v1
fixtures/golden/images/{bus,zidane}.jpg  reference images (from Ultralytics assets)
fixtures/golden/{name}.lb.rgba           640x640 letterboxed RGBA from Ultralytics
fixtures/golden/{name}.out.f32           raw [1,300,6] output from Python onnxruntime
fixtures/golden/{name}.json              source size + Ultralytics predictions
training/requirements-export.txt         pinned Python deps for export
training/scripts/export_onnx.py          .pt -> end-to-end ONNX + layout check + sha256
training/scripts/make_golden.py          writes fixtures/golden/*
web/package.json, tsconfig*.json, vite.config.ts, eslint.config.js, playwright.config.ts, index.html, vercel.json
web/models.json                          model manifest (file, url, sha256)
web/scripts/prepare-assets.mjs           downloads models + copies ORT wasm into public/
web/src/core/types.ts                    ClassKey, Box, Detection, Track, Point
web/src/core/geometry.ts                 letterbox geometry, unletterbox, iou, center
web/src/core/preprocess.ts               RGBA -> NCHW float tensor
web/src/core/postprocess.ts              [1,300,6] -> Detection[]
web/src/core/kalman.ts                   1-D constant-velocity Kalman + box wrapper
web/src/core/tracker.ts                  ByteTrack-lite
web/src/core/threat.ts                   K-of-N confirmation, ARMED, events
web/src/core/presets.ts                  Low / Balanced / High
web/src/core/pipeline.ts                 tracker + threat glued together
web/src/config/models.ts                 ModelConfig + ACTIVE_MODEL
web/src/engine/protocol.ts               worker message types
web/src/engine/detector.worker.ts        ONNX session in a worker
web/src/engine/detector-client.ts        main-thread client with backpressure
web/src/engine/live-loop.ts              video-frame loop feeding the client
web/src/render/overlay.ts                overlay primitives + canvas painter
web/src/lab/LabPage.tsx                  temporary dev harness (image + webcam)
web/e2e/lab.spec.ts                      Playwright tests
.github/workflows/web.yml                CI
```

---

### Task 1: Reset the repository to the v2 layout

**Files:**
- Delete: `backend/`, `frontend/`, `app.py`, `Dockerfile`, `render.yaml`, `requirements.txt`, `model/`, `scripts/`, `PROJECT_ACTIVITY_LOG.md`
- Replace: `LICENSE`, `README.md`, `.gitignore`
- Create: `docs/decisions.md`

**Interfaces:**
- Consumes: nothing
- Produces: a clean tree with `docs/` only (plus the spec and plans); tag `v1-legacy` on GitHub pointing at the old `main`.

- [ ] **Step 1: Tag the legacy version and push the tag**

```bash
cd /d/Projects/WeaponShield-AI
git fetch origin
git tag v1-legacy origin/main
git push origin v1-legacy
```
Expected: `* [new tag] v1-legacy -> v1-legacy`

- [ ] **Step 2: Remove v1 code**

```bash
git rm -r -q backend frontend app.py Dockerfile render.yaml requirements.txt model scripts PROJECT_ACTIVITY_LOG.md
git status --short | head
```
Expected: only `D ` lines for those paths.

- [ ] **Step 3: Replace LICENSE with AGPL-3.0**

```bash
curl -fsSL https://www.gnu.org/licenses/agpl-3.0.txt -o LICENSE
head -3 LICENSE
```
Expected first line contains `GNU AFFERO GENERAL PUBLIC LICENSE`.

- [ ] **Step 4: Write `.gitignore`**

```gitignore
# Node
node_modules/
dist/
web/public/models/
web/public/ort/
playwright-report/
test-results/
*.tsbuildinfo

# Python
.venv/
__pycache__/
*.pyc
.ruff_cache/
.pytest_cache/

# Models and data (published via GitHub Releases / Hugging Face, never committed)
*.onnx
*.pt
training/datasets/
training/runs/

# Local
.env
.env.*
!.env.example
.DS_Store
.playwright-mcp/
```

- [ ] **Step 5: Write `README.md`**

```markdown
# WeaponShield AI

Real-time weapon and person detection that runs entirely in your browser.

> **v2 is under construction on this branch.** The previous version (FastAPI backend + Colab) is preserved at the [`v1-legacy`](https://github.com/Halok600/WeaponShield-AI/tree/v1-legacy) tag.

## What v2 is

- **Detects** person, pistol, rifle and knife with a YOLO26 model fine-tuned on Roboflow Universe + Open Images V7 data, evaluated on a held-out test set.
- **Runs on the visitor's device** with ONNX Runtime Web (WebGPU, falling back to WebAssembly). No server, no uploads, no hosting cost.
- **Tracks** people and weapons across frames and flags an **ARMED** person when a confirmed weapon is held by them.
- **Edge tool** (`edge/`) for real RTSP/CCTV streams with Telegram alerts.

Design: [`docs/superpowers/specs/2026-10-07-weaponshield-v2-design.md`](docs/superpowers/specs/2026-10-07-weaponshield-v2-design.md)

## Licence

AGPL-3.0 (required by Ultralytics YOLO). See [LICENSE](LICENSE).
```

- [ ] **Step 6: Write `docs/decisions.md`**

```markdown
# Decisions and lessons carried over from v1

v1 (tag `v1-legacy`) was a FastAPI + React app with a YOLOv8s model. These are the lessons that shaped v2.

## Model
- **Domain gap is the main accuracy risk.** A public firearm model scoring 89% mAP50 on clean photos dropped to 1–3% confidence on CCTV frames. v2 trains with CCTV-style augmentation (motion blur, JPEG compression, low light, noise) instead of adding noise at inference time.
- **Hard negatives matter.** v1 flagged a tablet on a desk as a weapon. v2 trains on people holding phones, tablets, remotes, drills, hair dryers, umbrellas, cameras and flashlights.
- **Knife was the weakest class** (mAP50 0.636, recall 0.592). v2 adds Open Images V7 knife/dagger boxes.
- **Honest evaluation.** v1 reported held-out test metrics (overall mAP50 0.833). v2 keeps that discipline and adds de-duplication and group-aware splits so frames from one video never straddle train and test.

## Detection logic
- **Single-frame decisions flicker.** v1's K-of-N frame buffer with decay cut false positives. v2 keeps the idea but applies it per tracked object.
- **One threshold does not fit all modes.** v1 needed separate image, video and webcam thresholds. v2 expresses this as Low / Balanced / High presets tuned from validation PR curves.

## Deployment
- **There is no free, always-on GPU server.** Colab needed manual restarts and an ngrok URL that changed every session (the v1 live site died this way); Render's free tier ran out of memory with PyTorch and cold-started slowly with ONNX. v2 runs inference in the browser.
- **Public demos must not send alerts to the owner.** v1 emailed every visitor's detection to one fixed inbox. v2's website alerts stay in the browser; only the self-hosted edge tool sends Telegram messages, using the operator's own bot.
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: reset repo for v2 (remove v1 backend/frontend, AGPL licence, decisions log)"
git log -1 --format='%an <%ae>%n%B'
```
Expected: author line is Priyanshu Kumar Tiwari, body has no trailer.

---

### Task 2: Scaffold the `web/` app

**Files:**
- Create: `web/package.json`, `web/tsconfig.json`, `web/tsconfig.app.json`, `web/tsconfig.worker.json`, `web/tsconfig.node.json`, `web/vite.config.ts`, `web/eslint.config.js`, `web/index.html`, `web/vercel.json`, `web/src/main.tsx`, `web/src/App.tsx`, `web/src/index.css`, `web/src/vite-env.d.ts`

**Interfaces:**
- Consumes: nothing
- Produces: npm scripts `dev`, `build`, `preview`, `lint`, `typecheck`, `test`, `e2e`; Vitest picks up `src/**/*.test.{ts,tsx}` in the `node` environment by default.

- [ ] **Step 1: Create `web/package.json`**

```json
{
  "name": "weaponshield-web",
  "private": true,
  "version": "2.0.0",
  "type": "module",
  "scripts": {
    "predev": "node scripts/prepare-assets.mjs",
    "dev": "vite",
    "prebuild": "node scripts/prepare-assets.mjs",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "lint": "eslint .",
    "typecheck": "tsc -b",
    "pretest": "node scripts/prepare-assets.mjs",
    "test": "vitest run",
    "e2e": "playwright test"
  }
}
```

- [ ] **Step 2: Install dependencies**

```bash
cd /d/Projects/WeaponShield-AI/web
npm install react@^19.3.0 react-dom@^19.3.0 onnxruntime-web@1.30.0
npm install -D vite@^8.3.3 @vitejs/plugin-react@^6.1.2 typescript@~6.0.3 \
  @types/react @types/react-dom @types/node @webgpu/types \
  tailwindcss@^4.3.3 @tailwindcss/vite@^4.3.3 \
  vitest@^5.0.3 jsdom @testing-library/react \
  onnxruntime-node@1.30.0 @playwright/test@^1.63.0 \
  eslint@^10 @eslint/js typescript-eslint@^8.71.1 globals eslint-plugin-react-hooks eslint-plugin-react-refresh
```
Expected: exits 0. If npm reports a peer conflict, read it, pick the newest mutually compatible versions, and record the change in the commit message.

- [ ] **Step 3: Create TypeScript configs**

`web/tsconfig.json`:
```json
{
  "files": [],
  "references": [
    { "path": "./tsconfig.app.json" },
    { "path": "./tsconfig.worker.json" },
    { "path": "./tsconfig.node.json" }
  ]
}
```

`web/tsconfig.app.json`:
```json
{
  "compilerOptions": {
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.app.tsbuildinfo",
    "target": "ES2023",
    "lib": ["ES2023", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "types": ["vite/client", "@webgpu/types", "node"],
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "verbatimModuleSyntax": true,
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "skipLibCheck": true,
    "noEmit": true
  },
  "include": ["src"],
  "exclude": ["src/engine/detector.worker.ts"]
}
```

`web/tsconfig.worker.json`:
```json
{
  "compilerOptions": {
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.worker.tsbuildinfo",
    "target": "ES2023",
    "lib": ["ES2023", "WebWorker"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "types": ["@webgpu/types"],
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "verbatimModuleSyntax": true,
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "skipLibCheck": true,
    "noEmit": true
  },
  "include": ["src/engine/detector.worker.ts", "src/engine/protocol.ts", "src/core/**/*.ts", "src/config/models.ts"],
  "exclude": ["src/**/*.test.ts"]
}
```

`web/tsconfig.node.json`:
```json
{
  "compilerOptions": {
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.node.tsbuildinfo",
    "target": "ES2023",
    "lib": ["ES2023", "DOM"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "types": ["node"],
    "strict": true,
    "verbatimModuleSyntax": true,
    "allowImportingTsExtensions": true,
    "skipLibCheck": true,
    "noEmit": true
  },
  "include": ["vite.config.ts", "playwright.config.ts", "e2e"]
}
```

`web/src/vite-env.d.ts`:
```ts
/// <reference types="vite/client" />
```

- [ ] **Step 4: Create `web/vite.config.ts`**

```ts
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// Cross-origin isolation enables SharedArrayBuffer, which ONNX Runtime Web needs for WASM threads.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { headers: isolation },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
  test: {
    include: ['src/**/*.test.{ts,tsx}'],
    environment: 'node',
    testTimeout: 30_000,
  },
});
```

- [ ] **Step 5: Create `web/eslint.config.js`**

```js
import js from '@eslint/js';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'public', 'playwright-report', 'test-results'] },
  {
    files: ['**/*.{ts,tsx}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: { ecmaVersion: 2023, globals: { ...globals.browser, ...globals.worker } },
    plugins: { 'react-hooks': reactHooks, 'react-refresh': reactRefresh },
    rules: {
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
    },
  },
  { files: ['scripts/**/*.mjs'], languageOptions: { globals: globals.node } },
);
```

- [ ] **Step 6: Create the HTML entry, styles and a placeholder App**

`web/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>WeaponShield AI</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`web/src/index.css`:
```css
@import "tailwindcss";

:root {
  color-scheme: dark;
  --bg: #0b0f14;
  --fg: #e6edf3;
}

body {
  background: var(--bg);
  color: var(--fg);
}
```

`web/src/main.tsx`:
```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

`web/src/App.tsx`:
```tsx
export default function App() {
  return <main className="p-6 font-mono">WeaponShield AI v2</main>;
}
```

- [ ] **Step 7: Create `web/vercel.json`**

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Cross-Origin-Opener-Policy", "value": "same-origin" },
        { "key": "Cross-Origin-Embedder-Policy", "value": "require-corp" }
      ]
    },
    {
      "source": "/models/(.*)",
      "headers": [{ "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }]
    }
  ],
  "rewrites": [{ "source": "/((?!models/|ort/|assets/).*)", "destination": "/index.html" }]
}
```

- [ ] **Step 8: Create a stub `web/scripts/prepare-assets.mjs` and an empty manifest so the npm hooks run**

`web/models.json`:
```json
{ "models": [] }
```

`web/scripts/prepare-assets.mjs` (full version arrives in Task 3):
```js
console.log('prepare-assets: no models configured yet');
```

- [ ] **Step 9: Verify lint, typecheck and build**

```bash
cd /d/Projects/WeaponShield-AI/web
npm run lint && npm run typecheck && npm run build
```
Expected: all three exit 0; `dist/index.html` exists.

- [ ] **Step 10: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web
git commit -m "feat(web): scaffold Vite + React + TS app with Tailwind, Vitest, ESLint and isolation headers"
```

---

### Task 3: Export the stock YOLO26n model, publish it, and generate golden fixtures

**Files:**
- Create: `training/requirements-export.txt`, `training/scripts/export_onnx.py`, `training/scripts/make_golden.py`
- Create: `fixtures/golden/images/bus.jpg`, `fixtures/golden/images/zidane.jpg`, `fixtures/golden/{bus,zidane}.{lb.rgba,out.f32,json}`
- Modify: `web/models.json`, `web/scripts/prepare-assets.mjs`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `web/public/models/yolo26n-coco.onnx` (downloaded by `npm run predev|prebuild|pretest`)
  - `web/public/ort/ort-wasm*.{wasm,mjs}` (copied from `node_modules/onnxruntime-web/dist`)
  - Golden fixture format: `{name}.json` = `{ "srcW": int, "srcH": int, "size": 640, "ultralytics": [{ "classId": int, "conf": float, "box": [x1,y1,x2,y2] }] }` (boxes in source pixels, conf ≥ 0.25); `{name}.lb.rgba` = 640·640·4 bytes; `{name}.out.f32` = 300·6 little-endian float32.

- [ ] **Step 1: Create the export environment**

`training/requirements-export.txt`:
```text
ultralytics==8.4.174
onnx>=1.17
onnxruntime>=1.20
onnxslim>=0.1.50
```

```bash
cd /d/Projects/WeaponShield-AI/training
python -m venv .venv
.venv/Scripts/python -m pip install -q --upgrade pip
.venv/Scripts/python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -q -r requirements-export.txt
.venv/Scripts/python -c "import ultralytics; print(ultralytics.__version__)"
```
Expected: `8.4.174`

- [ ] **Step 2: Write `training/scripts/export_onnx.py`**

```python
"""Export a YOLO26 checkpoint to end-to-end ONNX for the web app and edge tool.

The web app expects a single output shaped [1, 300, 6] holding
[x1, y1, x2, y2, conf, class_id] in letterboxed input pixels. This script
refuses to produce anything else.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO

EXPECTED_SHAPE = [1, 300, 6]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(weights: str, out: Path, imgsz: int = 640) -> str:
    model = YOLO(weights)
    exported = Path(model.export(format="onnx", imgsz=imgsz, nms=False, opset=17, simplify=True, dynamic=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), out)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    shape = list(session.get_outputs()[0].shape)
    if shape != EXPECTED_SHAPE:
        raise SystemExit(f"Unexpected ONNX output shape {shape}, expected {EXPECTED_SHAPE}")
    return sha256(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="e.g. yolo26n.pt or runs/.../best.pt")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    digest = export(args.weights, args.out, args.imgsz)
    print(f"{args.out} sha256={digest} bytes={args.out.stat().st_size}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Export the stock model**

```bash
cd /d/Projects/WeaponShield-AI/training
mkdir -p ../web/public/models
.venv/Scripts/python scripts/export_onnx.py --weights yolo26n.pt --out ../web/public/models/yolo26n-coco.onnx
```
Expected: one line `../web/public/models/yolo26n-coco.onnx sha256=<64 hex> bytes=<~9-10 million>`. Copy the sha256. If it exits with "Unexpected ONNX output shape", stop and report the shape — the rest of this plan depends on `[1, 300, 6]`.

- [ ] **Step 4: Write `training/scripts/make_golden.py`**

```python
"""Write golden fixtures that pin the browser pipeline to Ultralytics' own results.

For each reference image this saves:
  {name}.lb.rgba  640x640 RGBA after Ultralytics' LetterBox (exact model input)
  {name}.out.f32  raw [1,300,6] float32 output from Python onnxruntime
  {name}.json     source size + Ultralytics predictions (conf >= 0.25, source pixels)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

NAMES = ("bus", "zidane")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    assets = Path(ultralytics.__file__).parent / "assets"

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    model = YOLO(str(args.model), task="detect")

    for name in NAMES:
        src = images_dir / f"{name}.jpg"
        shutil.copyfile(assets / f"{name}.jpg", src)
        bgr = cv2.imread(str(src))
        h, w = bgr.shape[:2]

        letterboxed = LetterBox(new_shape=(args.size, args.size), auto=False)(image=bgr)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        rgba.astype(np.uint8).tofile(args.out / f"{name}.lb.rgba")

        tensor = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        raw = session.run(None, {input_name: tensor})[0].astype("<f4")
        raw.tofile(args.out / f"{name}.out.f32")

        result = model.predict(bgr, imgsz=args.size, conf=0.25, verbose=False)[0]
        dets = [
            {"classId": int(c), "conf": float(s), "box": [float(v) for v in b]}
            for b, s, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
        ]
        meta = {"srcW": w, "srcH": h, "size": args.size, "ultralytics": dets}
        (args.out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        print(f"{name}: {w}x{h}, {len(dets)} detections")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Generate the fixtures**

```bash
cd /d/Projects/WeaponShield-AI/training
.venv/Scripts/python scripts/make_golden.py --model ../web/public/models/yolo26n-coco.onnx --out ../fixtures/golden
ls -la ../fixtures/golden
```
Expected: `bus: 810x1080, N detections` and `zidane: 1280x720, M detections` with N, M ≥ 2; files `bus.lb.rgba` and `zidane.lb.rgba` are exactly 1638400 bytes; `*.out.f32` are exactly 7200 bytes.

- [ ] **Step 6: Publish the model as a GitHub pre-release**

```bash
cd /d/Projects/WeaponShield-AI
gh release create models-v0 web/public/models/yolo26n-coco.onnx \
  --repo Halok600/WeaponShield-AI --target v2 --prerelease \
  --title "models-v0: stock YOLO26n (COCO) for development" \
  --notes "Unmodified Ultralytics YOLO26n COCO weights exported to end-to-end ONNX ([1,300,6]). Used only to develop the browser pipeline before the WeaponShield model is trained. AGPL-3.0."
```
Expected: prints the release URL.

- [ ] **Step 7: Fill in `web/models.json`** (replace `<sha256 from Step 3>` with the real digest)

```json
{
  "models": [
    {
      "file": "yolo26n-coco.onnx",
      "url": "https://github.com/Halok600/WeaponShield-AI/releases/download/models-v0/yolo26n-coco.onnx",
      "sha256": "<sha256 from Step 3>"
    }
  ]
}
```

- [ ] **Step 8: Write the real `web/scripts/prepare-assets.mjs`**

```js
// Downloads models listed in models.json (verified by SHA-256) and copies the
// ONNX Runtime Web wasm binaries into public/ so they are served same-origin.
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

async function fetchModels() {
  const { models } = JSON.parse(await readFile(join(web, 'models.json'), 'utf8'));
  const dir = join(web, 'public', 'models');
  await mkdir(dir, { recursive: true });
  for (const m of models) {
    const dest = join(dir, m.file);
    try {
      if (sha256(await readFile(dest)) === m.sha256) continue;
    } catch {
      // missing file: download below
    }
    console.log(`prepare-assets: downloading ${m.file}`);
    const res = await fetch(m.url);
    if (!res.ok) throw new Error(`${m.url}: HTTP ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    const got = sha256(buf);
    if (got !== m.sha256) throw new Error(`${m.file}: sha256 ${got} does not match manifest ${m.sha256}`);
    await writeFile(dest, buf);
  }
}

async function copyOrtWasm() {
  const src = join(web, 'node_modules', 'onnxruntime-web', 'dist');
  const dest = join(web, 'public', 'ort');
  await mkdir(dest, { recursive: true });
  const files = (await readdir(src)).filter((f) => /^ort-wasm.*\.(wasm|mjs)$/.test(f));
  if (files.length === 0) throw new Error(`No ort-wasm files found in ${src}`);
  await Promise.all(files.map((f) => copyFile(join(src, f), join(dest, f))));
}

await Promise.all([fetchModels(), copyOrtWasm()]);
console.log('prepare-assets: ok');
```

- [ ] **Step 9: Verify the download path from a clean state**

```bash
cd /d/Projects/WeaponShield-AI/web
rm -f public/models/yolo26n-coco.onnx
node scripts/prepare-assets.mjs
ls public/models public/ort | head
```
Expected: `prepare-assets: downloading yolo26n-coco.onnx`, then `prepare-assets: ok`; the model and at least one `ort-wasm-simd-threaded*.wasm` are listed.

- [ ] **Step 10: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add training/requirements-export.txt training/scripts fixtures web/models.json web/scripts/prepare-assets.mjs
git commit -m "feat: export stock YOLO26n to end-to-end ONNX, golden fixtures, model download step"
```

---

### Task 4: Core types, letterbox geometry and preprocessing

**Files:**
- Create: `web/src/core/types.ts`, `web/src/core/geometry.ts`, `web/src/core/preprocess.ts`
- Test: `web/src/core/geometry.test.ts`, `web/src/core/preprocess.test.ts`

**Interfaces:**
- Produces:
  - `type ClassKey = 'person' | 'pistol' | 'rifle' | 'knife'`; `CLASS_KEYS: readonly ClassKey[]`; `isWeapon(cls: ClassKey): boolean`
  - `interface Box { x1: number; y1: number; x2: number; y2: number }`, `interface Point { x: number; y: number }`
  - `interface Detection { cls: ClassKey; conf: number; box: Box }`
  - `interface Track { id: number; cls: ClassKey; conf: number; box: Box; hits: number; missed: number; history: readonly boolean[]; trail: readonly Point[] }`
  - `interface Letterbox { size: number; scale: number; newW: number; newH: number; padX: number; padY: number }`
  - `letterboxGeometry(srcW: number, srcH: number, size: number): Letterbox`
  - `unletterbox(b: Box, lb: Letterbox, srcW: number, srcH: number): Box`
  - `iou(a: Box, b: Box): number`, `center(b: Box): Point`, `area(b: Box): number`
  - `rgbaToTensor(rgba: ArrayLike<number>, size: number, out?: Float32Array): Float32Array`

- [ ] **Step 1: Write the failing tests**

`web/src/core/geometry.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { area, center, iou, letterboxGeometry, unletterbox } from './geometry';

describe('letterboxGeometry', () => {
  it('fits a landscape frame by width and pads top/bottom', () => {
    expect(letterboxGeometry(1280, 720, 640)).toEqual({ size: 640, scale: 0.5, newW: 640, newH: 360, padX: 0, padY: 140 });
  });

  it('fits a portrait frame by height and pads left/right', () => {
    const lb = letterboxGeometry(810, 1080, 640);
    expect(lb.newH).toBe(640);
    expect(lb.newW).toBe(480);
    expect(lb.padX).toBe(80);
    expect(lb.padY).toBe(0);
  });

  it('puts the odd pixel of padding on the bottom/right like Ultralytics', () => {
    const lb = letterboxGeometry(1000, 999, 640);
    expect(lb.newH).toBe(639);
    expect(lb.padY).toBe(0);
  });

  it('scales small frames up', () => {
    expect(letterboxGeometry(320, 320, 640).scale).toBe(2);
  });
});

describe('unletterbox', () => {
  it('maps a letterboxed box back to source pixels', () => {
    const lb = letterboxGeometry(1280, 720, 640);
    expect(unletterbox({ x1: 100, y1: 190, x2: 300, y2: 290 }, lb, 1280, 720)).toEqual({ x1: 200, y1: 100, x2: 600, y2: 300 });
  });

  it('clamps boxes that spill into the padding', () => {
    const lb = letterboxGeometry(1280, 720, 640);
    const b = unletterbox({ x1: -10, y1: 100, x2: 700, y2: 600 }, lb, 1280, 720);
    expect(b).toEqual({ x1: 0, y1: 0, x2: 1280, y2: 720 });
  });
});

describe('box helpers', () => {
  it('computes IoU', () => {
    expect(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 0, y1: 0, x2: 10, y2: 10 })).toBe(1);
    expect(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 20, y1: 20, x2: 30, y2: 30 })).toBe(0);
    expect(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 5, y1: 0, x2: 15, y2: 10 })).toBeCloseTo(50 / 150);
  });

  it('computes centre and area', () => {
    expect(center({ x1: 0, y1: 10, x2: 20, y2: 30 })).toEqual({ x: 10, y: 20 });
    expect(area({ x1: 0, y1: 0, x2: 4, y2: 5 })).toBe(20);
    expect(area({ x1: 5, y1: 5, x2: 1, y2: 1 })).toBe(0);
  });
});
```

`web/src/core/preprocess.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { rgbaToTensor } from './preprocess';

describe('rgbaToTensor', () => {
  it('converts RGBA bytes to planar RGB floats in [0,1] and drops alpha', () => {
    // 2x2 image: red, green, blue, white
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]);
    const t = rgbaToTensor(rgba, 2);
    expect(Array.from(t)).toEqual([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]);
  });

  it('reuses the provided output buffer', () => {
    const out = new Float32Array(3);
    expect(rgbaToTensor(new Uint8ClampedArray([51, 102, 204, 255]), 1, out)).toBe(out);
    expect(out[0]).toBeCloseTo(0.2);
  });

  it('rejects input of the wrong length', () => {
    expect(() => rgbaToTensor(new Uint8ClampedArray(8), 2)).toThrow(/expected 16/);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/core/geometry.test.ts src/core/preprocess.test.ts`
Expected: FAIL — cannot resolve `./geometry` / `./preprocess`.

- [ ] **Step 3: Implement**

`web/src/core/types.ts`:
```ts
export type ClassKey = 'person' | 'pistol' | 'rifle' | 'knife';

export const CLASS_KEYS: readonly ClassKey[] = ['person', 'pistol', 'rifle', 'knife'];

export const isWeapon = (cls: ClassKey): boolean => cls !== 'person';

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface Detection {
  cls: ClassKey;
  conf: number;
  box: Box;
}

/** A tracked object as seen by consumers. `history` holds one entry per processed frame (true = matched). */
export interface Track {
  id: number;
  cls: ClassKey;
  conf: number;
  box: Box;
  hits: number;
  missed: number;
  history: readonly boolean[];
  trail: readonly Point[];
}
```

`web/src/core/geometry.ts`:
```ts
import type { Box, Point } from './types';

export interface Letterbox {
  size: number;
  scale: number;
  newW: number;
  newH: number;
  padX: number;
  padY: number;
}

/** Same geometry as Ultralytics LetterBox(auto=False): centred, odd padding pixel on the bottom/right. */
export function letterboxGeometry(srcW: number, srcH: number, size: number): Letterbox {
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  return { size, scale, newW, newH, padX: Math.floor((size - newW) / 2), padY: Math.floor((size - newH) / 2) };
}

const clamp = (v: number, max: number) => Math.min(max, Math.max(0, v));

export function unletterbox(b: Box, lb: Letterbox, srcW: number, srcH: number): Box {
  return {
    x1: clamp((b.x1 - lb.padX) / lb.scale, srcW),
    y1: clamp((b.y1 - lb.padY) / lb.scale, srcH),
    x2: clamp((b.x2 - lb.padX) / lb.scale, srcW),
    y2: clamp((b.y2 - lb.padY) / lb.scale, srcH),
  };
}

export const area = (b: Box): number => Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);

export const center = (b: Box): Point => ({ x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 });

export function iou(a: Box, b: Box): number {
  const inter = area({ x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1), x2: Math.min(a.x2, b.x2), y2: Math.min(a.y2, b.y2) });
  const union = area(a) + area(b) - inter;
  return union > 0 ? inter / union : 0;
}
```

`web/src/core/preprocess.ts`:
```ts
/** RGBA bytes (row-major, size x size) -> planar RGB float32 in [0,1] (NCHW without the batch dim). */
export function rgbaToTensor(rgba: ArrayLike<number>, size: number, out?: Float32Array): Float32Array {
  const pixels = size * size;
  if (rgba.length !== pixels * 4) {
    throw new Error(`rgbaToTensor: got ${rgba.length} bytes, expected ${pixels * 4}`);
  }
  const t = out ?? new Float32Array(pixels * 3);
  for (let i = 0, p = 0; i < pixels; i++, p += 4) {
    t[i] = rgba[p]! / 255;
    t[i + pixels] = rgba[p + 1]! / 255;
    t[i + 2 * pixels] = rgba[p + 2]! / 255;
  }
  return t;
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/core/geometry.test.ts src/core/preprocess.test.ts`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/core
git commit -m "feat(core): types, Ultralytics-compatible letterbox geometry and RGBA-to-tensor"
```

---

### Task 5: Postprocessing and model configuration

**Files:**
- Create: `web/src/core/postprocess.ts`, `web/src/config/models.ts`
- Test: `web/src/core/postprocess.test.ts`

**Interfaces:**
- Consumes: `Letterbox`, `unletterbox`, `area` (Task 4); `ClassKey`, `Detection`, `CLASS_KEYS`
- Produces:
  - `type ClassThresholds = Readonly<Record<ClassKey, number>>`
  - `uniformThresholds(conf: number): ClassThresholds`
  - `decodeEndToEnd(data: ArrayLike<number>, dims: readonly number[], labels: Readonly<Record<number, ClassKey>>, thresholds: ClassThresholds, lb: Letterbox, srcW: number, srcH: number): Detection[]` — sorted by confidence, highest first
  - `interface ModelConfig { id: string; url: string; inputSize: number; labels: Readonly<Record<number, ClassKey>> }`
  - `STOCK_COCO_MODEL: ModelConfig`, `ACTIVE_MODEL: ModelConfig`

- [ ] **Step 1: Write the failing test**

`web/src/core/postprocess.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { letterboxGeometry } from './geometry';
import { decodeEndToEnd, uniformThresholds } from './postprocess';

const lb = letterboxGeometry(1280, 720, 640); // scale 0.5, padY 140
const labels = { 0: 'person', 43: 'knife' } as const;

function rows(...r: number[][]): Float32Array {
  const out = new Float32Array(300 * 6);
  r.forEach((row, i) => out.set(row, i * 6));
  return out;
}

describe('decodeEndToEnd', () => {
  it('keeps mapped classes above threshold, maps boxes to source pixels, sorts by confidence', () => {
    const data = rows([100, 190, 300, 290, 0.6, 0], [10, 150, 20, 160, 0.9, 43]);
    const dets = decodeEndToEnd(data, [1, 300, 6], labels, uniformThresholds(0.25), lb, 1280, 720);
    expect(dets).toEqual([
      { cls: 'knife', conf: expect.closeTo(0.9, 5), box: { x1: 20, y1: 20, x2: 40, y2: 40 } },
      { cls: 'person', conf: expect.closeTo(0.6, 5), box: { x1: 200, y1: 100, x2: 600, y2: 300 } },
    ]);
  });

  it('drops unmapped classes and low-confidence rows', () => {
    const data = rows([100, 190, 300, 290, 0.9, 2], [100, 190, 300, 290, 0.2, 0]);
    expect(decodeEndToEnd(data, [1, 300, 6], labels, uniformThresholds(0.25), lb, 1280, 720)).toEqual([]);
  });

  it('applies per-class thresholds', () => {
    const data = rows([100, 190, 300, 290, 0.5, 0], [100, 190, 300, 290, 0.5, 43]);
    const thresholds = { ...uniformThresholds(0.25), knife: 0.6 };
    const dets = decodeEndToEnd(data, [1, 300, 6], labels, thresholds, lb, 1280, 720);
    expect(dets.map((d) => d.cls)).toEqual(['person']);
  });

  it('drops boxes that collapse to under a pixel after clamping', () => {
    const data = rows([0, 0, 640, 100, 0.9, 0]); // entirely inside the top padding
    expect(decodeEndToEnd(data, [1, 300, 6], labels, uniformThresholds(0.25), lb, 1280, 720)).toEqual([]);
  });

  it('rejects unexpected output shapes', () => {
    expect(() => decodeEndToEnd(new Float32Array(84 * 8400), [1, 84, 8400], labels, uniformThresholds(0.25), lb, 1280, 720)).toThrow(
      /\[1,84,8400\]/,
    );
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/core/postprocess.test.ts`
Expected: FAIL — cannot resolve `./postprocess`.

- [ ] **Step 3: Implement**

`web/src/core/postprocess.ts`:
```ts
import { type Letterbox, unletterbox } from './geometry';
import { CLASS_KEYS, type ClassKey, type Detection } from './types';

export type ClassThresholds = Readonly<Record<ClassKey, number>>;

export const uniformThresholds = (conf: number): ClassThresholds =>
  Object.fromEntries(CLASS_KEYS.map((k) => [k, conf])) as Record<ClassKey, number>;

/** Decode an end-to-end YOLO26 output ([1, N, 6] = x1,y1,x2,y2,conf,class in letterboxed pixels). */
export function decodeEndToEnd(
  data: ArrayLike<number>,
  dims: readonly number[],
  labels: Readonly<Record<number, ClassKey>>,
  thresholds: ClassThresholds,
  lb: Letterbox,
  srcW: number,
  srcH: number,
): Detection[] {
  if (dims.length !== 3 || dims[0] !== 1 || dims[2] !== 6) {
    throw new Error(`Unexpected model output shape [${dims.join(',')}], expected [1,N,6]`);
  }
  const n = dims[1]!;
  const dets: Detection[] = [];
  for (let i = 0; i < n; i++) {
    const o = i * 6;
    const conf = data[o + 4]!;
    const cls = labels[Math.round(data[o + 5]!)];
    if (cls === undefined || conf < thresholds[cls]) continue;
    const box = unletterbox({ x1: data[o]!, y1: data[o + 1]!, x2: data[o + 2]!, y2: data[o + 3]! }, lb, srcW, srcH);
    if (box.x2 - box.x1 < 1 || box.y2 - box.y1 < 1) continue;
    dets.push({ cls, conf, box });
  }
  return dets.sort((a, b) => b.conf - a.conf);
}
```

`web/src/config/models.ts`:
```ts
import type { ClassKey } from '../core/types';

export interface ModelConfig {
  id: string;
  /** Same-origin URL; files are placed in public/models by scripts/prepare-assets.mjs. */
  url: string;
  inputSize: number;
  /** Model class id -> app class. Ids not listed are ignored. */
  labels: Readonly<Record<number, ClassKey>>;
}

/** Stock Ultralytics YOLO26n trained on COCO: only person (0) and knife (43) are relevant. Development only. */
export const STOCK_COCO_MODEL: ModelConfig = {
  id: 'yolo26n-coco',
  url: '/models/yolo26n-coco.onnx',
  inputSize: 640,
  labels: { 0: 'person', 43: 'knife' },
};

export const ACTIVE_MODEL: ModelConfig = STOCK_COCO_MODEL;
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/core/postprocess.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/core/postprocess.ts web/src/core/postprocess.test.ts web/src/config
git commit -m "feat(core): decode end-to-end YOLO26 output with per-class thresholds; model config"
```

---

### Task 6: Golden parity test against Ultralytics

**Files:**
- Test: `web/src/core/golden.test.ts`

**Interfaces:**
- Consumes: `rgbaToTensor` (Task 4); `decodeEndToEnd`, `uniformThresholds` (Task 5); `letterboxGeometry`, `iou`; fixtures from Task 3; `web/public/models/yolo26n-coco.onnx`
- Produces: nothing new (guards the pipeline)

- [ ] **Step 1: Write the test**

`web/src/core/golden.test.ts`:
```ts
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';
import { beforeAll, describe, expect, it } from 'vitest';
import { iou, letterboxGeometry } from './geometry';
import { decodeEndToEnd, uniformThresholds } from './postprocess';
import { rgbaToTensor } from './preprocess';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const golden = (f: string) => `${root}fixtures/golden/${f}`;
const MODEL = `${root}web/public/models/yolo26n-coco.onnx`;

interface Meta {
  srcW: number;
  srcH: number;
  size: number;
  ultralytics: { classId: number; conf: number; box: [number, number, number, number] }[];
}

let session: ort.InferenceSession;
beforeAll(async () => {
  session = await ort.InferenceSession.create(MODEL);
});

describe.each(['bus', 'zidane'])('golden parity: %s', (name) => {
  const meta = JSON.parse(readFileSync(golden(`${name}.json`), 'utf8')) as Meta;
  const rgba = new Uint8Array(readFileSync(golden(`${name}.lb.rgba`)));
  const pyRaw = new Float32Array(new Uint8Array(readFileSync(golden(`${name}.out.f32`))).buffer);

  async function run() {
    const tensor = new ort.Tensor('float32', rgbaToTensor(rgba, meta.size), [1, 3, meta.size, meta.size]);
    const out = await session.run({ [session.inputNames[0]!]: tensor });
    return out[session.outputNames[0]!]!;
  }

  it('raw output matches Python onnxruntime', async () => {
    const out = await run();
    expect(out.dims).toEqual([1, 300, 6]);
    const data = out.data as Float32Array;
    for (let i = 0; i < 300; i++) {
      if (pyRaw[i * 6 + 4]! < 0.25) continue;
      for (let j = 0; j < 6; j++) expect(Math.abs(data[i * 6 + j]! - pyRaw[i * 6 + j]!)).toBeLessThan(j < 4 ? 1e-2 : 1e-3);
    }
  });

  it('decoded person boxes match Ultralytics predictions (IoU >= 0.95, |dconf| <= 0.02)', async () => {
    const out = await run();
    const lb = letterboxGeometry(meta.srcW, meta.srcH, meta.size);
    const ours = decodeEndToEnd(out.data as Float32Array, out.dims, { 0: 'person' }, uniformThresholds(0.25), lb, meta.srcW, meta.srcH);
    const theirs = meta.ultralytics.filter((d) => d.classId === 0);
    expect(theirs.length).toBeGreaterThan(0);
    expect(ours).toHaveLength(theirs.length);
    for (const t of theirs) {
      const [x1, y1, x2, y2] = t.box;
      const best = ours.reduce((a, b) => (iou(b.box, { x1, y1, x2, y2 }) > iou(a.box, { x1, y1, x2, y2 }) ? b : a));
      expect(iou(best.box, { x1, y1, x2, y2 })).toBeGreaterThanOrEqual(0.95);
      expect(Math.abs(best.conf - t.conf)).toBeLessThanOrEqual(0.02);
    }
  });
});
```

- [ ] **Step 2: Run it**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/core/golden.test.ts`
Expected: PASS (4 tests). If the IoU assertion fails while the raw-output test passes, the bug is in `letterboxGeometry`/`unletterbox` — compare against `ultralytics/utils/ops.py::scale_boxes` and fix the geometry, not the tolerance.

- [ ] **Step 3: Run the whole suite**

Run: `cd /d/Projects/WeaponShield-AI/web && npm test`
Expected: PASS (all tests so far).

- [ ] **Step 4: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/core/golden.test.ts
git commit -m "test(core): golden parity with Ultralytics on reference images"
```

---

### Task 7: Kalman filter and ByteTrack-lite tracker

**Files:**
- Create: `web/src/core/kalman.ts`, `web/src/core/tracker.ts`
- Test: `web/src/core/kalman.test.ts`, `web/src/core/tracker.test.ts`

**Interfaces:**
- Consumes: `Box`, `Detection`, `Track`, `Point`, `ClassKey`, `isWeapon` (Task 4); `iou`, `center`
- Produces:
  - `class Kalman1D { x: number; v: number; constructor(x0: number, posVar: number, velVar: number); predict(qPos: number, qVel: number): number; update(z: number, r: number): void }`
  - `class BoxKalman { constructor(b: Box); predict(): Box; update(b: Box): void; box(): Box }`
  - `interface TrackerOptions { highConf: Readonly<Record<ClassKey, number>>; lowConf: number; firstIou: number; secondIou: number; minHits: number; maxMissed: number; historyLength: number; trailLength: number }`
  - `DEFAULT_TRACKER_OPTIONS: TrackerOptions`
  - `class Tracker { constructor(opts?: Partial<TrackerOptions>); setOptions(opts: Partial<TrackerOptions>): void; update(dets: readonly Detection[]): Track[]; reset(): void }` — `update` returns visible tracks only (`hits >= minHits && missed === 0`), sorted by id.

- [ ] **Step 1: Write the failing tests**

`web/src/core/kalman.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { BoxKalman, Kalman1D } from './kalman';

describe('Kalman1D', () => {
  it('learns a constant velocity', () => {
    const k = new Kalman1D(0, 1, 1);
    for (let t = 1; t <= 20; t++) {
      k.predict(0.01, 0.01);
      k.update(t * 3, 0.1);
    }
    expect(Math.abs(k.v - 3)).toBeLessThan(0.2);
    expect(Math.abs(k.predict(0.01, 0.01) - 63)).toBeLessThan(1);
  });
});

describe('BoxKalman', () => {
  it('predicts where a moving box goes next', () => {
    const kf = new BoxKalman({ x1: 0, y1: 0, x2: 50, y2: 100 });
    for (let t = 1; t <= 10; t++) {
      kf.predict();
      kf.update({ x1: t * 5, y1: 0, x2: 50 + t * 5, y2: 100 });
    }
    const p = kf.predict();
    expect(Math.abs(p.x1 - 55)).toBeLessThan(2);
    expect(Math.abs(p.x2 - 105)).toBeLessThan(2);
    expect(Math.abs(p.y2 - p.y1 - 100)).toBeLessThan(1);
  });
});
```

`web/src/core/tracker.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { Tracker } from './tracker';
import type { Box, ClassKey, Detection } from './types';

const det = (cls: ClassKey, conf: number, box: Box): Detection => ({ cls, conf, box });
const box = (x: number, y: number, w = 50, h = 100): Box => ({ x1: x, y1: y, x2: x + w, y2: y + h });

describe('Tracker', () => {
  it('waits for minHits before reporting a new track', () => {
    const tr = new Tracker();
    expect(tr.update([det('person', 0.9, box(0, 0))])).toEqual([]);
    expect(tr.update([det('person', 0.9, box(2, 0))])).toHaveLength(1);
  });

  it('keeps one id for an object moving steadily', () => {
    const tr = new Tracker();
    const ids = new Set<number>();
    for (let t = 0; t < 15; t++) for (const track of tr.update([det('person', 0.9, box(t * 6, 0))])) ids.add(track.id);
    expect([...ids]).toEqual([1]);
  });

  it('keeps ids apart for two people walking past each other', () => {
    const tr = new Tracker();
    let last: { id: number; x: number }[] = [];
    for (let t = 0; t < 20; t++) {
      const out = tr.update([det('person', 0.9, box(t * 10, 0)), det('person', 0.9, box(200 - t * 10, 40))]);
      last = out.map((o) => ({ id: o.id, x: o.box.x1 }));
    }
    const left = last.find((l) => l.x > 150)!; // the one that started on the left is now on the right
    expect(left.id).toBe(1);
  });

  it('re-acquires the same id after a short occlusion', () => {
    const tr = new Tracker();
    for (let t = 0; t < 5; t++) tr.update([det('person', 0.9, box(t * 5, 0))]);
    for (let t = 5; t < 10; t++) expect(tr.update([])).toEqual([]);
    const out = tr.update([det('person', 0.9, box(50, 0))]);
    expect(out.map((o) => o.id)).toEqual([1]);
    expect(out[0]!.history.slice(-6)).toEqual([false, false, false, false, false, true]);
  });

  it('uses low-confidence detections to continue a track but never to start one', () => {
    const tr = new Tracker();
    tr.update([det('pistol', 0.9, box(0, 0, 20, 20))]);
    tr.update([det('pistol', 0.9, box(1, 0, 20, 20))]);
    expect(tr.update([det('pistol', 0.2, box(2, 0, 20, 20))]).map((t) => t.id)).toEqual([1]);
    expect(tr.update([det('pistol', 0.2, box(300, 300, 20, 20))])).toEqual([]);
    expect(tr.update([det('pistol', 0.2, box(301, 300, 20, 20))])).toEqual([]);
  });

  it('never matches a person detection to a weapon track', () => {
    const tr = new Tracker();
    tr.update([det('knife', 0.9, box(0, 0))]);
    tr.update([det('knife', 0.9, box(0, 0))]);
    const out = tr.update([det('person', 0.9, box(0, 0))]);
    expect(out).toEqual([]); // knife track missed, person track is new (1 hit)
  });

  it('chooses the class by confidence-weighted vote within weapons', () => {
    const tr = new Tracker();
    const seq: ClassKey[] = ['pistol', 'pistol', 'rifle', 'pistol'];
    let out = tr.update([]);
    for (const cls of seq) out = tr.update([det(cls, 0.8, box(0, 0, 30, 30))]);
    expect(out[0]!.cls).toBe('pistol');
  });

  it('drops tracks missed for more than maxMissed frames', () => {
    const tr = new Tracker({ maxMissed: 3 });
    tr.update([det('person', 0.9, box(0, 0))]);
    tr.update([det('person', 0.9, box(0, 0))]);
    for (let i = 0; i < 4; i++) tr.update([]);
    tr.update([det('person', 0.9, box(0, 0))]);
    expect(tr.update([det('person', 0.9, box(0, 0))])[0]!.id).toBe(2);
  });

  it('records a trail of centres', () => {
    const tr = new Tracker();
    let out = tr.update([]);
    for (let t = 0; t < 4; t++) out = tr.update([det('person', 0.9, box(t * 10, 0))]);
    expect(out[0]!.trail.length).toBe(4);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/core/kalman.test.ts src/core/tracker.test.ts`
Expected: FAIL — cannot resolve `./kalman` / `./tracker`.

- [ ] **Step 3: Implement `web/src/core/kalman.ts`**

```ts
import type { Box } from './types';

/** Constant-velocity Kalman filter for one coordinate. State [x, v]; F = [[1,1],[0,1]], H = [1,0]. */
export class Kalman1D {
  x: number;
  v = 0;
  private p00: number;
  private p01 = 0;
  private p10 = 0;
  private p11: number;

  constructor(x0: number, posVar: number, velVar: number) {
    this.x = x0;
    this.p00 = posVar;
    this.p11 = velVar;
  }

  predict(qPos: number, qVel: number): number {
    this.x += this.v;
    // P = F P F^T + Q
    const p00 = this.p00 + this.p01 + this.p10 + this.p11 + qPos;
    const p01 = this.p01 + this.p11;
    const p10 = this.p10 + this.p11;
    const p11 = this.p11 + qVel;
    this.p00 = p00;
    this.p01 = p01;
    this.p10 = p10;
    this.p11 = p11;
    return this.x;
  }

  update(z: number, r: number): void {
    const s = this.p00 + r;
    const k0 = this.p00 / s;
    const k1 = this.p10 / s;
    const y = z - this.x;
    this.x += k0 * y;
    this.v += k1 * y;
    const p00 = (1 - k0) * this.p00;
    const p01 = (1 - k0) * this.p01;
    const p10 = this.p10 - k1 * this.p00;
    const p11 = this.p11 - k1 * this.p01;
    this.p00 = p00;
    this.p01 = p01;
    this.p10 = p10;
    this.p11 = p11;
  }
}

// Noise weights from ByteTrack's KalmanFilter, scaled by box height.
const POS_W = 1 / 20;
const VEL_W = 1 / 160;

/** Four independent constant-velocity filters over (cx, cy, w, h). */
export class BoxKalman {
  private readonly f: Kalman1D[];

  constructor(b: Box) {
    const s = Math.max(b.y2 - b.y1, 1);
    const posVar = (2 * POS_W * s) ** 2;
    const velVar = (10 * VEL_W * s) ** 2;
    this.f = toCxcywh(b).map((v) => new Kalman1D(v, posVar, velVar));
  }

  predict(): Box {
    const h = Math.max(this.f[3]!.x, 1);
    const q = (POS_W * h) ** 2;
    const qv = (VEL_W * h) ** 2;
    for (const k of this.f) k.predict(q, qv);
    return this.box();
  }

  update(b: Box): void {
    const z = toCxcywh(b);
    const r = (POS_W * Math.max(z[3]!, 1)) ** 2;
    this.f.forEach((k, i) => k.update(z[i]!, r));
  }

  box(): Box {
    const [cx, cy, w, h] = this.f.map((k) => k.x) as [number, number, number, number];
    const hw = Math.max(w, 1) / 2;
    const hh = Math.max(h, 1) / 2;
    return { x1: cx - hw, y1: cy - hh, x2: cx + hw, y2: cy + hh };
  }
}

function toCxcywh(b: Box): number[] {
  return [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, b.x2 - b.x1, b.y2 - b.y1];
}
```

- [ ] **Step 4: Implement `web/src/core/tracker.ts`**

```ts
import { center, iou } from './geometry';
import { BoxKalman } from './kalman';
import { type Box, type ClassKey, type Detection, isWeapon, type Point, type Track } from './types';

export interface TrackerOptions {
  /** A detection at or above this starts tracks and joins the first association pass. */
  highConf: Readonly<Record<ClassKey, number>>;
  /** Detections between lowConf and highConf can only continue tracks seen last frame. */
  lowConf: number;
  firstIou: number;
  secondIou: number;
  minHits: number;
  maxMissed: number;
  historyLength: number;
  trailLength: number;
}

export const DEFAULT_TRACKER_OPTIONS: TrackerOptions = {
  highConf: { person: 0.5, pistol: 0.45, rifle: 0.45, knife: 0.45 },
  lowConf: 0.1,
  firstIou: 0.2,
  secondIou: 0.4,
  minHits: 2,
  maxMissed: 30,
  historyLength: 30,
  trailLength: 30,
};

interface InternalTrack {
  id: number;
  kf: BoxKalman;
  predicted: Box;
  votes: Map<ClassKey, number>;
  cls: ClassKey;
  conf: number;
  hits: number;
  missed: number;
  history: boolean[];
  trail: Point[];
}

interface Matching {
  pairs: [number, number][];
  unmatchedTracks: number[];
  unmatchedDets: number[];
}

const sameGroup = (a: ClassKey, b: ClassKey) => isWeapon(a) === isWeapon(b);

function pushCapped<T>(arr: T[], v: T, cap: number): void {
  arr.push(v);
  if (arr.length > cap) arr.splice(0, arr.length - cap);
}

/** Greedy IoU matching (highest IoU first). Persons and weapons never match each other. */
function greedyMatch(tracks: readonly InternalTrack[], dets: readonly Detection[], minIou: number): Matching {
  const candidates: { t: number; d: number; v: number }[] = [];
  tracks.forEach((t, ti) =>
    dets.forEach((d, di) => {
      if (!sameGroup(t.cls, d.cls)) return;
      const v = iou(t.predicted, d.box);
      if (v >= minIou) candidates.push({ t: ti, d: di, v });
    }),
  );
  candidates.sort((a, b) => b.v - a.v);
  const usedT = new Set<number>();
  const usedD = new Set<number>();
  const pairs: [number, number][] = [];
  for (const c of candidates) {
    if (usedT.has(c.t) || usedD.has(c.d)) continue;
    usedT.add(c.t);
    usedD.add(c.d);
    pairs.push([c.t, c.d]);
  }
  return {
    pairs,
    unmatchedTracks: tracks.map((_, i) => i).filter((i) => !usedT.has(i)),
    unmatchedDets: dets.map((_, i) => i).filter((i) => !usedD.has(i)),
  };
}

export class Tracker {
  private opts: TrackerOptions;
  private tracks: InternalTrack[] = [];
  private nextId = 1;

  constructor(opts: Partial<TrackerOptions> = {}) {
    this.opts = { ...DEFAULT_TRACKER_OPTIONS, ...opts };
  }

  setOptions(opts: Partial<TrackerOptions>): void {
    this.opts = { ...this.opts, ...opts };
  }

  reset(): void {
    this.tracks = [];
    this.nextId = 1;
  }

  update(dets: readonly Detection[]): Track[] {
    const o = this.opts;
    for (const t of this.tracks) t.predicted = t.kf.predict();

    const high = dets.filter((d) => d.conf >= o.highConf[d.cls]);
    const low = dets.filter((d) => d.conf >= o.lowConf && d.conf < o.highConf[d.cls]);

    const first = greedyMatch(this.tracks, high, o.firstIou);
    for (const [ti, di] of first.pairs) this.apply(this.tracks[ti]!, high[di]!);

    const remaining = first.unmatchedTracks.map((i) => this.tracks[i]!);
    const seenLastFrame = remaining.filter((t) => t.missed === 0 && t.hits >= o.minHits);
    const second = greedyMatch(seenLastFrame, low, o.secondIou);
    const rescued = new Set<InternalTrack>();
    for (const [ti, di] of second.pairs) {
      this.apply(seenLastFrame[ti]!, low[di]!);
      rescued.add(seenLastFrame[ti]!);
    }

    for (const t of remaining) {
      if (rescued.has(t)) continue;
      t.missed += 1;
      pushCapped(t.history, false, o.historyLength);
    }

    // Confirmed tracks survive maxMissed frames; tentative ones die on their first miss.
    this.tracks = this.tracks.filter((t) => (t.hits >= o.minHits ? t.missed <= o.maxMissed : t.missed === 0));

    for (const di of first.unmatchedDets) this.spawn(high[di]!);

    return this.tracks
      .filter((t) => t.missed === 0 && t.hits >= o.minHits)
      .sort((a, b) => a.id - b.id)
      .map((t) => ({
        id: t.id,
        cls: t.cls,
        conf: t.conf,
        box: t.kf.box(),
        hits: t.hits,
        missed: t.missed,
        history: [...t.history],
        trail: [...t.trail],
      }));
  }

  private apply(t: InternalTrack, d: Detection): void {
    t.kf.update(d.box);
    t.hits += 1;
    t.missed = 0;
    t.conf = 0.6 * t.conf + 0.4 * d.conf;
    t.votes.set(d.cls, (t.votes.get(d.cls) ?? 0) + d.conf);
    let best = t.cls;
    for (const [cls, score] of t.votes) if (score > (t.votes.get(best) ?? 0)) best = cls;
    t.cls = best;
    pushCapped(t.history, true, this.opts.historyLength);
    pushCapped(t.trail, center(t.kf.box()), this.opts.trailLength);
  }

  private spawn(d: Detection): void {
    this.tracks.push({
      id: this.nextId++,
      kf: new BoxKalman(d.box),
      predicted: d.box,
      votes: new Map([[d.cls, d.conf]]),
      cls: d.cls,
      conf: d.conf,
      hits: 1,
      missed: 0,
      history: [true],
      trail: [center(d.box)],
    });
  }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/core/kalman.test.ts src/core/tracker.test.ts`
Expected: PASS (11 tests). If "keeps ids apart" fails, print both tracks' boxes per frame and check that `predicted` (not the last measured box) is used in `greedyMatch`.

- [ ] **Step 6: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/core/kalman.ts web/src/core/kalman.test.ts web/src/core/tracker.ts web/src/core/tracker.test.ts
git commit -m "feat(core): Kalman box filter and ByteTrack-lite tracker"
```

---

### Task 8: Threat engine, presets and pipeline

**Files:**
- Create: `web/src/core/threat.ts`, `web/src/core/presets.ts`, `web/src/core/pipeline.ts`
- Test: `web/src/core/threat.test.ts`, `web/src/core/pipeline.test.ts`

**Interfaces:**
- Consumes: `Tracker`, `TrackerOptions` (Task 7); `Track`, `Detection`, `ClassKey`, `isWeapon`, `center`
- Produces:
  - `interface ThreatOptions { confirmK: number; confirmN: number; armedExpand: number; eventGapMs: number }`, `DEFAULT_THREAT_OPTIONS`
  - `interface ThreatEvent { id: number; weaponTrackId: number; cls: ClassKey; personTrackId: number | null; startMs: number; endMs: number; maxConf: number; ended: boolean }`
  - `interface ThreatFrame { confirmedWeaponIds: number[]; armedPersonIds: number[]; weaponOwner: ReadonlyMap<number, number>; active: ThreatEvent[]; started: ThreatEvent[]; ended: ThreatEvent[] }`
  - `isConfirmed(history: readonly boolean[], k: number, n: number): boolean`
  - `findOwner(weapon: Track, persons: readonly Track[], expand: number): number | null`
  - `class ThreatEngine { constructor(opts?: Partial<ThreatOptions>); setOptions(opts: Partial<ThreatOptions>): void; update(tracks: readonly Track[], tMs: number): ThreatFrame; events(): ThreatEvent[]; reset(): void }` — all returned events are copies
  - `type SensitivityName = 'low' | 'balanced' | 'high'`; `interface Preset { lowConf: number; highConf: Record<ClassKey, number>; imageConf: Record<ClassKey, number>; confirmK: number; confirmN: number }`; `PRESETS: Record<SensitivityName, Preset>`
  - `filterForImage(dets: readonly Detection[], conf: Readonly<Record<ClassKey, number>>): Detection[]`
  - `interface PipelineFrame { tracks: Track[]; threat: ThreatFrame }`; `class Pipeline { constructor(preset: Preset); setPreset(p: Preset): void; process(dets: readonly Detection[], tMs: number): PipelineFrame; reset(): void }`

- [ ] **Step 1: Write the failing tests**

`web/src/core/threat.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { findOwner, isConfirmed, ThreatEngine } from './threat';
import type { Box, ClassKey, Track } from './types';

const track = (id: number, cls: ClassKey, box: Box, history: boolean[] = [true], conf = 0.8): Track => ({
  id,
  cls,
  conf,
  box,
  hits: history.filter(Boolean).length,
  missed: 0,
  history,
  trail: [],
});
const person = (id: number, x = 0) => track(id, 'person', { x1: x, y1: 0, x2: x + 100, y2: 200 });
const gun = (id: number, history: boolean[], x = 80, conf = 0.8) => track(id, 'pistol', { x1: x, y1: 90, x2: x + 30, y2: 110 }, history, conf);
const confirmedHistory = [true, true, true, true];

describe('isConfirmed', () => {
  it('needs K hits within the last N frames', () => {
    expect(isConfirmed([true, true, true, true], 4, 10)).toBe(true);
    expect(isConfirmed([true, false, true, false, true], 4, 10)).toBe(false);
    expect(isConfirmed([true, true, true, true, false, false, false, false, false, false, false], 4, 10)).toBe(false);
  });
});

describe('findOwner', () => {
  it('returns the person whose expanded box contains the weapon centre', () => {
    expect(findOwner(gun(9, [true], 105), [person(1), person(2, 300)], 0.2)).toBe(1);
  });
  it('picks the nearest person when several qualify', () => {
    // weapon centre x=105 lies inside both expanded boxes; person 2's centre (110) is nearer than person 1's (50)
    expect(findOwner(gun(9, [true], 90), [person(1, 0), person(2, 60)], 0.2)).toBe(2);
  });
  it('returns null when nobody holds it', () => {
    expect(findOwner(gun(9, [true], 600), [person(1)], 0.2)).toBeNull();
  });
});

describe('ThreatEngine', () => {
  it('does not raise an event before confirmation', () => {
    const te = new ThreatEngine();
    const f = te.update([gun(5, [true, true])], 0);
    expect(f.started).toEqual([]);
    expect(f.confirmedWeaponIds).toEqual([]);
  });

  it('starts one event on confirmation and marks the holder ARMED', () => {
    const te = new ThreatEngine();
    const f = te.update([person(1), gun(5, confirmedHistory)], 1000);
    expect(f.confirmedWeaponIds).toEqual([5]);
    expect(f.armedPersonIds).toEqual([1]);
    expect(f.weaponOwner.get(5)).toBe(1);
    expect(f.started).toEqual([
      { id: 1, weaponTrackId: 5, cls: 'pistol', personTrackId: 1, startMs: 1000, endMs: 1000, maxConf: 0.8, ended: false },
    ]);
    expect(te.update([person(1), gun(5, confirmedHistory)], 1100).started).toEqual([]);
  });

  it('extends the event while visible and tracks the peak confidence', () => {
    const te = new ThreatEngine();
    te.update([gun(5, confirmedHistory, 80, 0.7)], 0);
    const f = te.update([gun(5, confirmedHistory, 80, 0.95)], 500);
    expect(f.active[0]).toMatchObject({ endMs: 500, maxConf: 0.95 });
  });

  it('ends the event after the weapon is gone for longer than eventGapMs', () => {
    const te = new ThreatEngine({ eventGapMs: 2000 });
    te.update([gun(5, confirmedHistory)], 0);
    expect(te.update([], 1500).ended).toEqual([]);
    const f = te.update([], 2500);
    expect(f.ended).toHaveLength(1);
    expect(f.ended[0]).toMatchObject({ id: 1, ended: true, endMs: 0 });
    expect(f.active).toEqual([]);
    expect(te.events()).toHaveLength(1);
  });

  it('returns copies so callers cannot mutate engine state', () => {
    const te = new ThreatEngine();
    const f = te.update([gun(5, confirmedHistory)], 0);
    f.started[0]!.maxConf = 0;
    expect(te.events()[0]!.maxConf).toBe(0.8);
  });
});
```

`web/src/core/pipeline.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { Pipeline } from './pipeline';
import { filterForImage, PRESETS } from './presets';
import type { Detection } from './types';

const personDet: Detection = { cls: 'person', conf: 0.9, box: { x1: 0, y1: 0, x2: 100, y2: 200 } };
const gunDet: Detection = { cls: 'pistol', conf: 0.9, box: { x1: 80, y1: 90, x2: 110, y2: 110 } };

describe('Pipeline', () => {
  it('raises an ARMED event after the balanced preset confirms the weapon', () => {
    const p = new Pipeline(PRESETS.balanced);
    let startedAt = -1;
    for (let i = 0; i < 10; i++) {
      const f = p.process([personDet, gunDet], i * 100);
      if (f.threat.started.length > 0 && startedAt < 0) startedAt = i;
    }
    // Track becomes visible on frame 1 (minHits 2); history needs confirmK true entries.
    expect(startedAt).toBe(PRESETS.balanced.confirmK - 1);
    const last = p.process([personDet, gunDet], 1000);
    expect(last.threat.armedPersonIds).toHaveLength(1);
  });

  it('reset clears tracks and events', () => {
    const p = new Pipeline(PRESETS.balanced);
    for (let i = 0; i < 6; i++) p.process([gunDet], i * 100);
    p.reset();
    expect(p.process([gunDet], 0).tracks).toEqual([]);
  });
});

describe('filterForImage', () => {
  it('applies per-class confidence floors', () => {
    const dets: Detection[] = [
      { ...personDet, conf: 0.4 },
      { ...gunDet, conf: 0.7 },
    ];
    expect(filterForImage(dets, PRESETS.balanced.imageConf).map((d) => d.cls)).toEqual(['pistol']);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/core/threat.test.ts src/core/pipeline.test.ts`
Expected: FAIL — cannot resolve `./threat` / `./pipeline` / `./presets`.

- [ ] **Step 3: Implement `web/src/core/threat.ts`**

```ts
import { center } from './geometry';
import { type ClassKey, isWeapon, type Track } from './types';

export interface ThreatOptions {
  /** A weapon track is confirmed once matched in at least confirmK of its last confirmN frames. */
  confirmK: number;
  confirmN: number;
  /** A person box is grown by this fraction of its width/height on each side when testing who holds a weapon. */
  armedExpand: number;
  /** An event ends after its weapon has been unseen for this long. */
  eventGapMs: number;
}

export const DEFAULT_THREAT_OPTIONS: ThreatOptions = { confirmK: 4, confirmN: 10, armedExpand: 0.2, eventGapMs: 2000 };

export interface ThreatEvent {
  id: number;
  weaponTrackId: number;
  cls: ClassKey;
  personTrackId: number | null;
  startMs: number;
  endMs: number;
  maxConf: number;
  ended: boolean;
}

export interface ThreatFrame {
  confirmedWeaponIds: number[];
  armedPersonIds: number[];
  weaponOwner: ReadonlyMap<number, number>;
  active: ThreatEvent[];
  started: ThreatEvent[];
  ended: ThreatEvent[];
}

export function isConfirmed(history: readonly boolean[], k: number, n: number): boolean {
  let hits = 0;
  for (let i = Math.max(0, history.length - n); i < history.length; i++) if (history[i]) hits++;
  return hits >= k;
}

export function findOwner(weapon: Track, persons: readonly Track[], expand: number): number | null {
  const c = center(weapon.box);
  let best: number | null = null;
  let bestDist = Infinity;
  for (const p of persons) {
    const w = p.box.x2 - p.box.x1;
    const h = p.box.y2 - p.box.y1;
    const inside =
      c.x >= p.box.x1 - expand * w && c.x <= p.box.x2 + expand * w && c.y >= p.box.y1 - expand * h && c.y <= p.box.y2 + expand * h;
    if (!inside) continue;
    const pc = center(p.box);
    const d = Math.hypot(pc.x - c.x, pc.y - c.y);
    if (d < bestDist) {
      bestDist = d;
      best = p.id;
    }
  }
  return best;
}

const copy = (e: ThreatEvent): ThreatEvent => ({ ...e });

export class ThreatEngine {
  private opts: ThreatOptions;
  private confirmed = new Set<number>();
  private open = new Map<number, ThreatEvent>(); // weapon track id -> event
  private all: ThreatEvent[] = [];
  private nextId = 1;

  constructor(opts: Partial<ThreatOptions> = {}) {
    this.opts = { ...DEFAULT_THREAT_OPTIONS, ...opts };
  }

  setOptions(opts: Partial<ThreatOptions>): void {
    this.opts = { ...this.opts, ...opts };
  }

  reset(): void {
    this.confirmed.clear();
    this.open.clear();
    this.all = [];
    this.nextId = 1;
  }

  events(): ThreatEvent[] {
    return this.all.map(copy);
  }

  update(tracks: readonly Track[], tMs: number): ThreatFrame {
    const { confirmK, confirmN, armedExpand, eventGapMs } = this.opts;
    const persons = tracks.filter((t) => t.cls === 'person');
    const weapons = tracks.filter((t) => isWeapon(t.cls));
    const owner = new Map<number, number>();
    const armed = new Set<number>();
    const started: ThreatEvent[] = [];
    const ended: ThreatEvent[] = [];

    for (const w of weapons) {
      if (!this.confirmed.has(w.id) && isConfirmed(w.history, confirmK, confirmN)) this.confirmed.add(w.id);
      if (!this.confirmed.has(w.id)) continue;

      const pid = findOwner(w, persons, armedExpand);
      if (pid !== null) {
        owner.set(w.id, pid);
        armed.add(pid);
      }

      const ev = this.open.get(w.id);
      if (ev === undefined) {
        const created: ThreatEvent = {
          id: this.nextId++,
          weaponTrackId: w.id,
          cls: w.cls,
          personTrackId: pid,
          startMs: tMs,
          endMs: tMs,
          maxConf: w.conf,
          ended: false,
        };
        this.open.set(w.id, created);
        this.all.push(created);
        started.push(copy(created));
      } else {
        ev.endMs = tMs;
        ev.cls = w.cls;
        ev.maxConf = Math.max(ev.maxConf, w.conf);
        if (ev.personTrackId === null) ev.personTrackId = pid;
      }
    }

    const visible = new Set(weapons.map((w) => w.id));
    for (const [wid, ev] of this.open) {
      if (visible.has(wid) || tMs - ev.endMs <= eventGapMs) continue;
      ev.ended = true;
      ended.push(copy(ev));
      this.open.delete(wid);
      this.confirmed.delete(wid);
    }

    return {
      confirmedWeaponIds: weapons.filter((w) => this.confirmed.has(w.id)).map((w) => w.id),
      armedPersonIds: [...armed],
      weaponOwner: owner,
      active: [...this.open.values()].map(copy),
      started,
      ended,
    };
  }
}
```

- [ ] **Step 4: Implement `web/src/core/presets.ts`**

```ts
import type { ClassKey, Detection } from './types';

export type SensitivityName = 'low' | 'balanced' | 'high';

export interface Preset {
  /** Floor passed to the decoder; anything below is discarded before tracking. */
  lowConf: number;
  /** Per-class confidence needed to start a track (tracker's high-confidence pass). */
  highConf: Record<ClassKey, number>;
  /** Per-class confidence for single-image mode, where there is no temporal confirmation. */
  imageConf: Record<ClassKey, number>;
  confirmK: number;
  confirmN: number;
}

// Provisional values for the stock model. Plan 5 retunes them from the trained model's validation PR curves.
export const PRESETS: Record<SensitivityName, Preset> = {
  low: {
    lowConf: 0.15,
    highConf: { person: 0.5, pistol: 0.6, rifle: 0.6, knife: 0.6 },
    imageConf: { person: 0.5, pistol: 0.7, rifle: 0.7, knife: 0.7 },
    confirmK: 6,
    confirmN: 10,
  },
  balanced: {
    lowConf: 0.1,
    highConf: { person: 0.45, pistol: 0.45, rifle: 0.45, knife: 0.45 },
    imageConf: { person: 0.5, pistol: 0.55, rifle: 0.55, knife: 0.55 },
    confirmK: 4,
    confirmN: 10,
  },
  high: {
    lowConf: 0.05,
    highConf: { person: 0.4, pistol: 0.3, rifle: 0.3, knife: 0.3 },
    imageConf: { person: 0.4, pistol: 0.4, rifle: 0.4, knife: 0.4 },
    confirmK: 3,
    confirmN: 10,
  },
};

export const filterForImage = (dets: readonly Detection[], conf: Readonly<Record<ClassKey, number>>): Detection[] =>
  dets.filter((d) => d.conf >= conf[d.cls]);
```

- [ ] **Step 5: Implement `web/src/core/pipeline.ts`**

```ts
import type { Preset } from './presets';
import { type ThreatFrame, ThreatEngine } from './threat';
import { Tracker } from './tracker';
import type { Detection, Track } from './types';

export interface PipelineFrame {
  tracks: Track[];
  threat: ThreatFrame;
}

export class Pipeline {
  private readonly tracker = new Tracker();
  private readonly threat = new ThreatEngine();

  constructor(preset: Preset) {
    this.setPreset(preset);
  }

  setPreset(p: Preset): void {
    this.tracker.setOptions({ highConf: p.highConf, lowConf: p.lowConf });
    this.threat.setOptions({ confirmK: p.confirmK, confirmN: p.confirmN });
  }

  process(dets: readonly Detection[], tMs: number): PipelineFrame {
    const tracks = this.tracker.update(dets);
    return { tracks, threat: this.threat.update(tracks, tMs) };
  }

  reset(): void {
    this.tracker.reset();
    this.threat.reset();
  }
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/core/threat.test.ts src/core/pipeline.test.ts`
Expected: PASS (12 tests).

- [ ] **Step 7: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/core/threat.ts web/src/core/threat.test.ts web/src/core/presets.ts web/src/core/pipeline.ts web/src/core/pipeline.test.ts
git commit -m "feat(core): threat engine (K-of-N, ARMED, events), sensitivity presets, pipeline"
```

---

### Task 9: Detector worker, client with backpressure, and live loop

**Files:**
- Create: `web/src/engine/protocol.ts`, `web/src/engine/detector.worker.ts`, `web/src/engine/detector-client.ts`, `web/src/engine/live-loop.ts`
- Test: `web/src/engine/detector-client.test.ts`

**Interfaces:**
- Consumes: `letterboxGeometry` (Task 4), `rgbaToTensor`, `decodeEndToEnd`, `uniformThresholds` (Task 5), `ModelConfig`
- Produces:
  - `type Backend = 'webgpu' | 'wasm'`; `interface ReadyInfo { backend: Backend; loadMs: number }`; `interface DetectResult { detections: Detection[]; inferMs: number; width: number; height: number }`
  - `type ToWorker = { type: 'init'; model: ModelConfig; preferWebGPU: boolean } | { type: 'detect'; id: number; bitmap: ImageBitmap; lowConf: number }`
  - `type FromWorker = { type: 'ready'; backend: Backend; loadMs: number } | { type: 'progress'; loaded: number; total: number } | ({ type: 'result'; id: number } & DetectResult) | { type: 'error'; id?: number; message: string }`
  - `interface WorkerLike { postMessage(message: ToWorker, transfer: Transferable[]): void; onmessage: ((ev: MessageEvent<FromWorker>) => void) | null; terminate(): void }`
  - `class BusyError extends Error`
  - `class DetectorClient { constructor(worker: WorkerLike); onProgress: ((loaded: number, total: number) => void) | null; get busy(): boolean; init(model: ModelConfig, preferWebGPU?: boolean): Promise<ReadyInfo>; detect(bitmap: ImageBitmap, lowConf: number): Promise<DetectResult>; dispose(): void }`
  - `createDetectorWorker(): Worker`
  - `startLiveLoop(opts: { video: HTMLVideoElement; client: DetectorClient; lowConf: () => number; clock: () => number; onResult: (r: DetectResult, tMs: number) => void; onError: (e: unknown) => void }): () => void` — returns a stop function

- [ ] **Step 1: Write the failing test**

`web/src/engine/detector-client.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { STOCK_COCO_MODEL } from '../config/models';
import { BusyError, DetectorClient, type WorkerLike } from './detector-client';
import type { FromWorker, ToWorker } from './protocol';

class FakeWorker implements WorkerLike {
  sent: ToWorker[] = [];
  onmessage: ((ev: MessageEvent<FromWorker>) => void) | null = null;
  terminated = false;
  postMessage(message: ToWorker): void {
    this.sent.push(message);
  }
  terminate(): void {
    this.terminated = true;
  }
  reply(msg: FromWorker): void {
    this.onmessage?.({ data: msg } as MessageEvent<FromWorker>);
  }
}

const fakeBitmap = () => {
  const b: { closed: boolean; close(): void } = {
    closed: false,
    close() {
      b.closed = true;
    },
  };
  return b as unknown as ImageBitmap & { closed: boolean };
};

describe('DetectorClient', () => {
  it('resolves init when the worker reports ready and forwards progress', async () => {
    const w = new FakeWorker();
    const c = new DetectorClient(w);
    const progress: number[] = [];
    c.onProgress = (loaded) => progress.push(loaded);
    const ready = c.init(STOCK_COCO_MODEL);
    expect(w.sent[0]).toEqual({ type: 'init', model: STOCK_COCO_MODEL, preferWebGPU: true });
    w.reply({ type: 'progress', loaded: 10, total: 100 });
    w.reply({ type: 'ready', backend: 'wasm', loadMs: 42 });
    await expect(ready).resolves.toEqual({ backend: 'wasm', loadMs: 42 });
    expect(progress).toEqual([10]);
  });

  it('rejects init on a worker error without an id', async () => {
    const w = new FakeWorker();
    const c = new DetectorClient(w);
    const ready = c.init(STOCK_COCO_MODEL);
    w.reply({ type: 'error', message: 'HTTP 404' });
    await expect(ready).rejects.toThrow('HTTP 404');
  });

  it('allows one frame in flight and rejects extra frames with BusyError', async () => {
    const w = new FakeWorker();
    const c = new DetectorClient(w);
    const first = c.detect(fakeBitmap(), 0.1);
    expect(c.busy).toBe(true);
    const extra = fakeBitmap();
    await expect(c.detect(extra, 0.1)).rejects.toBeInstanceOf(BusyError);
    expect(extra.closed).toBe(true);
    const sent = w.sent[0] as Extract<ToWorker, { type: 'detect' }>;
    w.reply({ type: 'result', id: sent.id, detections: [], inferMs: 5, width: 640, height: 480 });
    await expect(first).resolves.toEqual({ detections: [], inferMs: 5, width: 640, height: 480 });
    expect(c.busy).toBe(false);
  });

  it('rejects a frame when the worker reports an error for its id', async () => {
    const w = new FakeWorker();
    const c = new DetectorClient(w);
    const p = c.detect(fakeBitmap(), 0.1);
    const sent = w.sent[0] as Extract<ToWorker, { type: 'detect' }>;
    w.reply({ type: 'error', id: sent.id, message: 'boom' });
    await expect(p).rejects.toThrow('boom');
    expect(c.busy).toBe(false);
  });

  it('dispose terminates the worker and rejects pending work', async () => {
    const w = new FakeWorker();
    const c = new DetectorClient(w);
    const p = c.detect(fakeBitmap(), 0.1);
    c.dispose();
    await expect(p).rejects.toThrow(/disposed/);
    expect(w.terminated).toBe(true);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/engine/detector-client.test.ts`
Expected: FAIL — cannot resolve `./detector-client`.

- [ ] **Step 3: Implement `web/src/engine/protocol.ts`**

```ts
import type { ModelConfig } from '../config/models';
import type { Detection } from '../core/types';

export type Backend = 'webgpu' | 'wasm';

export interface ReadyInfo {
  backend: Backend;
  loadMs: number;
}

export interface DetectResult {
  detections: Detection[];
  /** Worker-side time for letterbox + inference + decode. */
  inferMs: number;
  width: number;
  height: number;
}

export type ToWorker =
  | { type: 'init'; model: ModelConfig; preferWebGPU: boolean }
  | { type: 'detect'; id: number; bitmap: ImageBitmap; lowConf: number };

export type FromWorker =
  | { type: 'ready'; backend: Backend; loadMs: number }
  | { type: 'progress'; loaded: number; total: number }
  | ({ type: 'result'; id: number } & DetectResult)
  | { type: 'error'; id?: number; message: string };
```

- [ ] **Step 4: Implement `web/src/engine/detector-client.ts`**

```ts
import type { ModelConfig } from '../config/models';
import type { DetectResult, FromWorker, ReadyInfo, ToWorker } from './protocol';

export interface WorkerLike {
  postMessage(message: ToWorker, transfer: Transferable[]): void;
  onmessage: ((ev: MessageEvent<FromWorker>) => void) | null;
  terminate(): void;
}

export class BusyError extends Error {
  constructor() {
    super('Detector is busy with another frame');
    this.name = 'BusyError';
  }
}

interface Pending<T> {
  resolve: (v: T) => void;
  reject: (e: Error) => void;
}

export function createDetectorWorker(): Worker {
  return new Worker(new URL('./detector.worker.ts', import.meta.url), { type: 'module' });
}

/** Main-thread handle on the detector worker. At most one frame is in flight; extra frames are refused. */
export class DetectorClient {
  onProgress: ((loaded: number, total: number) => void) | null = null;
  private readonly worker: WorkerLike;
  private nextId = 1;
  private inFlight: { id: number; pending: Pending<DetectResult> } | null = null;
  private initPending: Pending<ReadyInfo> | null = null;

  constructor(worker: WorkerLike) {
    this.worker = worker;
    worker.onmessage = (ev) => this.handle(ev.data);
  }

  get busy(): boolean {
    return this.inFlight !== null;
  }

  init(model: ModelConfig, preferWebGPU = true): Promise<ReadyInfo> {
    return new Promise<ReadyInfo>((resolve, reject) => {
      this.initPending = { resolve, reject };
      this.worker.postMessage({ type: 'init', model, preferWebGPU }, []);
    });
  }

  detect(bitmap: ImageBitmap, lowConf: number): Promise<DetectResult> {
    if (this.inFlight) {
      bitmap.close();
      return Promise.reject(new BusyError());
    }
    const id = this.nextId++;
    return new Promise<DetectResult>((resolve, reject) => {
      this.inFlight = { id, pending: { resolve, reject } };
      this.worker.postMessage({ type: 'detect', id, bitmap, lowConf }, [bitmap]);
    });
  }

  dispose(): void {
    this.worker.terminate();
    const err = new Error('Detector disposed');
    this.inFlight?.pending.reject(err);
    this.initPending?.reject(err);
    this.inFlight = null;
    this.initPending = null;
  }

  private handle(msg: FromWorker): void {
    switch (msg.type) {
      case 'progress':
        this.onProgress?.(msg.loaded, msg.total);
        return;
      case 'ready':
        this.initPending?.resolve({ backend: msg.backend, loadMs: msg.loadMs });
        this.initPending = null;
        return;
      case 'result': {
        if (this.inFlight?.id !== msg.id) return;
        const { pending } = this.inFlight;
        this.inFlight = null;
        pending.resolve({ detections: msg.detections, inferMs: msg.inferMs, width: msg.width, height: msg.height });
        return;
      }
      case 'error': {
        const err = new Error(msg.message);
        if (msg.id === undefined) {
          this.initPending?.reject(err);
          this.initPending = null;
        } else if (this.inFlight?.id === msg.id) {
          const { pending } = this.inFlight;
          this.inFlight = null;
          pending.reject(err);
        }
        return;
      }
    }
  }
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/engine/detector-client.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 6: Implement `web/src/engine/detector.worker.ts`**

```ts
import * as ort from 'onnxruntime-web/webgpu';
import type { ModelConfig } from '../config/models';
import { letterboxGeometry } from '../core/geometry';
import { decodeEndToEnd, uniformThresholds } from '../core/postprocess';
import { rgbaToTensor } from '../core/preprocess';
import type { Backend, FromWorker, ToWorker } from './protocol';

const scope = self as unknown as DedicatedWorkerGlobalScope;

// WASM binaries are copied to /ort/ by scripts/prepare-assets.mjs and served same-origin.
ort.env.wasm.wasmPaths = new URL('/ort/', scope.location.origin).href;
ort.env.wasm.numThreads = scope.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let session: ort.InferenceSession | null = null;
let model: ModelConfig | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let tensorData: Float32Array | null = null;

const post = (msg: FromWorker) => scope.postMessage(msg);

async function download(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Model download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length') ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ type: 'progress', loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.length;
  }
  return bytes;
}

async function createSession(bytes: Uint8Array, preferWebGPU: boolean): Promise<{ s: ort.InferenceSession; backend: Backend }> {
  if (preferWebGPU && 'gpu' in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return { s: await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] }), backend: 'webgpu' };
    } catch (err) {
      console.warn('WebGPU session failed, falling back to WASM', err);
    }
  }
  return { s: await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] }), backend: 'wasm' };
}

async function init(m: ModelConfig, preferWebGPU: boolean): Promise<void> {
  const t0 = performance.now();
  const bytes = await download(m.url);
  const { s, backend } = await createSession(bytes, preferWebGPU);
  const size = m.inputSize;
  tensorData = new Float32Array(3 * size * size);
  // Warm-up run: compiles WebGPU shaders / allocates WASM memory before the first real frame.
  await s.run({ [s.inputNames[0]!]: new ort.Tensor('float32', tensorData, [1, 3, size, size]) });
  ctx = new OffscreenCanvas(size, size).getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
  session = s;
  model = m;
  post({ type: 'ready', backend, loadMs: Math.round(performance.now() - t0) });
}

async function detect(id: number, bitmap: ImageBitmap, lowConf: number): Promise<void> {
  if (!session || !model || !ctx || !tensorData) {
    bitmap.close();
    throw new Error('Detector not initialised');
  }
  const t0 = performance.now();
  const size = model.inputSize;
  const { width, height } = bitmap;
  const lb = letterboxGeometry(width, height, size);
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(bitmap, lb.padX, lb.padY, lb.newW, lb.newH);
  bitmap.close();
  rgbaToTensor(ctx.getImageData(0, 0, size, size).data, size, tensorData);

  const out = await session.run({ [session.inputNames[0]!]: new ort.Tensor('float32', tensorData, [1, 3, size, size]) });
  const o = out[session.outputNames[0]!]!;
  const detections = decodeEndToEnd(o.data as Float32Array, o.dims, model.labels, uniformThresholds(lowConf), lb, width, height);
  post({ type: 'result', id, detections, inferMs: performance.now() - t0, width, height });
}

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

scope.onmessage = (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  if (m.type === 'init') init(m.model, m.preferWebGPU).catch((e) => post({ type: 'error', message: message(e) }));
  else detect(m.id, m.bitmap, m.lowConf).catch((e) => post({ type: 'error', id: m.id, message: message(e) }));
};
```

- [ ] **Step 7: Implement `web/src/engine/live-loop.ts`**

```ts
import { BusyError, type DetectorClient } from './detector-client';
import type { DetectResult } from './protocol';

export interface LiveLoopOptions {
  video: HTMLVideoElement;
  client: DetectorClient;
  lowConf: () => number;
  /** Timestamp for a frame: performance.now() for live input, video.currentTime * 1000 for files. */
  clock: () => number;
  onResult: (r: DetectResult, tMs: number) => void;
  onError: (e: unknown) => void;
}

/** Feeds video frames to the detector, sending a new frame only when the previous one has finished. */
export function startLiveLoop(o: LiveLoopOptions): () => void {
  let stopped = false;
  let inFlight = false;
  let handle = 0;
  const usesVfc = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

  const schedule = () => {
    handle = usesVfc ? o.video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
  };

  const tick = () => {
    if (stopped) return;
    if (!inFlight && o.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && o.video.videoWidth > 0) {
      inFlight = true;
      const t = o.clock();
      createImageBitmap(o.video)
        .then((bitmap) => o.client.detect(bitmap, o.lowConf()))
        .then((r) => {
          if (!stopped) o.onResult(r, t);
        })
        .catch((e) => {
          if (!(e instanceof BusyError) && !stopped) o.onError(e);
        })
        .finally(() => {
          inFlight = false;
        });
    }
    schedule();
  };

  schedule();
  return () => {
    stopped = true;
    if (usesVfc) o.video.cancelVideoFrameCallback(handle);
    else cancelAnimationFrame(handle);
  };
}
```

- [ ] **Step 8: Typecheck and lint**

Run: `cd /d/Projects/WeaponShield-AI/web && npm run typecheck && npm run lint`
Expected: both exit 0. If `tsc` complains about `DedicatedWorkerGlobalScope` in the app project, confirm `src/engine/detector.worker.ts` is excluded from `tsconfig.app.json` and included in `tsconfig.worker.json`.

- [ ] **Step 9: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/engine
git commit -m "feat(engine): ONNX Runtime Web detector worker (WebGPU->WASM), client with backpressure, live loop"
```

---

### Task 10: Overlay primitives and painter

**Files:**
- Create: `web/src/render/overlay.ts`
- Test: `web/src/render/overlay.test.ts`

**Interfaces:**
- Consumes: `PipelineFrame` (Task 8), `Detection`, `Track`, `Box`, `Point`, `isWeapon`
- Produces:
  - `OVERLAY_COLORS = { person: '#22c55e', weapon: '#ef4444', armed: '#f59e0b' } as const`
  - `type Primitive = { kind: 'box'; box: Box; color: string; label: string; solid: boolean } | { kind: 'trail'; points: readonly Point[]; color: string }`
  - `buildTrackPrimitives(frame: PipelineFrame): Primitive[]`
  - `buildDetectionPrimitives(dets: readonly Detection[]): Primitive[]`
  - `paint(ctx: CanvasRenderingContext2D, prims: readonly Primitive[], scaleX: number, scaleY: number): void`

- [ ] **Step 1: Write the failing test**

`web/src/render/overlay.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import type { PipelineFrame } from '../core/pipeline';
import type { Track } from '../core/types';
import { buildDetectionPrimitives, buildTrackPrimitives, OVERLAY_COLORS, type Primitive } from './overlay';

type BoxPrim = Extract<Primitive, { kind: 'box' }>;
const isBox = (p: Primitive): p is BoxPrim => p.kind === 'box';

const t = (id: number, cls: Track['cls'], conf = 0.87): Track => ({
  id,
  cls,
  conf,
  box: { x1: 0, y1: 0, x2: 10, y2: 10 },
  hits: 3,
  missed: 0,
  history: [true, true, true],
  trail: [{ x: 1, y: 1 }, { x: 2, y: 2 }],
});

const frame = (tracks: Track[], confirmed: number[], armed: number[]): PipelineFrame => ({
  tracks,
  threat: { confirmedWeaponIds: confirmed, armedPersonIds: armed, weaponOwner: new Map(), active: [], started: [], ended: [] },
});

describe('buildTrackPrimitives', () => {
  it('colours people green, armed people amber, weapons red', () => {
    const prims = buildTrackPrimitives(frame([t(1, 'person'), t(2, 'person'), t(3, 'pistol')], [3], [2]));
    const boxes = prims.filter(isBox);
    expect(boxes.map((b) => [b.color, b.label, b.solid])).toEqual([
      [OVERLAY_COLORS.person, '#1 person', true],
      [OVERLAY_COLORS.armed, '#2 person · ARMED', true],
      [OVERLAY_COLORS.weapon, '#3 pistol 87%', true],
    ]);
  });

  it('draws unconfirmed weapons dashed', () => {
    const [box] = buildTrackPrimitives(frame([t(3, 'knife')], [], [])).filter(isBox);
    expect(box).toMatchObject({ solid: false });
  });

  it('adds a trail for each track with at least two points', () => {
    expect(buildTrackPrimitives(frame([t(1, 'person')], [], [])).filter((p) => p.kind === 'trail')).toHaveLength(1);
  });
});

describe('buildDetectionPrimitives', () => {
  it('labels raw detections with class and confidence', () => {
    const prims = buildDetectionPrimitives([{ cls: 'rifle', conf: 0.5, box: { x1: 0, y1: 0, x2: 5, y2: 5 } }]);
    expect(prims).toEqual([{ kind: 'box', box: { x1: 0, y1: 0, x2: 5, y2: 5 }, color: OVERLAY_COLORS.weapon, label: 'rifle 50%', solid: true }]);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/render/overlay.test.ts`
Expected: FAIL — cannot resolve `./overlay`.

- [ ] **Step 3: Implement `web/src/render/overlay.ts`**

```ts
import type { PipelineFrame } from '../core/pipeline';
import { type Box, type Detection, isWeapon, type Point } from '../core/types';

export const OVERLAY_COLORS = { person: '#22c55e', weapon: '#ef4444', armed: '#f59e0b' } as const;

export type Primitive =
  | { kind: 'box'; box: Box; color: string; label: string; solid: boolean }
  | { kind: 'trail'; points: readonly Point[]; color: string };

const pct = (c: number) => `${Math.round(c * 100)}%`;

export function buildTrackPrimitives(frame: PipelineFrame): Primitive[] {
  const confirmed = new Set(frame.threat.confirmedWeaponIds);
  const armed = new Set(frame.threat.armedPersonIds);
  const prims: Primitive[] = [];
  for (const tr of frame.tracks) {
    let color: string;
    let label: string;
    let solid = true;
    if (isWeapon(tr.cls)) {
      color = OVERLAY_COLORS.weapon;
      label = `#${tr.id} ${tr.cls} ${pct(tr.conf)}`;
      solid = confirmed.has(tr.id);
    } else if (armed.has(tr.id)) {
      color = OVERLAY_COLORS.armed;
      label = `#${tr.id} person · ARMED`;
    } else {
      color = OVERLAY_COLORS.person;
      label = `#${tr.id} person`;
    }
    if (tr.trail.length >= 2) prims.push({ kind: 'trail', points: tr.trail, color });
    prims.push({ kind: 'box', box: tr.box, color, label, solid });
  }
  // Boxes after trails so labels sit on top; keep track order among boxes.
  return [...prims.filter((p) => p.kind === 'trail'), ...prims.filter((p) => p.kind === 'box')];
}

export const buildDetectionPrimitives = (dets: readonly Detection[]): Primitive[] =>
  dets.map((d) => ({
    kind: 'box',
    box: d.box,
    color: isWeapon(d.cls) ? OVERLAY_COLORS.weapon : OVERLAY_COLORS.person,
    label: `${d.cls} ${pct(d.conf)}`,
    solid: true,
  }));

export function paint(ctx: CanvasRenderingContext2D, prims: readonly Primitive[], scaleX: number, scaleY: number): void {
  const lw = Math.max(2, Math.round(2 * Math.max(scaleX, scaleY)));
  ctx.lineJoin = 'round';
  for (const p of prims) {
    ctx.strokeStyle = p.color;
    if (p.kind === 'trail') {
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = lw;
      ctx.setLineDash([]);
      ctx.beginPath();
      p.points.forEach((pt, i) => (i === 0 ? ctx.moveTo(pt.x * scaleX, pt.y * scaleY) : ctx.lineTo(pt.x * scaleX, pt.y * scaleY)));
      ctx.stroke();
      ctx.globalAlpha = 1;
      continue;
    }
    const x = p.box.x1 * scaleX;
    const y = p.box.y1 * scaleY;
    const w = (p.box.x2 - p.box.x1) * scaleX;
    const h = (p.box.y2 - p.box.y1) * scaleY;
    ctx.lineWidth = lw;
    ctx.setLineDash(p.solid ? [] : [6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    const fontPx = Math.max(12, Math.round(13 * Math.max(scaleX, scaleY)));
    ctx.font = `600 ${fontPx}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    const tw = ctx.measureText(p.label).width + 8;
    const th = fontPx + 6;
    const ly = y - th < 0 ? y : y - th;
    ctx.fillStyle = p.color;
    ctx.fillRect(x, ly, tw, th);
    ctx.fillStyle = '#0b0f14';
    ctx.fillText(p.label, x + 4, ly + fontPx);
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/render/overlay.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/render
git commit -m "feat(render): overlay primitives (person/weapon/ARMED colours, trails) and canvas painter"
```

---

### Task 11: Lab page and Playwright end-to-end tests

**Files:**
- Create: `web/src/lab/LabPage.tsx`, `web/playwright.config.ts`, `web/e2e/lab.spec.ts`
- Modify: `web/src/App.tsx`

**Interfaces:**
- Consumes: `DetectorClient`, `createDetectorWorker`, `startLiveLoop` (Task 9); `Pipeline`, `PRESETS`, `filterForImage` (Task 8); `buildTrackPrimitives`, `buildDetectionPrimitives`, `paint` (Task 10); `ACTIVE_MODEL` (Task 5)
- Produces: test ids `status` (text `loading` | `ready` | `error`), `backend` (`webgpu` | `wasm`), `count-person`, `count-weapon`, `frames`, inputs `image-input` and button `webcam-toggle`. Plan 3 replaces this page; the test ids are internal to this plan.

- [ ] **Step 1: Write the e2e test**

`web/playwright.config.ts`:
```ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  timeout: 90_000,
  retries: process.env.CI ? 1 : 0,
  use: {
    ...devices['Desktop Chrome'],
    baseURL: 'http://localhost:4173',
    permissions: ['camera'],
    launchOptions: { args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] },
  },
  webServer: {
    command: 'npm run preview -- --port 4173 --strictPort',
    url: 'http://localhost:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
```

`web/e2e/lab.spec.ts`:
```ts
import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

const bus = fileURLToPath(new URL('../../fixtures/golden/images/bus.jpg', import.meta.url));

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('status')).toHaveText('ready', { timeout: 60_000 });
});

test('page is cross-origin isolated so WASM threads are available', async ({ page }) => {
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
});

test('detects people in an uploaded image', async ({ page }) => {
  await expect(page.getByTestId('backend')).toHaveText(/webgpu|wasm/);
  await page.getByTestId('image-input').setInputFiles(bus);
  await expect(page.getByTestId('count-person')).not.toHaveText('0', { timeout: 30_000 });
  const people = Number(await page.getByTestId('count-person').textContent());
  expect(people).toBeGreaterThanOrEqual(3);
});

test('processes webcam frames continuously', async ({ page }) => {
  await page.getByTestId('webcam-toggle').click();
  await expect.poll(async () => Number(await page.getByTestId('frames').textContent()), { timeout: 30_000 }).toBeGreaterThan(5);
  await page.getByTestId('webcam-toggle').click();
});
```

- [ ] **Step 2: Install the browser and build so the test can run (it fails until the page exists)**

```bash
cd /d/Projects/WeaponShield-AI/web
npx playwright install chromium
npm run build && npm run e2e
```
Expected: FAIL — `getByTestId('status')` not found.

- [ ] **Step 3: Implement `web/src/lab/LabPage.tsx`**

```tsx
import { useEffect, useRef, useState } from 'react';
import { ACTIVE_MODEL } from '../config/models';
import { Pipeline } from '../core/pipeline';
import { filterForImage, PRESETS } from '../core/presets';
import { type Detection, isWeapon } from '../core/types';
import { createDetectorWorker, DetectorClient } from '../engine/detector-client';
import { startLiveLoop } from '../engine/live-loop';
import type { Backend } from '../engine/protocol';
import { buildDetectionPrimitives, buildTrackPrimitives, paint } from '../render/overlay';

type Status = 'loading' | 'ready' | 'error';
const preset = PRESETS.balanced;

function counts(dets: readonly { cls: Detection['cls'] }[]) {
  return { person: dets.filter((d) => d.cls === 'person').length, weapon: dets.filter((d) => isWeapon(d.cls)).length };
}

/** Temporary development harness for the detection core. Replaced by the real console in Plan 3. */
export function LabPage() {
  const [status, setStatus] = useState<Status>('loading');
  const [backend, setBackend] = useState<Backend | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [count, setCount] = useState({ person: 0, weapon: 0 });
  const [inferMs, setInferMs] = useState(0);
  const [frames, setFrames] = useState(0);
  const [live, setLive] = useState(false);
  const clientRef = useRef<DetectorClient | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const client = new DetectorClient(createDetectorWorker());
    clientRef.current = client;
    client.onProgress = (loaded, total) => setProgress(total > 0 ? loaded / total : 0);
    client
      .init(ACTIVE_MODEL)
      .then((info) => {
        setBackend(info.backend);
        setStatus('ready');
      })
      .catch((e: Error) => {
        setError(e.message);
        setStatus('error');
      });
    return () => client.dispose();
  }, []);

  useEffect(() => {
    const client = clientRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!live || !client || !video || !canvas) return;
    let stopLoop: (() => void) | null = null;
    let stream: MediaStream | null = null;
    const pipeline = new Pipeline(preset);
    let cancelled = false;

    navigator.mediaDevices
      .getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      .then(async (s) => {
        if (cancelled) return s.getTracks().forEach((t) => t.stop());
        stream = s;
        video.srcObject = s;
        await video.play();
        stopLoop = startLiveLoop({
          video,
          client,
          lowConf: () => preset.lowConf,
          clock: () => performance.now(),
          onResult: (r, t) => {
            const frame = pipeline.process(r.detections, t);
            canvas.width = r.width;
            canvas.height = r.height;
            const ctx = canvas.getContext('2d')!;
            ctx.drawImage(video, 0, 0, r.width, r.height);
            paint(ctx, buildTrackPrimitives(frame), 1, 1);
            setCount(counts(frame.tracks));
            setInferMs(r.inferMs);
            setFrames((f) => f + 1);
          },
          onError: (e) => setError(e instanceof Error ? e.message : String(e)),
        });
      })
      .catch((e: Error) => setError(e.message));

    return () => {
      cancelled = true;
      stopLoop?.();
      stream?.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    };
  }, [live]);

  async function onImage(file: File) {
    const client = clientRef.current;
    const canvas = canvasRef.current;
    if (!client || !canvas) return;
    setLive(false);
    const shown = await createImageBitmap(file);
    const sent = await createImageBitmap(file);
    canvas.width = shown.width;
    canvas.height = shown.height;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(shown, 0, 0);
    shown.close();
    const r = await client.detect(sent, preset.lowConf);
    const dets = filterForImage(r.detections, preset.imageConf);
    paint(ctx, buildDetectionPrimitives(dets), 1, 1);
    setCount(counts(dets));
    setInferMs(r.inferMs);
  }

  return (
    <main className="mx-auto max-w-5xl space-y-4 p-4 font-mono text-sm">
      <h1 className="text-lg font-semibold">WeaponShield v2 — detection lab</h1>
      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-6">
        <div>status: <span data-testid="status">{status}</span></div>
        <div>backend: <span data-testid="backend">{backend ?? '…'}</span></div>
        <div>download: {Math.round(progress * 100)}%</div>
        <div>person: <span data-testid="count-person">{count.person}</span></div>
        <div>weapon: <span data-testid="count-weapon">{count.weapon}</span></div>
        <div>frames: <span data-testid="frames">{frames}</span> · {inferMs.toFixed(0)} ms</div>
      </dl>
      {error && <p className="text-red-400">{error}</p>}
      <div className="flex flex-wrap gap-3">
        <label className="cursor-pointer rounded border border-slate-600 px-3 py-1">
          Upload image
          <input
            data-testid="image-input"
            type="file"
            accept="image/*"
            className="sr-only"
            disabled={status !== 'ready'}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void onImage(file);
            }}
          />
        </label>
        <button
          data-testid="webcam-toggle"
          type="button"
          className="rounded border border-slate-600 px-3 py-1 disabled:opacity-50"
          disabled={status !== 'ready'}
          onClick={() => setLive((v) => !v)}
        >
          {live ? 'Stop webcam' : 'Start webcam'}
        </button>
      </div>
      <video ref={videoRef} muted playsInline className="hidden" />
      <canvas ref={canvasRef} className="w-full rounded border border-slate-700" />
    </main>
  );
}
```

`web/src/App.tsx`:
```tsx
import { LabPage } from './lab/LabPage';

export default function App() {
  return <LabPage />;
}
```

- [ ] **Step 4: Run lint, typecheck, unit tests and e2e**

```bash
cd /d/Projects/WeaponShield-AI/web
npm run lint && npm run typecheck && npm test && npm run build && npm run e2e
```
Expected: all pass; Playwright reports 3 passed. If the image test reports fewer than 3 people, open `npm run preview`, upload `fixtures/golden/images/bus.jpg` manually and compare the boxes with `fixtures/golden/bus.json`.

- [ ] **Step 5: Check WebGPU manually in a real browser**

```bash
cd /d/Projects/WeaponShield-AI/web
npm run dev
```
Open the printed URL in Chrome on the RTX 3050 laptop. Expected: `backend: webgpu`; webcam mode shows boxes on you labelled `#1 person`; note the ms per frame in the commit message of the next step.

- [ ] **Step 6: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/lab web/src/App.tsx web/playwright.config.ts web/e2e
git commit -m "feat(web): detection lab page (image + webcam) with Playwright e2e"
```

---

### Task 12: Continuous integration and push

**Files:**
- Create: `.github/workflows/web.yml`

**Interfaces:**
- Consumes: npm scripts from Task 2
- Produces: a `web` CI job on pushes to `main`/`v2` and on pull requests

- [ ] **Step 1: Write `.github/workflows/web.yml`**

```yaml
name: web

on:
  push:
    branches: [main, v2]
  pull_request:

jobs:
  web:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: web
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 24
          cache: npm
          cache-dependency-path: web/package-lock.json
      - run: npm ci
      - run: npm run lint
      - run: npm run typecheck
      - run: npm test
      - run: npm run build
      - run: npx playwright install --with-deps chromium
      - run: npm run e2e
      - if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: web/playwright-report
```

- [ ] **Step 2: Commit, check attribution, push**

```bash
cd /d/Projects/WeaponShield-AI
git add .github/workflows/web.yml
git commit -m "ci: lint, typecheck, unit, build and e2e for web"
git log origin/v2..HEAD --format=%B | grep -i -E "co-authored|claude|generated with" || echo "attribution check: clean"
git push origin v2
```
Expected: `attribution check: clean`, then the push succeeds.

- [ ] **Step 3: Watch the CI run**

```bash
gh run watch --repo Halok600/WeaponShield-AI $(gh run list --repo Halok600/WeaponShield-AI --branch v2 --workflow web --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status
```
Expected: the run finishes with success. If it fails, fetch logs with `gh run view <id> --log-failed`, fix, commit and push again.

---

## Notes for later plans

- Vercel still builds `main` from the old `frontend/` root, so production is unaffected by this plan. Preview deployments of `v2` will fail until Plan 5 points the project root at `web/`; that is expected.
- Plan 3 (frontend) replaces `LabPage` and reuses `DetectorClient`, `startLiveLoop`, `Pipeline`, `PRESETS` and `paint` unchanged.
- Plan 4 (edge) ports `tracker.ts`/`threat.ts` to Python and adds `fixtures/parity/` generated from these TypeScript classes.
- Plan 5 swaps `ACTIVE_MODEL` for the trained model and retunes `PRESETS`.
