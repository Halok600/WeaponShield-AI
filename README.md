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
