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
