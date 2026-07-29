# Deploying CityVision on Hugging Face Spaces — full guide

This is an end-to-end guide to deploy CityVision as **two Hugging Face Spaces**:

- **API Space** — the FastAPI prediction service (carries the model weights, uses PyTorch).
- **App Space** — the Streamlit demo UI (lean, no PyTorch; calls the API over HTTP).

It covers the *why* (separation, Docker) as well as every concrete step (creating
Spaces, git-lfs for the weights, what to push where, the port gotcha, wiring the
two together, updating, troubleshooting).

> Repo references in this guide:
> `deploy/hf/api/Dockerfile`, `deploy/hf/api/README.md`,
> `deploy/hf/app/Dockerfile`, `deploy/hf/app/README.md`,
> `docker/*.Dockerfile`, `docker-compose.yml`, `.dockerignore`.

---

## 0. Mental model — how HF Spaces work

A Hugging Face **Space is just a git repository** hosted on `huggingface.co`, with
some extra rules:

1. **A `README.md` with a YAML front-matter header configures the Space.** The key
   fields for us:
   ```yaml
   ---
   title: CityVision Segmentation API
   emoji: 🛣️
   colorFrom: blue
   colorTo: green
   sdk: docker        # <- build a Dockerfile (not Gradio/Streamlit auto-SDK)
   app_port: 7860     # <- the port HF routes public traffic to
   pinned: false
   ---
   ```
2. **`sdk: docker` means HF builds the `Dockerfile` at the repo root** and runs the
   resulting container. There is no magic — whatever your Dockerfile's `CMD` starts
   is your app.
3. **HF exposes exactly one port to the world: `app_port` (7860 by convention).**
   Your server *must* listen on that port. This is why the HF Dockerfiles serve on
   `7860`, while local `docker-compose` uses `8000`/`8501` (see §6, the port gotcha).
4. **Public URLs:**
   - Space page (build logs, settings): `https://huggingface.co/spaces/<user>/<space>`
   - Direct app URL (what clients hit): `https://<user>-<space>.hf.space`
     (username + space name, lowercased, non-alphanumerics → `-`).
   - Example here: user `DaryaEL`, space `cityvision-api`
     → `https://daryael-cityvision-api.hf.space`.
5. **Files > 10 MB must be tracked with git-lfs** (our ResNet50 weight is ~157 MB).

You will end up with **two independent Space repos**, each containing a
root-level `Dockerfile` + `README.md` and the subset of the project it needs.

---

## 1. Why separate frontend and backend?

| Reason | Detail |
|---|---|
| **Different dependency weight** | The backend needs PyTorch + torchvision (~hundreds of MB) and the model weights. The frontend needs only Streamlit + requests + Pillow + numpy — **no torch**. Splitting keeps the UI image small and fast to build/boot. |
| **Independent scaling & cost** | Inference is the expensive part. You can put the **API on paid always-on hardware** (to avoid cold starts) while the **UI stays on the free tier**. One monolith couldn't do that. |
| **Clean contract** | The frontend only ever makes HTTP calls (`/predict/image`, `/predict/classes`, `/health`). Any client — curl, a browser, another app — can use the API identically. |
| **No training/serving skew** | Both the API and training import the **same `cityvision/` package** (class taxonomy, palette, model definitions), so preprocessing is identical everywhere. |

The frontend reaches the backend purely through the `CITYVISION_API` environment
variable — locally it's `http://backend:8000` (compose network), in the cloud it's
the API Space's public URL.

---

## 2. Why Docker (and why the Docker SDK on HF)?

- **Reproducibility** — the image pins exact versions (`torch==2.12.0`,
  `fastapi==0.136.3`, …). "Works on my machine" becomes "works everywhere".
- **Same artifact locally and in the cloud** — you validate with `docker compose up`
  on your laptop, then deploy the *same* Dockerfile logic to HF. No surprises.
- **Full control over the environment** — critically, we install the **CPU-only
  PyTorch wheels** (`--index-url https://download.pytorch.org/whl/cpu`), which are
  far smaller than the default CUDA build. HF free/basic hardware is CPU anyway.
- **Portability / no lock-in** — the exact same Docker images run on AWS/GCP/Azure/
  Render. HF is a convenient host, not a dependency.
- **HF `sdk: docker`** gives us this control instead of HF's opinionated
  auto-SDKs — we decide the base image, the deps, and the start command.

---

## 3. Prerequisites (one-time)

1. **A Hugging Face account** → https://huggingface.co/join
2. **git** and **git-lfs** installed locally:
   ```bash
   git lfs version    # should print a version; if not, install git-lfs first
   git lfs install     # one-time, enables the lfs filters for your user
   ```
3. **A trained weight present** at `backend/model/resnet50_best.pth` (the served
   model; `CITYVISION_ARCH=ResNet50-UNet`).
4. **An HF access token** with *write* permission for pushing over HTTPS:
   HF → Settings → Access Tokens → *New token* (role: **Write**). You'll use it as
   the git password when prompted (username = your HF username).
5. *(Optional but recommended)* **Docker Desktop** to test locally first.

---

## 4. Step 0 — Sanity-check locally with docker compose

Before touching the cloud, confirm the two-service setup works end to end:

```bash
docker compose up --build
# open http://localhost:8501  (UI)   and   http://localhost:8000/docs  (API)
```

What this proves:
- The backend image builds, loads `backend/model/resnet50_best.pth`, and serves.
- The frontend image builds (no torch) and talks to the backend via
  `CITYVISION_API=http://backend:8000` (set in `docker-compose.yml`).

If this works, cloud deployment is almost entirely "same Dockerfile, port 7860,
public URL instead of the compose hostname".

Stop with `docker compose down`.

---

## 5. Step 1 — Deploy the **API** Space

### 5a. Create the Space (web UI)
1. https://huggingface.co/new-space
2. **Owner**: you. **Space name**: `cityvision-api`.
3. **SDK**: choose **Docker** → **Blank / from scratch** (not a template).
4. **Hardware**: start with **CPU basic (free)**. Upgrade to a **paid always-on**
   CPU later if you want to avoid cold-start sleeps (see §9).
5. **Visibility**: Public (so the demo is reachable).
6. Create — HF makes an (almost) empty git repo.

### 5b. Populate the Space repo
The API Space needs, **at its repo root**:
```
Dockerfile          <- from deploy/hf/api/Dockerfile
README.md           <- from deploy/hf/api/README.md (has the HF front-matter)
cityvision/         <- shared package (constants + model defs)
backend/            <- FastAPI app + backend/model/resnet50_best.pth (via LFS!)
```

> **Why these files:** the API Dockerfile does `COPY cityvision/ …` and
> `COPY backend/ …`, then `ENV CITYVISION_ARCH=ResNet50-UNet`, and serves
> `uvicorn backend.app:app` on port 7860. It does **not** need `frontend/`,
> `data/`, `notebooks/`, `mlflow.db`, etc.

Clone the Space and copy the files in:

```bash
# 1. Clone the empty Space repo (HTTPS; use your HF token as the password)
git clone https://huggingface.co/spaces/<user>/cityvision-api
cd cityvision-api

# 2. Enable LFS and TRACK THE WEIGHTS *before* adding them
git lfs install
git lfs track "*.pth"          # creates/updates .gitattributes

# 3. Copy the needed files from your project (adjust the source path)
#    - the API Dockerfile/README go to the ROOT and are renamed:
cp   /path/to/CityVision/deploy/hf/api/Dockerfile ./Dockerfile
cp   /path/to/CityVision/deploy/hf/api/README.md  ./README.md
cp -r /path/to/CityVision/cityvision ./cityvision
cp -r /path/to/CityVision/backend    ./backend
# ensure the weight came along:
ls -lh backend/model/resnet50_best.pth   # ~157 MB

# 4. Commit and push
git add .gitattributes Dockerfile README.md cityvision backend
git commit -m "Deploy CityVision prediction API (Docker Space)"
git push
```

> **Critical ordering:** run `git lfs track "*.pth"` **before** `git add`-ing the
> weight. If you add a >10 MB file without LFS, the push is rejected and you have to
> rewrite history. Verify with `git lfs ls-files` — the `.pth` must be listed.

### 5c. Watch the build & test
- On the Space page, open the **Logs** tab — you'll see `pip install torch…`, the
  `COPY`s, then `uvicorn … running on 0.0.0.0:7860`.
- Test:
  ```bash
  curl https://<user>-cityvision-api.hf.space/health
  # {"status":"ok","arch":"ResNet50-UNet", ...}

  # open the Swagger UI in a browser:
  #   https://<user>-cityvision-api.hf.space/docs

  # a real prediction (PNG overlay):
  curl -X POST "https://<user>-cityvision-api.hf.space/predict/image?format=overlay" \
       -F "file=@some_street.png" -o overlay.png
  ```

**Note this URL** — the App Space needs it.

---

## 6. The port gotcha (read this)

| Environment | Backend port | Frontend port | Set where |
|---|---|---|---|
| Local `docker-compose` | 8000 | 8501 | `docker/*.Dockerfile`, `docker-compose.yml` |
| **Hugging Face Space** | **7860** | **7860** | `deploy/hf/*/Dockerfile` + `app_port: 7860` in each `README.md` |

HF only routes public traffic to `app_port` (7860). Each HF container serves a
**single** service on 7860 — that's the whole reason we run two Spaces rather than
one. The `EXPOSE 7860` and the `--port 7860` in the HF Dockerfiles must match the
`app_port` in the front-matter.

---

## 7. Step 2 — Deploy the **App** Space

### 7a. Create it
Same as §5a, but name it `cityvision` (the UI). SDK = **Docker**, **CPU basic (free)**
is fine — the UI is lightweight.

### 7b. Populate & point it at the API
The App Space needs at its root:
```
Dockerfile          <- from deploy/hf/app/Dockerfile
README.md           <- from deploy/hf/app/README.md
cityvision/         <- torch-free constants (palette/classes)
frontend/           <- Streamlit UI
data/samples/       <- the committed sample images (so the demo works w/o dataset)
```

```bash
git clone https://huggingface.co/spaces/<user>/cityvision
cd cityvision
git lfs install                 # samples are small PNGs, but harmless to enable

cp   /path/to/CityVision/deploy/hf/app/Dockerfile ./Dockerfile
cp   /path/to/CityVision/deploy/hf/app/README.md  ./README.md
cp -r /path/to/CityVision/cityvision   ./cityvision
cp -r /path/to/CityVision/frontend     ./frontend
mkdir -p data && cp -r /path/to/CityVision/data/samples ./data/samples

git add .
git commit -m "Deploy CityVision demo UI (Docker Space)"
git push
```

### 7c. Set the API URL
The app finds the backend through the `CITYVISION_API` env var. Two ways:

- **Simplest:** edit the App Space's `Dockerfile` line
  `ENV CITYVISION_API=https://daryael-cityvision-api.hf.space` to **your** API URL,
  then commit/push. (This is already the default in `deploy/hf/app/Dockerfile`.)
- **Cleaner (no rebuild of the value into the image):** Space → **Settings** →
  **Variables and secrets** → add a **Variable** `CITYVISION_API` =
  `https://<user>-cityvision-api.hf.space`. Space env vars override the Dockerfile
  `ENV`.

Open `https://huggingface.co/spaces/<user>/cityvision` → pick an image → it calls
your API and shows **real image · real mask · predicted mask**.

> The frontend already tolerates a sleeping API: `wait_for_api()` polls `/health`
> and retries through the ~30–60 s cold-start window (503s) instead of failing.

---

## 8. git-lfs — the model weight, in detail

- HF rejects files > 10 MB unless tracked by LFS. `resnet50_best.pth` is ~157 MB.
- The `.gitattributes` line `*.pth filter=lfs diff=lfs merge=lfs -text` (created by
  `git lfs track "*.pth"`) is what routes the weight through LFS. **Commit
  `.gitattributes` first / in the same commit as the weight.**
- Verify before pushing: `git lfs ls-files` must show the `.pth`.
- If you accidentally committed the weight without LFS: the fastest fix is usually
  to delete the local clone, re-clone, `git lfs track` first, then re-copy and
  re-commit (avoids history surgery).
- The full Cityscapes dataset is **not** deployed (it's gitignored and excluded via
  `.dockerignore`); only `data/samples/` ships in the App image.

---

## 9. Hardware, cold starts, and "always-on"

- **Free CPU basic** Spaces **sleep after inactivity** and take ~30–60 s to wake
  ("cold start"), returning 503 meanwhile.
- For a smooth public demo we run the **API Space on paid always-on** CPU hardware so
  the model is always resident. The **App Space** can stay free (its cold start is
  cheap, and it politely waits for the API via `wait_for_api`).
- Change this in Space → **Settings** → **Hardware**. (Keep the two Spaces separate —
  this is precisely the flexibility the split buys you.)

---

## 10. Configuration reference (env vars)

| Variable | Used by | Meaning | Default |
|---|---|---|---|
| `CITYVISION_API` | App | URL of the API Space | `http://backend:8000` (local) / your API URL (cloud) |
| `CITYVISION_ARCH` | API | Which architecture to serve | `ResNet50-UNet` (set in the API Dockerfile) |
| `CITYVISION_CHECKPOINT` | API | Explicit weight path (optional) | derived from arch |
| `CITYVISION_MAX_UPLOAD_MB` | API | Upload size cap | `10` |

Set these under Space → **Settings → Variables and secrets**. Use a **secret**
(not a variable) only for truly sensitive values — none here are secret.

---

## 11. Updating a deployment

HF redeploys on every push:

```bash
cd cityvision-api        # or cityvision
# ...make changes / drop in a newly trained weight...
git add -A && git commit -m "Update model / code" && git push
# HF rebuilds the image and restarts the container automatically.
```

To ship a **new model weight**: replace `backend/model/resnet50_best.pth` in the API
Space repo, commit (LFS handles it), push. No code change needed.

---

## 12. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Build fails on `git push` with "file too large" | Weight not tracked by LFS. `git lfs track "*.pth"` **before** adding; check `git lfs ls-files`. |
| Space builds but shows "no application on port" | Server not on **7860**. Ensure `--port 7860` in `CMD` and `app_port: 7860` in `README.md`. |
| API `/health` OK but App shows connection error | `CITYVISION_API` wrong/missing. Set it to the API's `https://<user>-<space>.hf.space` (no trailing slash, `https`). |
| App works, predictions 503 for a while then recover | API cold start. Expected on free tier; `wait_for_api` retries. Use always-on to avoid. |
| `ModuleNotFoundError: cityvision` | `cityvision/` wasn't copied into the Space repo root. |
| Model file missing at runtime | `backend/model/*.pth` didn't ship (LFS pointer only, or not copied). Confirm real bytes in the Space repo. |
| Wrong model served | `CITYVISION_ARCH` mismatch with the weight present. |

---

## 13. Deployment checklist

**API Space**
- [ ] Space created, SDK = Docker, port 7860
- [ ] `Dockerfile` (from `deploy/hf/api/`) + `README.md` at repo root
- [ ] `cityvision/` and `backend/` copied in
- [ ] `git lfs track "*.pth"` done **before** adding the weight; `.gitattributes` committed
- [ ] `backend/model/resnet50_best.pth` present as real bytes (`git lfs ls-files`)
- [ ] Build green; `/health` and `/docs` reachable
- [ ] (optional) upgraded to always-on hardware

**App Space**
- [ ] Space created, SDK = Docker, port 7860
- [ ] `Dockerfile` (from `deploy/hf/app/`) + `README.md` at repo root
- [ ] `cityvision/`, `frontend/`, `data/samples/` copied in
- [ ] `CITYVISION_API` set to the API Space URL (Dockerfile ENV or Space variable)
- [ ] UI loads and shows image · real mask · predicted mask

---

### TL;DR
Two Docker Spaces. Each is a git repo with a root `Dockerfile` + `README.md`
(front-matter, `app_port: 7860`). API Space carries `cityvision/` + `backend/` +
the LFS-tracked weight and serves FastAPI on 7860. App Space carries `cityvision/`
+ `frontend/` + `data/samples/`, serves Streamlit on 7860, and points at the API via
`CITYVISION_API`. Test locally with `docker compose up`, then `git push` each Space.
