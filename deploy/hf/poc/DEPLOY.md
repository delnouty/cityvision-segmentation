# Deploying the PoC dashboard on Hugging Face Spaces

A **third** Space, separate from the API and the demo app (which stay as they
are). It runs `poc_app` with the three models in-process, on CPU. Same pattern
as `deploy/hf/api` (Docker Space, weights through Git LFS); background in
`deploy/hf/DEPLOYMENT.md`.

## 0. Before staging (once, or after retraining)

```powershell
python scripts/build_eda_stats.py        # poc_app/data: dataset stats + 30 samples
python scripts/evaluate_poc.py           # results/poc_comparison.json
python scripts/build_poc_dashboard.py    # poc_app/data/poc_results.json (+ dashboard/poc.html)
```

## 1. Create the Space

On huggingface.co: **New Space** → name `cityvision-poc` → SDK **Docker** →
hardware **CPU basic (free)**. Free Spaces sleep when idle and wake in about
1-2 minutes; switch to paid always-on hardware later in **Settings → Hardware**
if needed (no code change).

## 2. Clone it and stage the files

```powershell
cd C:\Users\DFILATOVA\Desktop
git clone https://huggingface.co/spaces/DaryaEL/cityvision-poc hf-poc
cd hf-poc
git lfs install
cd ..\CityVision
.\deploy\hf\poc\stage.ps1 -Target C:\Users\DFILATOVA\Desktop\hf-poc
```

`stage.ps1` keeps the Space's own `.gitattributes` (created by Hugging Face)
and only adds the missing LFS rules, so `*.pth` and `*.png` go through Git LFS.

## 2b. Optional: test the image locally first (Docker Desktop running)

`--cpus 2` reproduces the free Space (2 vCPU). The app reads the container's
CPU limit and sets PyTorch's thread count to match.

```powershell
cd C:\Users\DFILATOVA\Desktop\hf-poc
docker build -t cityvision-poc .
docker run -d --name cityvision-poc-test --cpus 2 --memory 16g -p 7860:7860 cityvision-poc
# open http://localhost:7860 ; stop with:  docker rm -f cityvision-poc-test
```

Measured this way (laptop, 2 CPUs): about 1.6 s per image for ResNet50-UNet,
3.2 s for Mask2Former, 4.0 s for EoMT; RAM about 1.4 GB with all three loaded.

## 3. Commit and push

```powershell
cd C:\Users\DFILATOVA\Desktop\hf-poc
git add .gitattributes
git add .
git lfs ls-files          # must list the 3 .pth files and the sample .png files
git commit -m "CityVision PoC dashboard"
git push
```

The Space builds the Docker image (several minutes: CPU PyTorch + transformers),
then starts. URL: `https://huggingface.co/spaces/DaryaEL/cityvision-poc`.

## 4. Update later

Re-run step 0 if results changed, then `stage.ps1` again, commit and push.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Push rejected, "binary files" or "file too large" | Files not in LFS: `.gitattributes` must be committed before the files; check `git lfs ls-files`. |
| "No checkpoint found" on the Prediction page | A `.pth` was pushed as an LFS pointer only, or not staged. Re-run `stage.ps1`, check `git lfs ls-files`. |
| First prediction slow | Each model is loaded on first use (a few seconds), then cached. CPU inference takes a few seconds per image. |
| App takes 1-2 min to open | Free hardware was asleep; it wakes on the first visit. |
