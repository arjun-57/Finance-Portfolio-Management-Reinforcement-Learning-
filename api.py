"""
api.py — FastAPI server bridging the Python backend and React frontend.

Endpoints:
  POST /api/train              — start training as a subprocess
  POST /api/train/stop         — stop training
  GET  /api/train/status       — poll training progress
  GET  /api/evaluate           — run evaluation on latest checkpoint
  GET  /api/metrics            — serve results/metrics.json
  GET  /api/portfolio-values   — serve results/portfolio_values.csv
  GET  /api/weights            — serve results/weights_history.csv
  GET  /api/reward-log         — serve results/reward_log.csv
  GET  /api/health             — health check
"""

import os
import sys
import json
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add portfolio_rl to sys.path
BASE_DIR = Path(__file__).parent
RL_DIR   = BASE_DIR / "portfolio_rl"
sys.path.insert(0, str(RL_DIR))

from portfolio_rl.config import (
    METRICS_PATH, PORTFOLIO_VALUES_PATH, WEIGHTS_HISTORY_PATH,
    REWARD_LOG_PATH, RESULTS_DIR, RUNS_DIR,
)

app = FastAPI(title="RL Portfolio API", version="1.0.0")

# Allow Vite dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

STATUS_FILE = str(BASE_DIR / "results" / "train_status.json")
_train_process: Optional[subprocess.Popen] = None
_train_lock = threading.Lock()


def _write_status(payload: dict):
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(payload, f)


def _read_status() -> dict:
    if not os.path.exists(STATUS_FILE):
        return {"status": "idle"}
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except Exception:
        return {"status": "idle"}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/train")
def start_training(timesteps: int = 500_000, device: str = "cpu"):
    global _train_process
    with _train_lock:
        if _train_process and _train_process.poll() is None:
            return JSONResponse({"message": "Training already in progress"}, status_code=409)

        _write_status({"status": "starting", "episode": 0, "timestep": 0, "latest_reward": 0})

        cmd = [
            sys.executable,
            str(RL_DIR / "train.py"),
            "--timesteps", str(timesteps),
            "--device", device,
            "--status-file", STATUS_FILE,
        ]
        _train_process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
        )

    return {"message": "Training started", "pid": _train_process.pid}


@app.post("/api/train/stop")
def stop_training():
    global _train_process
    with _train_lock:
        if _train_process and _train_process.poll() is None:
            _train_process.terminate()
            _write_status({"status": "stopped"})
            return {"message": "Training stopped"}
    return {"message": "No training process running"}


@app.get("/api/train/status")
def get_train_status():
    global _train_process
    status = _read_status()
    is_active_status = status.get("status") in ["training", "starting"]
    
    # Detect if process ended unexpectedly or API server restarted while training
    if is_active_status:
        if _train_process is None or _train_process.poll() is not None:
            status["status"] = "stopped"
            _write_status(status)
            
    return status


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/evaluate")
def run_evaluation(device: str = "cpu"):
    """Runs evaluate-only pass on latest checkpoint."""
    cmd = [
        sys.executable,
        str(RL_DIR / "train.py"),
        "--eval-only",
        "--device", device,
    ]
    result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr[-2000:])
    return {"message": "Evaluation complete", "stdout": result.stdout[-2000:]}


# ─────────────────────────────────────────────────────────────────────────────
# DATA SERVING ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/metrics")
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="metrics.json not found. Run training + evaluation first.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.get("/api/portfolio-values")
def get_portfolio_values():
    if not os.path.exists(PORTFOLIO_VALUES_PATH):
        raise HTTPException(status_code=404, detail="portfolio_values.csv not found. Run evaluation first.")
    df = pd.read_csv(PORTFOLIO_VALUES_PATH, index_col=0)
    # Normalise to 100 at start
    df = df / df.iloc[0] * 100
    records = df.reset_index().rename(columns={"index": "date"}).to_dict(orient="records")
    return records


@app.get("/api/weights")
def get_weights():
    if not os.path.exists(WEIGHTS_HISTORY_PATH):
        raise HTTPException(status_code=404, detail="weights_history.csv not found. Run evaluation first.")
    df = pd.read_csv(WEIGHTS_HISTORY_PATH)
    return df.to_dict(orient="records")


@app.get("/api/reward-log")
def get_reward_log():
    if not os.path.exists(REWARD_LOG_PATH):
        raise HTTPException(status_code=404, detail="reward_log.csv not found. Run training first.")
    df = pd.read_csv(REWARD_LOG_PATH)
    return df.to_dict(orient="records")


@app.get("/api/checkpoints")
def list_checkpoints():
    import glob
    ckpts = sorted(glob.glob(str(Path(RUNS_DIR) / "checkpoint_*.pt")))
    return {"checkpoints": [os.path.basename(c) for c in ckpts]}


@app.get("/api/health")
def health():
    return {"status": "ok", "api_version": "1.0.0"}


# ─────────────────────────────────────────────────────────────────────────────
# SERVE BUILT FRONTEND (production mode)
# ─────────────────────────────────────────────────────────────────────────────

frontend_dist = BASE_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
