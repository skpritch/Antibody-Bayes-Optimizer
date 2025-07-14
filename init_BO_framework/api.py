# api.py

import os
import uuid
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import warnings

warnings.filterwarnings("ignore")

# Redis & RQ for background jobs
from redis import Redis
from rq import Queue

from scripts.utils import load_data, preprocess
from scripts.helpers import FE, FD, FM, find_global_max
from scripts.bayes_optimizer import BayesianOptimizer

# Initialize Redis connection and RQ queue
redis_conn = Redis(host="localhost", port=6379, db=0)
task_queue = Queue("bo_tasks", connection=redis_conn)

app = FastAPI(
    title="Antibody Bayesian Optimization API",
    description="Wraps the Bayesian optimization pipeline as HTTP endpoints",
    version="0.2"
)

app.mount("/results", StaticFiles(directory="results"), name="results")

class OptimizeRequest(BaseModel):
    func: str
    data_filename: str = "FE_d2_N50.json"
    feature_columns: list[str]
    target_column: str = "y"
    surrogate_kind: str = "gp"
    acquisition: str = "ei"
    opt_method: str = "grad_opt"
    n_iter: int = 30
    seed: int = 42

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/optimize")
async def optimize(request: OptimizeRequest):
    job_id = uuid.uuid4().hex
    task_name = f"{request.func}_d{len(request.feature_columns)}_{job_id}"
    job_dir = os.path.join("results", task_name)
    os.makedirs(job_dir, exist_ok=True)
    job = task_queue.enqueue(
        _run_and_save,
        request.dict(),
        job_id,
        job_dir,
        job_id=job_id
    )
    return {"job_id": job.id}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = task_queue.fetch_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    status = job.get_status()
    response = {"status": status}
    if status == "finished":
        for d in os.listdir("results"):
            if job_id in d:
                response["files"] = os.listdir(os.path.join("results", d))
                break
    elif status == "failed":
        response["error"] = job.exc_info
    return response

def _run_and_save(request_dict, job_id: str, job_dir: str):
    try:
        # 1) Load & preprocess
        data_path = os.path.join("datasets/synth_data", request_dict["data_filename"])
        x_raw, y, df = load_data(data_path, request_dict["feature_columns"], request_dict["target_column"])
        x_init, _ = preprocess(x_raw)

        bounds = [[0, 10]] * len(request_dict["feature_columns"])
        func_map = {"FE": FE, "FD": FD, "FM": FM}
        func_oracle = func_map.get(request_dict["func"])
        if not func_oracle:
            raise ValueError(f"Unknown func '{request_dict['func']}'")

        model_map = {
            "Gaussian Process": "gp",
            "Bayesian Linear Regressor": "blr",
            "K-Nearest Neighbors": "knn",
            "Random Forest": "rf"
        }
        model_oracle = model_map.get(request_dict["surrogate_kind"])
        if not model_oracle:
            raise ValueError(f"Unknown surrogate_kind '{request_dict['surrogate_kind']}'")

        acq_map = {"Expected Improvement": "ei", "Uniform Confidence Band": "ucb", "Thompson Sampling": "ts"}
        acq_oracle = acq_map.get(request_dict["acquisition"])
        if not acq_oracle:
            raise ValueError(f"Unknown acquisition '{request_dict['acquisition']}'")

        opt_map = {"Gradient Optimization": "grad_opt", "Random Sampling": "rand_opt"}
        opt_oracle = opt_map.get(request_dict["opt_method"])
        if not opt_oracle:
            raise ValueError(f"Unknown opt_method '{request_dict['opt_method']}'")

        bo = BayesianOptimizer(
            func=func_oracle,
            feature_columns=request_dict["feature_columns"],
            target_column=request_dict["target_column"],
            surrogate_kind=model_oracle,
            acq=acq_oracle,
            opt_method=opt_oracle,
            seed=request_dict["seed"]
        )

        x_all, y_all, pred_y, y_train_max = bo.optimize(x_init, y, bounds, request_dict["n_iter"])
        _, global_max = find_global_max(func_oracle, np.array(bounds))

        # 2) Save predictions as JSON
        json_file = os.path.join(job_dir, "bo_preds.json")
        predictions = []
        suggestions = x_all[-request_dict["n_iter"]:]
        for i, x_vec in enumerate(suggestions, start=1):
            entry = {"iteration": i}
            for idx, col in enumerate(request_dict["feature_columns"]):
                entry[col] = float(x_vec[idx])
            entry["pred_y"] = float(pred_y[i - 1])
            entry["train_max"] = float(y_train_max)
            entry["global_max"] = float(global_max)
            predictions.append(entry)
        with open(json_file, "w") as f:
            json.dump(predictions, f)

        return "SUCCESS"
    except Exception:
        raise
