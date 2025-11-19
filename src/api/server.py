import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from dataclasses import replace
from typing import Dict, List, Optional, Any

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config
from main import (
    load_progress,
    save_progress,
    process_spu,
)
from src.b_image_sampling import BImageSampler
from src.comfyui_runner import ComfyUIRunner
from src.feature_extractor import VITFeatureExtractor


logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Auto Image API", version="0.1.0")


class CreateJobRequest(BaseModel):
    metadata_file: str


class JobInfo(BaseModel):
    job_id: str
    category: str
    spu_ids: List[str]
    metadata_dir: str
    metadata_file: str
    status: str
    message: Optional[str] = None
    processed: List[str] = Field(default_factory=list)
    result_paths: Dict[str, List[str]] = Field(default_factory=dict)
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


# 全局依赖初始化
cfg = Config()
progress_file = Path(cfg.progress_file_path)
progress_lock = threading.Lock()
progress_data = load_progress(progress_file)

logger.info("初始化特征提取器...")
vit_config: Dict[str, Any] = {"model_path": cfg.model_path, "model_name": cfg.model_name}
feature_extractor = VITFeatureExtractor(vit_config)

logger.info("初始化B图采样器...")
b_img_sampler = BImageSampler(
    data_dir=Path(cfg.b_image_dataset_path),
    usage_file=Path(cfg.b_image_usage_file),
    cluster_mapping_file=Path(cfg.cluster_mapping_file),
)

logger.info("初始化ComfyUI运行器...")
comfy_runner = ComfyUIRunner(
    server_address=cfg.comfyui_server_address,
    workflow_path=cfg.comfyui_workflow_path,
    node_mapping=cfg.comfyui_node_mapping,
)

# 任务管理
executor = ThreadPoolExecutor(max_workers=2)
jobs_lock = threading.Lock()
jobs: Dict[str, JobInfo] = {}


def _collect_results(category: str, spu_id: str) -> List[str]:
    """读取 C 图目录，收集指定 SPU 生成的所有文件路径。"""
    result_dir = Path(cfg.c_image_dataset_path) / category / spu_id
    if not result_dir.exists():
        return []
    return [str(p) for p in result_dir.iterdir() if p.is_file()]


def _with_metadata_dir(config: Config, metadata_dir: str) -> Config:
    """基于传入的 metadata_dir 创建一个临时 Config 副本。"""
    return replace(config, metadata_dir=metadata_dir)


def _infer_category_from_metadata_filename(metadata_path: Path) -> str:
    """根据元数据文件名推断品类名称。"""
    stem = metadata_path.stem
    prefix = "images_metadata_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    raise HTTPException(
        status_code=400,
        detail=f"无法从文件名 {metadata_path.name} 推断品类，文件需以 {prefix} 开头。",
    )


def _load_spu_ids_from_metadata_file(metadata_path: Path) -> List[str]:
    """从指定元数据文件中提取所有 SPU ID。"""
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail=f"未找到元数据文件: {metadata_path}")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"元数据文件解析失败: {exc}") from exc

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="元数据文件格式错误，需为列表。")

    spu_ids = [str(item["spu_id"]) for item in data if isinstance(item, dict) and "spu_id" in item]
    if not spu_ids:
        raise HTTPException(status_code=400, detail="元数据文件中未找到任何 spu_id。")
    return spu_ids


def _prepare_job_from_metadata(metadata_file: str) -> tuple[str, Path, Path, List[str]]:
    """解析元数据文件，返回品类、目录、绝对路径及 SPU 列表。"""
    if not metadata_file:
        raise HTTPException(status_code=400, detail="metadata_file 不能为空。")
    metadata_path = Path(metadata_file).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = Path(cfg.metadata_dir) / metadata_path
    metadata_path = metadata_path.resolve()

    category = _infer_category_from_metadata_filename(metadata_path)
    spu_ids = _load_spu_ids_from_metadata_file(metadata_path)
    metadata_dir = metadata_path.parent
    return category, metadata_dir, metadata_path, spu_ids


def _format_seconds(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    if seconds < 0:
        seconds = 0
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


def _get_memory_stats() -> Dict[str, Any]:
    """解析 /proc/meminfo，输出人类可读的内存信息。"""
    meminfo: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as mem_file:
            for line in mem_file:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip().split()[0]
                meminfo[key] = int(value)
    except FileNotFoundError:
        return {"available": False, "message": "/proc/meminfo 不可用"}

    total = meminfo.get("MemTotal", 0) * 1024
    free = meminfo.get("MemFree", 0) * 1024
    buffers = meminfo.get("Buffers", 0) * 1024
    cached = meminfo.get("Cached", 0) * 1024
    available = meminfo.get("MemAvailable", 0) * 1024
    used = total - free - buffers - cached
    to_gb = lambda b: round(b / (1024 ** 3), 2)

    used_gb = to_gb(max(used, 0))
    total_gb = to_gb(total)
    usage_percent = round((used / total) * 100, 2) if total else None

    return {
        "available": True,
        "total_gb": total_gb,
        "used_gb": used_gb,
        "free_gb": to_gb(free),
        "cached_gb": to_gb(cached),
        "buffers_gb": to_gb(buffers),
        "available_gb": to_gb(available),
        "usage_percent": usage_percent,
        "summary": f"{used_gb} GB / {total_gb} GB 已用 ({usage_percent}%)" if usage_percent is not None else "",
    }


def _get_gpu_stats() -> Dict[str, Any]:
    """执行 nvidia-smi 并解析 GPU 使用情况。"""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        gpus = []
        summaries = []
        for idx, line in enumerate(result.stdout.strip().splitlines()):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            gpu_info = {
                "index": idx,
                "name": parts[0],
                "memory_total_mb": float(parts[1]),
                "memory_used_mb": float(parts[2]),
                "memory_free_mb": float(parts[3]),
                "utilization_percent": float(parts[4]),
            }
            if len(parts) > 5:
                gpu_info["temperature_c"] = float(parts[5])
            gpus.append(gpu_info)
            summaries.append(
                f"GPU{idx} {gpu_info['name']}: {gpu_info['memory_used_mb']}/{gpu_info['memory_total_mb']} MB, "
                f"负载 {gpu_info['utilization_percent']}%"
            )
        return {"available": True, "gpus": gpus, "summary": summaries}
    except FileNotFoundError:
        return {"available": False, "message": "未找到 nvidia-smi 命令"}
    except subprocess.CalledProcessError as exc:
        return {"available": False, "message": exc.stderr or str(exc)}


def _run_job(job_id: str):
    with jobs_lock:
        job = jobs[job_id]
        job.status = "running"
        job.started_at = datetime.utcnow()
        jobs[job_id] = job
    try:
        if not comfy_runner.is_server_running():
            raise RuntimeError("ComfyUI 服务器未运行，无法开始任务。")

        for spu_id in job.spu_ids:
            with progress_lock:
                process_spu(
                    spu_id=spu_id,
                    category_name=job.category,
                    config=_with_metadata_dir(cfg, job.metadata_dir),
                    progress=progress_data,
                    extractor=feature_extractor,
                    sampler=b_img_sampler,
                    comfy_runner=comfy_runner,
                )
                save_progress(progress_data, progress_file)

            job.processed.append(spu_id)
            job.result_paths[spu_id] = _collect_results(job.category, spu_id)
            with jobs_lock:
                jobs[job_id] = job

        job.status = "completed"
        job.message = "任务完成"
        job.ended_at = datetime.utcnow()
    except Exception as exc:  # noqa: BLE001
        logger.exception("任务 %s 失败: %s", job_id, exc)
        job.status = "failed"
        job.message = str(exc)
        job.ended_at = datetime.utcnow()
    finally:
        with jobs_lock:
            jobs[job_id] = job
        b_img_sampler.save_usage_counts()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/system/status")
def system_status():
    disk = shutil.disk_usage(cfg.c_image_dataset_path)
    gpu_available = torch.cuda.is_available()
    load_avg: Optional[tuple] = None
    if hasattr(os, "getloadavg"):
        load_avg = os.getloadavg()
    load_dict = None
    if load_avg:
        load_dict = {"1m": load_avg[0], "5m": load_avg[1], "15m": load_avg[2]}

    disk_total = round(disk.total / (1024 ** 3), 2)
    disk_used = round(disk.used / (1024 ** 3), 2)
    disk_free = round(disk.free / (1024 ** 3), 2)
    disk_summary = f"{disk_used} GB / {disk_total} GB 已用 (剩余 {disk_free} GB)"

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu": {
            "count": os.cpu_count(),
            "load_avg": load_dict,
        },
        "memory": _get_memory_stats(),
        "disk": {
            "total_gb": disk_total,
            "used_gb": disk_used,
            "free_gb": disk_free,
            "summary": disk_summary,
        },
        "gpu_runtime_available": gpu_available,
        "gpu_detail": _get_gpu_stats(),
        "jobs": {
            "running": [j_id for j_id, j in jobs.items() if j.status == "running"],
            "queued": [j_id for j_id, j in jobs.items() if j.status == "queued"],
        },
    }


@app.post("/api/jobs", response_model=JobInfo)
def create_job(req: CreateJobRequest):
    category, meta_dir, metadata_path, spu_ids = _prepare_job_from_metadata(req.metadata_file)

    job_id = str(uuid.uuid4())
    job = JobInfo(
        job_id=job_id,
        category=category,
        spu_ids=spu_ids,
        metadata_dir=str(meta_dir),
        metadata_file=str(metadata_path),
        status="queued",
        processed=[],
        result_paths={},
        created_at=datetime.utcnow(),
    )
    with jobs_lock:
        jobs[job_id] = job
    executor.submit(_run_job, job_id)
    with jobs_lock:
        return jobs[job_id]


STATUS_LABELS = {
    "queued": "排队中",
    "running": "运行中",
    "completed": "已完成",
    "failed": "已失败",
}


@app.get("/api/jobs/summary")
def summarize_jobs():
    with jobs_lock:
        total = len(jobs)
        status_counts_cn: Dict[str, int] = {}
        condensed = []
        now = datetime.utcnow()
        for job in jobs.values():
            status_cn = STATUS_LABELS.get(job.status, job.status)
            status_counts_cn[status_cn] = status_counts_cn.get(status_cn, 0) + 1

            elapsed_seconds = None
            eta_seconds = None
            completed_spu = len(job.processed)
            remaining_spu = max(len(job.spu_ids) - completed_spu, 0)
            if job.started_at:
                end_point = job.ended_at or now
                elapsed_seconds = (end_point - job.started_at).total_seconds()
                if completed_spu > 0:
                    avg_per_spu = elapsed_seconds / completed_spu
                    eta_seconds = avg_per_spu * remaining_spu

            condensed.append({
                "任务ID": job.job_id,
                "状态": status_cn,
                "品类": job.category,
                "SPU总数": len(job.spu_ids),
                "已完成SPU": len(job.processed),
                "元数据文件": job.metadata_file,
                "已运行时间": _format_seconds(elapsed_seconds) if elapsed_seconds is not None else None,
                "预计剩余时间": _format_seconds(eta_seconds) if eta_seconds is not None else None,
            })
    return {
        "任务总数": total,
        "状态统计": status_counts_cn,
        "任务列表": condensed,
    }


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="未找到任务。")
    return job


@app.get("/api/jobs", response_model=List[JobInfo])
def list_jobs():
    with jobs_lock:
        return list(jobs.values())


@app.get("/api/results/{category}/{spu_id}/{filename}")
def download_result(category: str, spu_id: str, filename: str):
    file_path = Path(cfg.c_image_dataset_path) / category / spu_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在。")
    return FileResponse(path=file_path)


@app.get("/api/results/{job_id}")
def download_job_results(job_id: str, background_tasks: BackgroundTasks):
    """按 job_id 打包下载所有结果。"""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="未找到任务。")
    if not job.result_paths:
        raise HTTPException(status_code=404, detail="该任务暂无结果文件。")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        archive_path = Path(tmp.name)
    try:
        import zipfile

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for spu_id, paths in job.result_paths.items():
                for p in paths:
                    file_path = Path(p)
                    if file_path.exists():
                        arcname = f"{job.category}/{spu_id}/{file_path.name}"
                        zipf.write(file_path, arcname=arcname)
        background_tasks.add_task(archive_path.unlink, missing_ok=True)
        return FileResponse(
            path=archive_path,
            media_type="application/zip",
            filename=f"{job_id}.zip",
        )
    except Exception as exc:  # noqa: BLE001
        if archive_path.exists():
            archive_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"打包失败: {exc}") from exc
