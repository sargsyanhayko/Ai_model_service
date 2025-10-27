# app.py
import os, re, shlex, subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

app = FastAPI()

MODELS_ROOT = Path(__file__).parent / "models"

# Конфиг трёх моделей: как запускать, где cwd, какие ключи разрешены
ALLOWED_MODELS: Dict[str, Dict[str, Any]] = {
    # 1) supervised -> python -m complex_audits_frauds ...
    "supervised_model": {
        "runner": "module",                    # module|script
        "entry": "complex_audits_frauds",      # имя модуля для -m
        "workdirs": ["."],                     # где выполнять -m
        "prefix_args": [],                     # позиционные до флагов
        "venv_candidates": ["venv", "myenv"],
        "prefer_py": ["python3", "python"],    # порядок при падении
        "allowed_args": {
            "tin": {"multi": True},
            "year": {},
            "path-to-plots": {},
            "path_to_plots": {},
            "usergroup-id": {},
            "usergroup_id": {},
            "added-by-user": {},
            "added_by_user": {},
            "dashboard_title": {},
            "file": {},
        },
    },
    # 2) unsupervised -> cd src && python main.py input ...
    "unsupervised_model": {
        "runner": "script",
        "entry": "main.py",
        "workdirs": ["src"],
        "prefix_args": ["input"],
        "venv_candidates": ["venv", "myenv"],
        "prefer_py": ["python3", "python"],
        "allowed_args": {
            "file": {},
            "dashboard_title": {},
            "usergroup_id": {},
            "added_by_user": {},
        },
    },
    # 3) time_series -> cd time_series_analysis && python3.12 main.py input ...
  "model_unsupervised_time_series": {
    "runner": "script",
    "entry": "main.py",  # оставляем по умолчанию main.py
    "workdirs": ["time_series_analysis", "time_series_analysis/src", "."],
    "prefix_args": ["input"],
    "venv_candidates": ["venv", "myenv"],
    "prefer_py": ["python3.12", "python3", "python"],
    "allowed_args": {
        "file": {},
        "dashboard_title": {},
        "usergroup_id": {},
        "added_by_user": {},
    },
},

}

ENV_ALLOWLIST = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "DATABASE_URL"}

class CommandRequest(BaseModel):
    models_name: str
    target: List[List[Union[str, int, float, bool]]]

    @field_validator("models_name")
    @classmethod
    def _model_name_ok(cls, v: str) -> str:
        if v not in ALLOWED_MODELS:
            raise ValueError(f"Model '{v}' is not allowed. Allowed: {list(ALLOWED_MODELS)}")
        if not re.fullmatch(r"[a-zA-Z0-9_\-]+", v):
            raise ValueError("Invalid models_name.")
        return v

    @field_validator("target")
    @classmethod
    def _target_ok(cls, t):
        if not t or not isinstance(t, list):
            raise ValueError("'target' must be a non-empty list of lists")
        for g in t:
            if not isinstance(g, list) or not g:
                raise ValueError("Each group must be a non-empty list")
        return t

def _norm_key_to_cli(k: str) -> str:
    s = str(k)
    if s.startswith("--"):
        return s
    # если пользователь прислал snake_case — оставляем snake_case
    if "_" in s and "-" not in s:
        return f"--{s}"
    # если прислал kebab-case — оставляем kebab-case
    if "-" in s and "_" not in s:
        return f"--{s}"
    # если непонятно, по умолчанию в kebab
    return f"--{s.replace('_', '-')}"

def _find_python(model_dir: Path, venvs: List[str], prefer: List[str]) -> List[str]:
    # 1) venv/bin/pythonX
    for v in venvs:
        for exe in prefer:
            p_unix = model_dir / v / "bin" / exe
            p_win = model_dir / v / "Scripts" / (exe + ".exe")
            if p_unix.exists():
                return [str(p_unix)]
            if p_win.exists():
                return [str(p_win)]
    # 2) poetry (если есть pyproject)
    if (model_dir / "pyproject.toml").exists():
        return ["poetry", "run", "python"]
    # 3) системный
    return [prefer[0]]

def _build_command(req: CommandRequest) -> Dict[str, Any]:
    cfg = ALLOWED_MODELS[req.models_name]
    model_dir = (MODELS_ROOT / req.models_name).resolve()
    if not str(model_dir).startswith(str(MODELS_ROOT.resolve())):
        raise ValueError("Path traversal detected")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    # выбрать рабочую директорию: первый из workdirs, где найден entry (для script)
    workdir: Optional[Path] = None
    if cfg["runner"] == "script":
        for wd in cfg["workdirs"]:
            cand = model_dir / wd
            if (cand / cfg["entry"]).exists():
                workdir = cand
                break
        if workdir is None:
            raise FileNotFoundError(f"Entry '{cfg['entry']}' not found in: {cfg['workdirs']}")
    else:
        # module: достаточно существования папки; cwd — первый workdir (обычно ".")
        for wd in cfg["workdirs"]:
            cand = model_dir / wd
            if cand.exists():
                workdir = cand
                break
        if workdir is None:
            raise FileNotFoundError(f"Workdir not found among: {cfg['workdirs']}")

    py = _find_python(model_dir, cfg["venv_candidates"], cfg["prefer_py"])

    # белый список аргументов
    allowed = set(cfg["allowed_args"].keys())

    # начало команды
    if cfg["runner"] == "module":
        cmd = [*py, "-m", cfg["entry"]]
    else:
        cmd = [*py, cfg["entry"]]

    # позиционные до флагов (например, "input")
    cmd += cfg.get("prefix_args", [])

    # собрать флаги/значения
    for group in req.target:
        raw_key = str(group[0])
        variants = {raw_key, raw_key.replace("-", "_"), raw_key.replace("_", "-")}
        if not (variants & allowed):
            raise ValueError(f"Argument '{raw_key}' is not allowed for model '{req.models_name}'")
        # выбрать spec (если обе формы — берём первую попавшуюся)
        spec = None
        for v in variants:
            if v in cfg["allowed_args"]:
                spec = cfg["allowed_args"][v]
                break
        spec = spec or {}
        key_cli = _norm_key_to_cli(raw_key)

        if spec.get("flag"):
            if len(group) != 1:
                raise ValueError(f"Flag '{raw_key}' must not have a value")
            cmd.append(key_cli)
        else:
            if len(group) == 1:
                raise ValueError(f"Argument '{raw_key}' requires a value")
            cmd.append(key_cli)
            # поддержка множественных значений (напр., --tin a b c)
            for v in group[1:]:
                cmd.append(str(v))

    env = {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST}
    return {"cmd": cmd, "cwd": workdir, "env": env}

@app.post("/run_command")
def run_command(request: CommandRequest):
    try:
        built = _build_command(request)
        res = subprocess.run(
            built["cmd"],
            cwd=built["cwd"],
            env=built["env"] or None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60*30,           # 30 минут
            check=False,
        )
        def tail(s: str, lim: int = 20000) -> str:
            return (s or "")[-lim:]

        out = {
            "model": request.models_name,
            "workdir": str(built["cwd"]),
            "final_command": " ".join(shlex.quote(x) for x in built["cmd"]),
            "return_code": res.returncode,
            "stdout": tail(res.stdout),
            "stderr": tail(res.stderr),
        }
        if res.returncode != 0:
            out["error"] = f"Script failed with code {res.returncode}"
        else:
            out["message"] = "✅ Script finished successfully"
        return out
    except subprocess.TimeoutExpired:
        return {"error": "⏳ Timeout: script exceeded time limit"}
    except Exception as e:
        return {"error": f"❌ {type(e).__name__}: {e}"}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8888, reload=True)