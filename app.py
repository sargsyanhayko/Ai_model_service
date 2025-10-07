import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import subprocess

app = FastAPI()

class CommandRequest(BaseModel):
    models_name: str
    target: List[List[Any]]

@app.post("/run_command")
def run_command(request: CommandRequest):
    try:
        folder_path = os.path.abspath(os.path.join("models", request.models_name))
        if not os.path.exists(folder_path):
            return {"error": f"❌ Model '{request.models_name}' does not exist at {folder_path}"}

        possible_venvs = ["venv", "myenv"]
        venv_path = None
        for v in possible_venvs:
            candidate = os.path.join(folder_path, v, "bin", "python3")
            if os.path.exists(candidate):
                venv_path = candidate
                break
        if not venv_path:
            return {"error": f"❌ No venv found in '{folder_path}'. Expected in 'venv/bin/python3' or 'myenv/bin/python3'"}

        src_path = os.path.join(folder_path, "src")
        current_cwd = src_path if os.path.exists(src_path) else folder_path

        if not request.target or not isinstance(request.target, list):
            return {"error": "❌ 'target' must be a non-empty list of argument lists"}

        command = [venv_path, "main.py"]
        for group in request.target:
            if isinstance(group, list):
                if group and not str(group[0]).startswith("--"):
                    group[0] = "--" + str(group[0])
                command += [str(x) for x in group]
            else:
                return {"error": f"❌ Each element in 'target' must be a list, got: {group}"}

        result = subprocess.run(
            command,
            cwd=current_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            return {
                "model": request.models_name,
                "final_command": " ".join(command),
                "error": f"❌ Script exited with code {result.returncode}",
                "stderr": result.stderr.strip()
            }

        return {
            "model": request.models_name,
            "final_command": " ".join(command),
            "message": "✅ Script finished successfully!"
        }

    except Exception as e:
        return {"error": f"❌ Unexpected error: {str(e)}"}
