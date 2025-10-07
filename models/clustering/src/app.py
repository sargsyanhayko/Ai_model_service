from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.post("/run_main")
def run_main():
    subprocess.Popen([
        "python3", "main.py", 
        "--year", "2024",
        "--adg", "2402.0", "2710.0",
        "--path_to_plots", "plots",
        "--input_path", "input/input.json"
    ])

    return {"status": "main.py is run"}