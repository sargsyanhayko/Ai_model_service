from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.post("/run_main")
def run_main():
    subprocess.Popen([
        "python3", "main.py", 
        "--tin", "01282006", "00029448", "00024444", 
        "--year", "2023", 
        "--path_to_plots", "plots"
    ])

    return {"status": "main.py is run"}