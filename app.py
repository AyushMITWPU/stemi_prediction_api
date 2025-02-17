from fastapi import FastAPI, File, UploadFile, Form
import os
import pandas as pd
import shutil
import random
import subprocess

app = FastAPI()

# Define the base path for storing uploaded files
BASE_PATH = os.path.join(os.getcwd(), "data", "Test_data")
METADATA_FILE = os.path.join(os.getcwd(), "data", "Test_metadata.csv")

# Ensure directories exist
os.makedirs(BASE_PATH, exist_ok=True)

@app.post("/upload/")
async def upload_files(file: UploadFile = File(...), label: str = Form(...), age: int = Form(...), sex: int = Form(...)):
    """Upload ECG files (.dat & .hea), save them, and update metadata."""
    folder_id = 1  # Always use folder ID 1 (overwrite previous data)
    new_dir = os.path.join(BASE_PATH, str(folder_id))

    # Clear previous files if the directory exists
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir, exist_ok=True)

    # Save the uploaded file
    ext = file.filename.split('.')[-1]
    file_path = os.path.join(new_dir, f"{folder_id}.{ext}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Generate a unique patient ID
    patient_id = random.randint(10000, 99999)

    # Create metadata
    new_record = pd.DataFrame({
        "id_rnd": [folder_id],
        "label": [label],
        "patient_id": [patient_id],
        "path": [f"/{folder_id}/{folder_id}"],
        "age": [age],
        "sex": [sex]
    })

    # Save metadata
    new_record.to_csv(METADATA_FILE, index=False)

    return {"message": "File uploaded and metadata saved successfully!"}

@app.post("/predict/")
def predict():
    """Run the prediction script and return the result."""
    try:
        result = subprocess.run(["python", "prediction.py"], capture_output=True, text=True)
        return {"output": result.stdout}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "ECG Prediction API is running!"}



import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
