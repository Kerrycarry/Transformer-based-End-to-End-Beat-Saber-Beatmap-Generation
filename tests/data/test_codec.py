# app.py
from audiocraft.solvers import CompressionSolver
import torch
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load the model globally to avoid repeated loading
model = CompressionSolver.model_from_checkpoint('//pretrained/facebook/encodec_32khz', device='cuda')

class AudioInput(BaseModel):
    y_beat_times: list  # Expecting a 1D list of floats representing the tensor

@app.post("/generate_audio_token")
async def generate_audio_token(input_data: AudioInput):
    """
    API endpoint to generate audio tokens for a given y_beat_times.

    Parameters:
        input_data (AudioInput): Input data containing y_beat_times as a list.

    Returns:
        dict: A dictionary containing the audio tokens as a list.
    """
    y_beat_times = torch.tensor(input_data.y_beat_times, device='cuda').unsqueeze(0).unsqueeze(0)  # B, 1, T
    audio_token = model.model.encoder(y_beat_times).permute(0, 2, 1).squeeze(0)  # B, S, D
    return {"audio_token": audio_token.cpu().detach().numpy().tolist()}

# uvicorn tests.data.test_codec:app --host 0.0.0.0 --port 8001