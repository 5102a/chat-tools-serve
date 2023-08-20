import os
import torch

GOOGLE_CSE_ID = "e7f1a28840e8c4715"
GOOGLE_API_KEY = "AIzaSyBA7QfcsE2NnhqmIFUTiuYadZfHKWeF3lk"
RapidAPIKey = "90bbe925ebmsh1c015166fc5e12cp14c503jsn6cca55551ae4"

LLM_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
