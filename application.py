from http import HTTPStatus
from io import BytesIO
import json

from fastapi import HTTPException, FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F

from src.models.model import ImageClassifier


app = FastAPI()

@app.get("/Health_Check")
def connection():
    """ Checks if there is a connection to app. """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.post("/infer_image")
async def single_inference(bird_image: UploadFile = File(...)):
    """Takes a jpeg image of a bird and runs inference on it."""
    if not bird_image.filename.lower().endswith(('.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG image.")

    image_data = await bird_image.read()
    image = Image.open(BytesIO(image_data))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.473, 0.468, 0.395], std=[0.240, 0.234, 0.255]),
    ])

    image = transform(image).unsqueeze(0)

    with open("src/utils/idx_to_class.json", 'r') as fp:
        idx_to_class = json.load(fp)

    model = ImageClassifier.load_from_checkpoint(
        checkpoint_path="models/model-epoch=12-val_accuracy=0.98.ckpt",
        map_location="cpu",
    )

    model.eval()
    with torch.no_grad():
        logits = model(image)
        probabilities = F.softmax(logits, dim=1)
        top_prob, top_idx = probabilities.topk(1, dim=1)
        class_name = idx_to_class[str(top_idx.item())]

    response = {
        "prediction": f"The bird is predicted to be a {class_name} with {float(top_prob[0]):.5f} certainty",
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK
    }
    return response

# uvicorn --reload --port 8000 application:app