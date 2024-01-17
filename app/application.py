from http import HTTPStatus
from io import BytesIO
import json

from fastapi import HTTPException, FastAPI, UploadFile, File # type: ignore
from PIL import Image # type: ignore
from torchvision import transforms # type: ignore
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import timm # type: ignore


class ModelSkeleton(pl.LightningModule):
    def __init__(self, model_name="caformer_s36.sail_in22k_ft_in1k", num_classes=525):
        super(ModelSkeleton, self).__init__()
        self.model = timm.create_model(model_name, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)


app = FastAPI()

@app.post("/infer_image")
async def single_inference(bird_image: UploadFile = File(...)):
    """Takes a jpeg image of a bird and runs inference on it."""
    if not bird_image.filename.lower().endswith(('.jpg', '.jpeg')): # type: ignore
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG image.")

    image_data = await bird_image.read()
    image = Image.open(BytesIO(image_data))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.473, 0.468, 0.395], 
                             std=[0.240, 0.234, 0.255])
        ])

    image = transform(image).unsqueeze(0)

    with open("app/idx_to_class.json", 'r') as fp:
        idx_to_class = json.load(fp)

    model = ModelSkeleton.load_from_checkpoint(
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

# uvicorn --reload --port 8000 app.application:app