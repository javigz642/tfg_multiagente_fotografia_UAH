import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import json
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Política de resolución del clasificador: PIL se protege temporalmente y el sistema
# rechaza imágenes por encima de 400 MP.
LIMITE_MEGAPIXELES = 400.0

# Por encima de este umbral se solicita a PIL una decodificación reducida.
MEGAPIXELES_DECODIFICADO_REDUCIDO = 40.0


@contextmanager
def _guarda_pil_desactivada():
    """Desactiva temporalmente `MAX_IMAGE_PIXELS` y restaura su valor anterior."""
    previo = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previo


def _abrir_para_clasificar(ruta: Path) -> Image.Image:
    """Abre la imagen en RGB, la reduce si supera 40 MP y rechaza más de 400 MP."""
    with _guarda_pil_desactivada():
        img = Image.open(ruta)
        try:
            ancho, alto = img.size
            megapixeles = ancho * alto / 1e6

            if megapixeles > LIMITE_MEGAPIXELES:
                raise ValueError(
                    f"La imagen '{ruta.name}' tiene {megapixeles:.1f} MP "
                    f"({ancho}x{alto}) y el límite del sistema son "
                    f"{LIMITE_MEGAPIXELES:.0f} MP. Redúcela antes de analizarla."
                )

            if megapixeles > MEGAPIXELES_DECODIFICADO_REDUCIDO:
                # Se mantiene margen sobre el tamaño final de entrada de la red.
                img.draft("RGB", (1024, 1024))

            return img.convert("RGB")
        except Exception:
            img.close()
            raise

def build_model(num_classes:int = 5, pretrained: bool= True):
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return resnet

class ContextClassifier:
    def __init__(self, weights_path, class_to_idx_path, threshold_path, gradcam_dir, device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(class_to_idx_path, encoding="utf-8") as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        with open(threshold_path, encoding="utf-8") as f:
            self.threshold = json.load(f)["umbral_otro"]

        self.model = build_model(num_classes=len(self.class_to_idx)).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location= self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.gradcam_dir = Path(gradcam_dir)
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)
        self.cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])

    def predict(self, image_path:str) -> dict:
        image_path = Path(image_path)
        img = _abrir_para_clasificar(image_path)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, idx_pred = probs.max(dim=1)
        confidence, idx_pred = confidence.item(), idx_pred.item()

        label = "otro" if confidence < self.threshold else self.idx_to_class[idx_pred]

        gradcam_path = None
        if label != "otro":
            grayscale_cam = self.cam(input_tensor=tensor, targets=[ClassifierOutputTarget(idx_pred)])[0]
            img_np = np.array(img.resize((224, 224))) / 255.0
            overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            gradcam_path = self.gradcam_dir / f"{image_path.stem}_gradcam.png"
            Image.fromarray(overlay).save(gradcam_path)

        return {
            "label": label,
            "confidence": confidence,
            "gradcam_path": str(gradcam_path) if gradcam_path else "NO",
        }
