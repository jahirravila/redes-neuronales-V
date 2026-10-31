"""Preprocesamiento de imágenes faciales: detección, alineación, recorte, control de calidad y aumentos.

Flujo por imagen:
    leer (EXIF) -> RGB -> detectar rostro (MTCNN) -> alinear por los ojos -> recortar con margen
    -> redimensionar a 160x160 -> [estandarizar al pasar por el encoder]
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 160                      # entrada de FaceNet
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".pgm")


@dataclass
class FaceRecord:
    """Resultado del preprocesamiento de una foto."""
    person: str
    path: str
    face: np.ndarray | None = None            # rostro alineado 160x160x3 uint8
    confidence: float = 0.0
    n_faces: int = 0
    blur: float = 0.0                          # varianza del Laplaciano (nitidez)
    brightness: float = 0.0
    face_px: int = 0                           # ancho de la caja del rostro en la foto original
    status: str = "ok"
    notes: list = field(default_factory=list)


def read_image(path: str, max_side: int = 1024) -> np.ndarray:
    """Lee la imagen respetando la orientación EXIF del móvil y la reduce si es muy grande."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(img)


def align_face(img: np.ndarray, left_eye, right_eye) -> tuple[np.ndarray, np.ndarray]:
    """Rota la imagen para que la línea entre los ojos quede horizontal. Devuelve imagen y matriz."""
    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return rotated, M


def crop_square(img: np.ndarray, box, margin: float = 0.2, size: int = IMG_SIZE) -> np.ndarray:
    """Recorte cuadrado centrado en la caja del rostro con un margen relativo; redimensiona a size x size."""
    x, y, w, h = box
    cx, cy, side = x + w / 2, y + h / 2, max(w, h) * (1 + margin)
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = x1 + int(round(side)), y1 + int(round(side))
    pad = max(0, -x1, -y1, x2 - img.shape[1], y2 - img.shape[0])
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    return cv2.resize(img[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


class FacePreprocessor:
    """Detecta (MTCNN), alinea y recorta rostros; calcula métricas de calidad."""

    def __init__(self, min_confidence: float = 0.90, margin: float = 0.2, min_face_px: int = 60):
        from mtcnn import MTCNN                       # import diferido: es pesado
        self.detector = MTCNN()
        self.min_confidence, self.margin, self.min_face_px = min_confidence, margin, min_face_px

    def detect(self, img):
        dets = [d for d in self.detector.detect_faces(img) if d["confidence"] >= self.min_confidence]
        return sorted(dets, key=lambda d: d["box"][2] * d["box"][3], reverse=True)   # mayor primero

    def process(self, path: str, person: str) -> FaceRecord:
        rec = FaceRecord(person=person, path=path)
        img = read_image(path)
        dets = self.detect(img)
        rec.n_faces = len(dets)
        if not dets:
            rec.status = "sin_rostro"
            return rec
        if len(dets) > 1:
            rec.notes.append(f"{len(dets)} rostros: se usa el más grande")
        d = dets[0]
        rec.confidence, rec.face_px = float(d["confidence"]), int(d["box"][2])
        kp = d["keypoints"]
        # en MTCNN 'left_eye' es el ojo que aparece a la izquierda de la imagen
        rotated, M = align_face(img, kp["left_eye"], kp["right_eye"])
        x, y, w, h = d["box"]
        cx, cy = M @ np.array([x + w / 2, y + h / 2, 1.0])        # centro de la caja tras la rotación
        rec.face = crop_square(rotated, (cx - w / 2, cy - h / 2, w, h), self.margin)
        gray = cv2.cvtColor(rec.face, cv2.COLOR_RGB2GRAY)
        rec.blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        rec.brightness = float(gray.mean())
        if rec.face_px < self.min_face_px:
            rec.notes.append("rostro pequeño")
        return rec


def natkey(s: str):
    """Orden natural: s2 antes que s10."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def list_dataset(root: str) -> dict[str, list[str]]:
    """{persona: [rutas]} a partir de root/<persona>/<imagen>."""
    people = {}
    for person in sorted(os.listdir(root), key=natkey):
        pdir = os.path.join(root, person)
        if os.path.isdir(pdir) and not person.startswith((".", "_")):
            files = sorted((os.path.join(pdir, f) for f in os.listdir(pdir) if f.lower().endswith(VALID_EXT)), key=natkey)
            if files:
                people[person] = files
    return people


def standardize(faces: np.ndarray) -> np.ndarray:
    """Estandarización por imagen (prewhitening de FaceNet): media 0 y desviación 1 en cada rostro."""
    x = faces.astype("float32")
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    std = np.maximum(x.std(axis=(1, 2, 3), keepdims=True), 1.0 / np.sqrt(x[0].size))
    return (x - mean) / std


def augment(face: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aumento de datos fotométrico y geométrico leve (sólo para entrenamiento)."""
    img = face.astype("float32")
    if rng.random() < 0.5:
        img = img[:, ::-1]                                               # espejo horizontal
    angle, scale = rng.uniform(-10, 10), rng.uniform(0.95, 1.08)         # rotación y zoom leves
    tx, ty = rng.uniform(-6, 6, size=2)
    M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, scale); M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)
    img = img * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)           # contraste y brillo
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.clip(img, 0, 255).astype("uint8")


def anonymize(face: np.ndarray, blocks: int = 8) -> np.ndarray:
    """Pixela un rostro para figuras públicas cuando no hay permiso de publicar la imagen."""
    small = cv2.resize(face, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, face.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
