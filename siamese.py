"""Red siamesa con Triplet Loss en TensorFlow/Keras (API tf_keras, compatible con FaceNet de deepface).

Componentes
-----------
* Encoders f(x) -> embedding de norma 1:
    - build_facenet_head:   cabeza residual entrenable sobre embeddings FaceNet pre-entrenados (transfer learning)
    - build_scratch_encoder: CNN pequeña entrenada desde cero (línea base)
* build_siamese_triplet: modelo siamés de 3 ramas (ancla, positivo, negativo) que COMPARTEN el encoder
* triplet_loss:  L = max(||f(a)-f(p)||² - ||f(a)-f(n)||² + margen, 0)
* mine_triplets: selección de tripletas semi-difíciles (FaceNet, Schroff et al., 2015)
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
import tf_keras as keras

MARGIN = 0.2


# ----------------------------------------------------------------------------- encoders
class L2Normalize(keras.layers.Layer):
    """Proyecta el embedding sobre la hiperesfera unitaria: ||f(x)||₂ = 1."""
    def call(self, x):
        return tf.math.l2_normalize(x, axis=-1)


def load_facenet():
    """FaceNet (Inception-ResNet v1, 128-d) pre-entrenado; pesos públicos distribuidos por deepface."""
    from deepface import DeepFace
    model = DeepFace.build_model("Facenet").model
    model.trainable = False
    return model


def build_facenet_head(in_dim: int = 128, emb_dim: int = 128, hidden: int = 256, dropout: float = 0.2):
    """Cabeza residual: f(x) = normalize(x + g(x)). g se inicializa en cero, así que al empezar
    el encoder reproduce exactamente FaceNet y el entrenamiento sólo aprende la corrección."""
    x_in = keras.Input(shape=(in_dim,), name="facenet_emb")
    x_n = L2Normalize(name="l2_entrada")(x_in)                 # FaceNet entrega vectores sin normalizar
    h = keras.layers.Dense(hidden, activation="relu", name="head_dense")(x_n)
    h = keras.layers.Dropout(dropout)(h)
    delta = keras.layers.Dense(emb_dim, kernel_initializer="zeros", name="head_delta")(h)
    out = L2Normalize(name="l2norm")(keras.layers.Add()([x_n, delta]) if in_dim == emb_dim else delta)
    return keras.Model(x_in, out, name="encoder_facenet_head")


def build_scratch_encoder(input_shape=(96, 96, 3), emb_dim: int = 64):
    """CNN compacta entrenada desde cero (sin conocimiento previo de rostros)."""
    x_in = keras.Input(shape=input_shape, name="rostro")
    x = x_in
    for filters in (32, 64, 128, 128):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(emb_dim)(x)
    return keras.Model(x_in, L2Normalize(name="l2norm")(x), name="encoder_cnn_desde_cero")


# ----------------------------------------------------------------------------- red siamesa
class TripletDistances(keras.layers.Layer):
    """Calcula d(a,p) y d(a,n) como distancias euclidianas al cuadrado."""
    def call(self, inputs):
        a, p, n = inputs
        return tf.stack([tf.reduce_sum(tf.square(a - p), -1), tf.reduce_sum(tf.square(a - n), -1)], axis=-1)


def triplet_loss(margin: float = MARGIN):
    """Triplet Loss: empuja d(a,p) por debajo de d(a,n) con al menos 'margin' de separación."""
    def loss(_, d):                                   # y_true no se usa: la supervisión está en la tripleta
        return tf.reduce_mean(tf.maximum(d[:, 0] - d[:, 1] + margin, 0.0))
    loss.__name__ = "triplet_loss"
    return loss


def triplet_accuracy(_, d):
    """Fracción de tripletas bien ordenadas: d(a,p) < d(a,n)."""
    return tf.reduce_mean(tf.cast(d[:, 0] < d[:, 1], tf.float32))


def build_siamese_triplet(encoder, margin: float = MARGIN, lr: float = 1e-3):
    """Arquitectura siamesa de tres ramas con pesos compartidos (un único 'encoder')."""
    shape = encoder.input_shape[1:]
    a, p, n = (keras.Input(shape=shape, name=nm) for nm in ("ancla", "positivo", "negativo"))
    d = TripletDistances(name="distancias")([encoder(a), encoder(p), encoder(n)])
    model = keras.Model([a, p, n], d, name="red_siamesa_triplet")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=triplet_loss(margin), metrics=[triplet_accuracy])
    return model


# ----------------------------------------------------------------------------- minado de tripletas
def pairwise_sq_dist(e: np.ndarray) -> np.ndarray:
    sq = (e ** 2).sum(1)
    return np.maximum(sq[:, None] + sq[None, :] - 2 * e @ e.T, 0.0)


def mine_triplets(emb: np.ndarray, labels: np.ndarray, margin: float = MARGIN, rng=None, max_per_anchor: int = 4):
    """Para cada par (ancla, positivo) elige un negativo semi-difícil:
    d(a,p) < d(a,n) < d(a,p) + margen. Si no existe, toma el negativo más cercano que no sea
    más cercano que el positivo; en último caso uno aleatorio."""
    rng = rng or np.random.default_rng(0)
    D = pairwise_sq_dist(emb)
    triplets, kinds = [], {"semi-difícil": 0, "difícil": 0, "aleatorio": 0}
    for a in range(len(labels)):
        pos = np.where((labels == labels[a]) & (np.arange(len(labels)) != a))[0]
        neg = np.where(labels != labels[a])[0]
        if not len(pos) or not len(neg):
            continue
        for p in rng.permutation(pos)[:max_per_anchor]:
            dap, dan = D[a, p], D[a, neg]
            semi = neg[(dan > dap) & (dan < dap + margin)]
            if len(semi):
                n = rng.choice(semi); kinds["semi-difícil"] += 1
            elif (dan > dap).any():
                cand = neg[dan > dap]; n = cand[np.argmin(D[a, cand])]; kinds["difícil"] += 1
            else:
                n = rng.choice(neg); kinds["aleatorio"] += 1
            triplets.append((a, p, n))
    return np.array(triplets), kinds


# ----------------------------------------------------------------------------- utilidades de evaluación
def embed(encoder, x, batch: int = 64) -> np.ndarray:
    return encoder.predict(x, batch_size=batch, verbose=0)


def pair_scores(emb: np.ndarray, labels: np.ndarray):
    """Distancias de todos los pares (i<j) y si son de la misma persona (genuino) o no (impostor)."""
    D = pairwise_sq_dist(emb)
    iu = np.triu_indices(len(labels), k=1)
    return D[iu], (labels[iu[0]] == labels[iu[1]]).astype(int)


def verification_metrics(dist: np.ndarray, same: np.ndarray, threshold: float | None = None):
    """ROC-AUC, EER y exactitud de verificación con un umbral (si no se da, el que iguala FAR y FRR)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    score = -dist                                     # menor distancia = más parecido
    auc = roc_auc_score(same, score)
    fpr, tpr, thr = roc_curve(same, score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    eer, eer_thr = (fpr[i] + fnr[i]) / 2, -thr[i]
    t = eer_thr if threshold is None else threshold
    pred = (dist <= t).astype(int)
    far = ((pred == 1) & (same == 0)).sum() / max((same == 0).sum(), 1)
    frr = ((pred == 0) & (same == 1)).sum() / max((same == 1).sum(), 1)
    return {"AUC": auc, "EER": eer, "umbral_EER": eer_thr, "umbral": t, "exactitud": (pred == same).mean(),
            "FAR": far, "FRR": frr, "roc": (fpr, tpr)}
