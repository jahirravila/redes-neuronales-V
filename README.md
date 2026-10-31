# Redes Neuronales V — Reconocimiento facial con redes siamesas y Triplet Loss



Red neuronal siamesa en **TensorFlow + Keras** para verificación e identificación de rostros con **Triplet Loss** (minado de tripletas semi-difíciles), con pruebas **one-shot**, **few-shot** y **zero-shot**.

| # | Notebook | Contenido |
|---|---|---|
| 01 | `01_dataset_preprocesamiento.ipynb` | Dataset, detección MTCNN, alineación por los ojos, recorte 160×160, control de calidad, partición (30 identidades de entrenamiento + 10 no vistas), aumento de datos |
| 02 | `02_red_siamesa_triplet_loss.ipynb` | Red siamesa de 3 ramas con pesos compartidos; Triplet Loss; encoder A (FaceNet pre-entrenado + cabeza siamesa) vs encoder B (CNN desde cero) |
| 03 | `03_pruebas_reconocimiento.ipynb` | Verificación (ROC, AUC, EER, FAR/FRR), identificación 1/3/5-shot, zero-shot (identidades no vistas y rechazo de no registrados), función `reconocer()` |

## Dataset
Base de rostros **ORL / AT&T** (Olivetti Research Laboratory, 1992-1994): 40 personas × 10 fotos, uso académico con crédito al laboratorio (Samaria y Harter, 1994). El notebook 01 la descarga automáticamente. Para usar fotos propias con autorización: `FUENTE = "propio"` y carpeta `dataset_propio/<persona>/<foto>.jpg` (los rostros se pixelan en las figuras y las fotos no se suben: ver `.gitignore`).

## Resultados principales

| Prueba | FaceNet sin ajuste | A. FaceNet + cabeza | B. CNN desde cero |
|---|---|---|---|
| Verificación AUC / EER | 1.0000 / 0.13 % | 0.9997 / 0.42 % | 0.9272 / 14.10 % |
| One-shot (40 personas) | 99.94 % | 98.94 % | 74.76 % |
| Few-shot k = 5 | 100 % | 100 % | 85.75 % |
| Zero-shot: identidades no vistas (k = 1) | 99.96 % | 97.52 % | 58.30 % |
| Zero-shot: rechazo de no registrados | 90.10 % | 99.58 % | 30.85 % |

## Ejecución en Google Colab
1. Abra los notebooks en orden (01 → 02 → 03); la primera celda instala `deepface`, `tf-keras` y `mtcnn`.
2. Ejecute los tres en la misma sesión (comparten `procesado/` y `modelos/`). El 02 tarda ≈ 20 min en CPU (menos con GPU).

## Estructura
```
notebooks/   01, 02, 03 (ejecutados) · figures/ · results/
src/         faces.py (preprocesamiento facial) · siamese.py (red siamesa, Triplet Loss, minado, métricas)
docs/        informe técnico (PDF)
```
