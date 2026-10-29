# Reconocimiento de emociones en el habla (RAVDESS)

Trabajo práctico final de Machine Learning. El objetivo es **clasificar la emoción**
a partir de una grabación de voz. El dataset de desarrollo es
[RAVDESS](https://zenodo.org/record/1188976) (24 actores, canal *speech*), con 8 clases de
emoción: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`.

El modelo final es un **ensemble** de clasificadores lineales y una red MLP sobre
*embeddings* de tres backbones de audio (WavLM-large, HuBERT-large, wav2vec2-base),
con normalización **CMVN por hablante**.

---

## Estructura de la entrega

```
.
├── README.md             # este archivo
├── informe.pdf           # informe
├── requirements.txt      # dependencias de Python
├── notebooks/
│   └── main.ipynb        # notebook principal (todo el trabajo, de punta a punta)
├── splits/
│   └── split.json        # partición train/val por actor (fija, seed=42)
└── src/                  # código fuente reutilizable que usa el notebook
    ├── config.py         # rutas, seed, mapeos de RAVDESS, columnas de metadata
    ├── data.py           # parseo de nombres RAVDESS, armado de metadata, get_xy
    ├── splits.py         # armado y carga de la partición train/val por actor
    ├── normalize.py      # CMVN por hablante (features pooled y frame-level)
    ├── metrics.py        # métricas de evaluación (accuracy, F1, etc.)
    ├── evaluation.py     # cross-validation agrupada por actor (GroupKFold)
    ├── domain_shift.py   # metadata de EmoDB para el test out-of-domain
    ├── plots.py          # matriz de confusión y gráficos de EDA
    ├── predict.py        # inferencia sobre audios nuevos (clase + CLI)
    ├── features/
    │   ├── egemaps.py    # extracción de features eGeMAPS (openSMILE)
    │   ├── wav2vec.py    # embeddings wav2vec2/WavLM/HuBERT (pooled, por capa y frame-level)
    │   └── augment.py    # data augmentation (speed/ruido) → pseudo-hablantes
    └── models/
        ├── mlp.py        # MLPClassifier (PyTorch, API sklearn) sobre features pooled
        ├── crnn.py       # CRNNClassifier sobre secuencias frame-level
        └── finetune.py   # FineTuneClassifier: fine-tuning end-to-end de wav2vec2
```

### `notebooks/main.ipynb`

Es el hilo conductor del trabajo. Lee funciones de `src/`, genera todos los
resultados y gráficos, y entrena el modelo final. Está organizado en secciones:

1. **Datos y split** — armado de metadata y partición train/val.
2. **Features** — eGeMAPS y wav2vec.
3. **EDA** — balance de clases, distribuciones, PCA 2D, análisis de confounds (género).
4. **Modelos comparación** — baseline lineal vs. otros clasificadores.
5. **Investigación de capas de wav2vec** — qué capa del transformer rinde mejor.
6. **Feature engineering** — importancia de features, selección top-k, decorrelación/PCA.
7. **Modelo "final" (v1)** — CMVN + multi-capa.
8. **Redes neuronales** — MLP sobre features pooled y CRNN sobre secuencias.
9. **Optimización avanzada** — WavLM-large, combinación de representaciones,
   augmentation con pseudo-hablantes, ensemble multi-modelo y fine-tuning end-to-end.
10. **Modelo final** — el ganador, entrenamiento sobre todos los datos, demo de
    predicción sobre un audio y **evaluación en otro dataset (EmoDB)** como prueba
    de generalización a otro dominio.

### `splits/split.json`

Partición **por actor** (no por grabación) para evitar fuga de identidad entre
train y val, estratificada por género y fija con `seed=42`:

- `train_actors`: 20 actores
- `val_actors`: 4 actores (2, 9, 11, 20)

La estimación de performance durante el desarrollo se hace además con
**cross-validation agrupada por actor** (`GroupKFold`, ver `src/evaluation.py`),
que es la métrica principal. **EmoDB** se usa como test independiente (otro idioma
y dominio). El set de evaluación oficial que se provea cumple el rol de test final.

### `src/`

Módulos importables (todos se usan desde `notebooks/main.ipynb`). Ninguna función
lleva prefijo `_` en su API pública. Los puntos de entrada más relevantes:

| Módulo                           | Qué expone                                                                      |
| --------------------------------- | -------------------------------------------------------------------------------- |
| `data.py`                       | `build_metadata()`, `get_xy(df, vocal_channel, split)`, `parse_filename()` |
| `splits.py`                     | `make_split()`, `load_split()`, `assign_split()`                           |
| `features/wav2vec.py`           | `load_audio()`, `extract_wav2vec*()` (pooled / por capa / frames)            |
| `features/egemaps.py`           | `extract_egemaps()`                                                            |
| `features/augment.py`           | `extract_augmented()`, `speed_perturb()`, `add_noise()`                    |
| `normalize.py`                  | `cmvn_by_speaker()`, `cmvn_frames_by_speaker()`                              |
| `evaluation.py`                 | `grouped_cv()`, `grouped_cv_proba()`                                         |
| `models/{mlp,crnn,finetune}.py` | clasificadores con API estilo sklearn (`fit`/`predict`)                      |
| `predict.py`                    | `EmotionPredictor`, `predict()` y CLI de inferencia                          |

---

## Instalación

Requiere **Python 3.11**.

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

La primera vez que se corre inferencia se descargan los backbones de HuggingFace
(WavLM-large, HuBERT-large, wav2vec2-base), varios cientos de MB. Con GPU
(CUDA / Apple MPS) es notablemente más rápido; funciona igual en CPU.

---

## Cómo estimar la emoción de un audio

La inferencia vive en `src/predict.py`. El audio se remuestrea a 16 kHz
automáticamente (`load_audio`), así que puede ser cualquier `.wav`.

### Desde la terminal

```bash
python -m src.predict ruta/al/audio.wav [otro_audio.wav ...]
```

Imprime la emoción predicha por cada archivo:

```
happy       ruta/al/audio.wav
```

### Desde Python

```python
from src.predict import EmotionPredictor

predictor = EmotionPredictor()                 # carga el modelo final
emociones = predictor.predict(["audio.wav"])   # -> ['happy']

# Si además querés las probabilidades por clase:
proba, clases = predictor.predict_proba(["audio.wav"])
```

### Sobre el modelo entrenado y CMVN

- `EmotionPredictor` carga por defecto `data/processed/final_model.pkl`. Ese archivo
  **no se versiona** (es pesado y `data/` está en `.gitignore`): se genera corriendo
  la sección *"Entrenamiento sobre todos los datos"* de `notebooks/main.ipynb`, o se
  entrega aparte. Se puede indicar otra ruta con `EmotionPredictor(model_path=...)`.
- El modelo rinde mejor con **CMVN por hablante**, que resta la voz base de cada
  persona. Para aprovecharlo hay que pasar los IDs de hablante:

  ```python
  predictor.predict(["a1.wav", "a2.wav"], speakers=["spk1", "spk1"])
  ```

  Sin `speakers` (caso de un audio suelto y desconocido) se usan los embeddings sin
  normalizar, que funcionan pero con menor accuracy.

```

```
