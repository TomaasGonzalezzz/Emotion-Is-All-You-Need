# Reproducir el proyecto

Esta guía documenta los archivos recibidos. El código original se conserva sin cambios.
La ruta principal es `notebooks/main.ipynb`; la inferencia está en `src/predict.py`.

## 1. Entorno

Usar Python 3.11, como indica el README original. El notebook registra 3.11.15.
Desde la raíz del repositorio:

```sh
python -m venv .venv
```

Activar el entorno:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source .venv/bin/activate
```

Instalar y abrir Jupyter:

```sh
python -m pip install -r requirements.txt
python -m notebook notebooks/main.ipynb
```

Las versiones de `requirements.txt` son cotas mínimas de la entrega. No se verificó
una instalación completa de todas las dependencias para esta publicación. El código
usa PyTorch, torchaudio, Transformers, openSMILE y scikit-learn, entre otras bibliotecas.

## 2. Datos y rutas

El ZIP no contiene audios. El README original referencia
[RAVDESS en Zenodo](https://zenodo.org/record/1188976). Preparar los corpus en las
rutas definidas por `src/config.py`:

```text
data/
├── Audio_Speech_Actors_01-24/
│   ├── Actor_01/
│   │   └── 03-01-01-01-01-01-01.wav
│   └── ...
├── Audio_Song_Actors_01-24/         # si se incluye el canal song
├── emodb_raw/
│   └── extracted/
│       └── wav/                   # audios EMODB para la evaluación externa
└── processed/                     # features, cachés y modelo generados
```

`build_metadata()` recorre los canales speech y song si están presentes; las
comparaciones principales se filtran a speech. El notebook guardado muestra 2.452
filas de metadata de ambos canales; eso no cambia que el análisis principal del
informe use 1.440 audios de habla.

Mantener los nombres originales de los audios porque codifican emoción, actor y
otras variables. EMODB requiere nombres con el patrón que interpreta
`src/domain_shift.py`; se excluye boredom y se conservan las seis clases compartidas.
Los corpus no se redistribuyen en este repositorio.

## 3. Ejecutar el desarrollo

Ejecutar el notebook desde el inicio. La primera celda ajusta el directorio si se
abre desde `notebooks/`. Las etapas posteriores dependen de variables calculadas antes.

1. Cargar metadata y `splits/split.json`.
2. Extraer eGeMAPS y embeddings; construir las cachés de `data/processed/`.
3. Reproducir EDA, modelos base y análisis de capas/features.
4. Reproducir CMVN, comparaciones con MLP/CRNN y optimización avanzada.
5. Construir el ensemble y ejecutar «Entrenamiento sobre todos los datos».
6. Ejecutar inferencia y, con EMODB preparado, el análisis de cambio de dominio.

Los backbones identificados en el código son `facebook/wav2vec2-base`,
`microsoft/wavlm-large` y `facebook/hubert-large-ll60k`. Transformers los descarga
al necesitarlos. La ejecución completa requiere almacenamiento, memoria y tiempo
para features y modelos; el fine-tuning es una etapa especialmente costosa.

Las celdas pueden reutilizar archivos de caché cuando existen. Si se cambian corpus,
capas o parámetros, revisar qué resultados se están leyendo antes de compararlos.
`data/` y `reports/` quedan excluidas de Git.

## 4. Modelo entrenado e inferencia

La sección «Entrenamiento sobre todos los datos» genera:

```text
data/processed/final_model.pkl
```

Incluye tres clasificadores lineales, un MLP, capas elegidas y pesos de votación.
Ese archivo no estaba incluido en el ZIP. Las salidas guardadas que dicen que ya
existía corresponden al equipo utilizado en la ejecución original.

Con el modelo disponible, ejecutar desde la raíz:

```sh
python -m src.predict ruta/al/audio.wav
```

También se puede indicar el archivo de modelo desde Python:

```python
from src.predict import EmotionPredictor

predictor = EmotionPredictor(model_path="data/processed/final_model.pkl")
emociones = predictor.predict(["audio.wav"])
```

La CLI devuelve una emoción por archivo. `load_audio()` adapta el audio a mono y
16 kHz. Para la celda de demo, aportar un audio propio y cambiar `data/feliz.wav`,
que no estaba incluido.

### CMVN por hablante

La implementación calcula media y desvío sobre los audios recibidos de cada hablante.
No utiliza una grabación neutral de calibración independiente. Si se desea aplicar
esa normalización, pasar un conjunto con varias grabaciones por hablante:

```python
emociones = predictor.predict(
    ["a1.wav", "a2.wav", "b1.wav", "b2.wav"],
    speakers=["a", "a", "b", "b"],
)
```

Con una sola muestra y un ID de hablante, el centrado genera un vector nulo;
para el caso de un audio aislado, la API original permite omitir `speakers`.
Los resultados del benchmark con CMVN utilizan información no etiquetada del
conjunto de grabaciones de cada hablante y no deben interpretarse como una medida
idéntica al escenario de una sola grabación desconocida.

### Puntajes del ensemble

`predict_proba()` devuelve la suma ponderada de probabilidades y las clases.
Para mostrar una distribución cuya suma sea uno, normalizarla como hace la demo:

```python
puntajes, clases = predictor.predict_proba(["audio.wav"])
probabilidades = puntajes / puntajes.sum(axis=1, keepdims=True)
```

## 5. Alcance de lo publicado

| Incluido | A obtener o generar para ejecutar |
| --- | --- |
| Código, notebook con salidas, informe y póster | Audios RAVDESS y EMODB |
| Partición fija por actor | Backbones descargados por Transformers |
| Dependencias originales | Cachés de features y `final_model.pkl` |
| CLI e interfaz Python | Audio local de demo e interfaz en vivo Gobbie |

No se ejecutaron entrenamientos ni inferencia real durante esta publicación.
Las comprobaciones locales están documentadas en `verificacion_basica.json`.
