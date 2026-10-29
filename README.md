# Emotion Is All You Need
### Reconocimiento de emociones en la voz

¿Cómo reconocer una emoción sin confundirla con la identidad de quien habla? Este proyecto compara representaciones acústicas y modelos de aprendizaje profundo para clasificar ocho emociones, con validación por hablante y una evaluación adicional en otro idioma.

**Bautista García y Tomás González** · Proyecto final de Machine Learning, UdeSA · AI Fest 2026.

[Informe](informe.pdf) · [Póster](docs/Garcia_Gonzalez_Poster_PF.pdf) · [Notebook](notebooks/main.ipynb) · [Reproducción](docs/reproduccion.md)

![Comparación de experimentos del proyecto](assets/recorrido-optimizacion.png)

## Del audio a la predicción

Comparamos features eGeMAPS con embeddings de **WavLM, HuBERT y wav2vec 2.0**. El ensemble final combina tres regresiones logísticas y un MLP, con backbones congelados y votación ponderada. También estudiamos CRNN, fine-tuning, aumentación y normalización CMVN por hablante.

La evaluación principal utiliza RAVDESS: 1.440 audios, 24 actores y ocho emociones, con GroupKFold de cinco folds por actor.

| Evaluación | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Ensemble · RAVDESS, validación por actor | 0,890 | 0,888 |
| EMODB · sin CMVN | 0,570 | 0,578 |
| EMODB · con CMVN | 0,767 | 0,787 |

EMODB contiene 454 audios y seis emociones compartidas. Cambian idioma, corpus y clases; sus métricas no son directamente equivalentes a las de RAVDESS. CMVN requiere varias grabaciones por hablante: estos resultados no describen una predicción aislada sobre una voz desconocida.

## Explorar o ejecutar

El notebook conserva las salidas de los experimentos. Para reproducirlos, usar Python 3.11 y seguir la [guía de instalación, datos y entrenamiento](docs/reproduccion.md).

La CLI de predicción es:

~~~sh
python -m src.predict ruta/al/audio.wav
~~~

Requiere generar antes el modelo **data/processed/final_model.pkl**. El repositorio no incluye ese modelo, los corpus ni la interfaz en vivo mencionada en el póster.

**Herramientas:** Python, PyTorch, Transformers, scikit-learn, OpenSMILE, NumPy y pandas.

Los resultados de la tabla proceden de la entrega original. El [detalle técnico](DOCUMENTACION.md) conserva la descripción de experimentos, verificaciones y archivos.
