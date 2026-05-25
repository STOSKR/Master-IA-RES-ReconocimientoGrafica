# Memoria tecnica inicial

## Objetivo

El proyecto implementa un pipeline de vision artificial y OCR para reconstruir series temporales a partir de capturas de graficas historicas Steam/CS2. La salida queda preparada para modelos predictivos o procesos de arbitraje automatico.

## Metodologia

1. Segmentacion espacial de plot area y ejes.
2. Preprocesado de regiones de texto.
3. OCR con PaddleOCR o anclas manuales equivalentes para evaluacion controlada.
4. Extraccion de la senal de precio mediante filtrado HSV, morfologia y componentes conectados.
5. Calibracion lineal pixel-valor usando anclas de fecha y precio.
6. Exportacion CSV/JSON y visualizaciones debug.

## Evaluacion

La evaluacion se separa en tres niveles:

- OCR: CER y WER sobre etiquetas de ejes.
- Geometria: inspeccion/anotacion de anclas y senal extraida.
- Serie final: MAE, RMSE y MSE frente a ground truth con columnas `date` y `price`.

## Limitaciones v1

- El dominio queda acotado a graficas tipo Steam/CS2.
- Se asumen ejes lineales.
- Se requieren al menos dos anclas por eje.
- La extraccion de linea depende de que la senal tenga color distinguible del fondo.

## Mejoras futuras

- Soporte para ejes no lineales o truncados.
- Deteccion de leyendas y elementos superpuestos.
- Segmentacion entrenada para layout.
- Modelos OCR ajustados con capturas reales del dominio.
