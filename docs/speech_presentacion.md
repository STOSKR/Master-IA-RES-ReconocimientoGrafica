# Speech de la presentación

## 1. Portada

Buenas tardes, yo soy Shiyi y os voy a presentar mi proyecto. Consiste en un flujo de trabajo para el reconocimiento de gráficas y digitalizar sus resultados.

## 2. Motivación

La motivación viene del mercado de artículos digitales de CS2. Hay veces que los precios no se pueden scrapear y es por eso que se ha hecho este trabajo.

## 3. Parte del TFM

El trabajo formaría parte de mi TFM, exactamente el paso anterior a almacenar los datos y pasarselo al modelo de predicción de precios.

## 4. Objetivo

El objetivo principal es transformar una captura de pantalla en una serie temporal estructurada.

## 5. Técnicas implementadas

El proyecto combina tres partes. Con visión artificial uso OpenCV para localizar regiones y extraer la línea. Con OCR uso PaddleOCR para leer fechas y precios. Y se ha usado NumPy e interpolación lineal para pasar de coordenadas de píxel a valores.

## 6. Pipeline

El proceso completo tiene cinco fases. Primero se localiza el layout, después se leen los ejes con OCR, luego se extrae la señal visual, se calibra la relación píxel-dato y finalmente se exporta a CSV y JSON.

## 7. Layout + OCR

Aquí tenemos una representación visual de cómo es el layout y como se ha extraido los precios y las fechas.

## 8. Señal visual

En esta parte se extrae la línea de precio. El sistema filtra por color, genera una máscara fina, calcula el centro de la señal en cada columna y reduce el trazo a una línea de un píxel.

## 9. Calibración

Una vez tenemos píxeles y anclas, hacemos la transformación geométrica. La posición X se convierte en fecha y la posición Y se convierte en precio mediante interpolación lineal.

## 10. Extracción: línea trazada

Aquí se ve la señal detectada sobre la gráfica original. Esta vista sirve para comprobar visualmente si el algoritmo está siguiendo la línea correcta y si respeta los cambios bruscos de precio.

## 11. Extracción: máscara binaria

Aquí se muestra la misma señal aislada en blanco y negro. El blanco representa los píxeles que el sistema considera parte de la línea, ya sin el ruido visual de la interfaz.

## 12. Problemas corregidos

Durante el desarrollo aparecieron problemas prácticos: el eje X quedaba demasiado bajo, la máscara inicial era demasiado gruesa, el OCR detectaba algunas anclas falsas y algunas fechas salían incompletas. Estas correcciones mejoraron la estabilidad del pipeline.

## 13. Cierre

En resumen, el proyecto convierte una gráfica rasterizada en datos estructurados. La visión artificial permite entender la imagen, el OCR permite leer las referencias de escala y la geometría permite reconstruir la serie temporal.
