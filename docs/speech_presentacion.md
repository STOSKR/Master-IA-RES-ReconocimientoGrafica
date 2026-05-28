# Speech de la presentación

## 1. Portada

Buenos días. Mi proyecto se llama Reconocimiento de gráficas y consiste en convertir capturas de pantalla de gráficas financieras en series temporales utilizables. La idea es que una imagen rasterizada deje de ser solo una captura y pase a ser una fuente de datos.

## 2. Objetivo

El objetivo principal es transformar una captura de pantalla en una serie temporal estructurada. Para eso el sistema tiene que detectar dónde está la gráfica, leer las escalas de los ejes y reconstruir los valores reales de precio y fecha.

## 3. Técnicas implementadas

El proyecto combina tres partes. Con visión artificial uso OpenCV para localizar regiones y extraer la línea. Con OCR uso PaddleOCR para leer fechas y precios. Y con geometría uso interpolación lineal para pasar de coordenadas de píxel a valores reales.

## 4. Pipeline

El proceso completo tiene cinco fases. Primero se localiza el layout, después se leen los ejes con OCR, luego se extrae la señal visual, se calibra la relación píxel-dato y finalmente se exporta a CSV y JSON. Cada fase genera imágenes de depuración para poder revisar qué ha pasado.

## 5. Layout + OCR

Aquí se ve cómo se delimitan las zonas importantes: el área de la gráfica, el eje X y el eje Y. Sobre esas zonas se aplica OCR para obtener anclas, por ejemplo fechas en el eje horizontal y precios en el vertical. Esas anclas son la referencia para convertir la imagen en datos.

## 6. Señal visual

En esta parte se extrae la línea de precio. El sistema filtra por color, genera una máscara fina, calcula el centro de la señal en cada columna y reduce el trazo a una línea de un píxel. Esto evita que el grosor de la línea introduzca errores en los picos.

## 7. Calibración

Una vez tenemos píxeles y anclas, hacemos la transformación geométrica. La posición X se convierte en fecha y la posición Y se convierte en precio. Para ello se usa interpolación lineal entre las anclas detectadas por OCR.

## 8. Extracción: línea trazada

En esta diapositiva se ve la señal detectada sobre la gráfica original. Sirve para comprobar visualmente si el algoritmo está siguiendo la línea correcta y si respeta los cambios bruscos de precio.

## 9. Extracción: máscara binaria

Aquí se muestra la misma señal aislada en blanco y negro. El blanco representa los píxeles que el sistema considera parte de la línea. Esta imagen es útil porque elimina el ruido visual de la interfaz y deja solo la estructura que se va a reconstruir.

## 10. Problemas corregidos

Durante el desarrollo aparecieron varios problemas prácticos. El eje X quedaba demasiado bajo, la máscara inicial era demasiado gruesa, el OCR detectaba algunas anclas falsas y algunas fechas salían incompletas. Estas correcciones mejoraron la precisión y la estabilidad del pipeline.

## 11. Cierre

En resumen, el proyecto convierte una gráfica rasterizada en datos estructurados. La visión artificial permite entender la imagen, el OCR permite leer las referencias de escala y la geometría permite reconstruir la serie temporal. El resultado queda listo para usarse después en análisis o modelos predictivos.
