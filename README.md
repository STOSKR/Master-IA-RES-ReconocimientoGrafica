# ReconGrafica

Pipeline reproducible para reconstruir series temporales financieras a partir de capturas rasterizadas de graficas historicas de Steam/CS2.

La primera version esta acotada a capturas tipo Steam Community Market: detecta regiones de interes, lee anclas de ejes con OCR, extrae la linea visual del precio y transforma coordenadas de pixel a valores reales.

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Para activar OCR real con PaddleOCR:

```powershell
pip install -e ".[dev,ocr]"
```

## Uso CLI

```powershell
python -m recongrafica extract --image data/raw/captura.png --out outputs/captura --truth data/ground_truth/captura.csv
```

Si PaddleOCR no esta instalado o se quiere fijar una referencia manual, se puede aportar un JSON de anclas:

```powershell
python -m recongrafica extract --image data/raw/captura.png --out outputs/captura --anchors data/anchors/captura.json
```

Formato minimo de `anchors.json`:

```json
{
  "y": [
    {"text": "0,50€", "pixel": [58, 420]},
    {"text": "2,00€", "pixel": [58, 120]}
  ],
  "x": [
    {"text": "26/09/2025", "pixel": [120, 465]},
    {"text": "26/12/2025", "pixel": [760, 465]}
  ]
}
```

## Salidas

Para un prefijo `outputs/captura`, el pipeline genera:

- `outputs/captura.csv`: serie reconstruida con `date`, `price`, `pixel_x`, `pixel_y`, `confidence`.
- `outputs/captura.json`: metadatos, anclas, resultados OCR, serie y metricas.
- `outputs/captura_debug.png`: visualizacion de plot area, ROIs, cajas OCR/anclas y linea extraida.
- `outputs/captura_mask.png`: mascara binaria usada para extraer la senal.

## Evaluacion

Si se aporta `--truth`, el CSV de control debe contener columnas `date` y `price`. Se calculan MAE, RMSE y MSE tras interpolar la serie real sobre las fechas reconstruidas.

Las metricas OCR CER/WER se calculan cuando se aporta ground truth textual de OCR en un JSON con:

```json
{
  "ocr_text": {
    "expected": ["0,50€", "2,00€", "26/09/2025"]
  }
}
```

Si el JSON incluye tambien `predicted`, se usara ese valor; si no, se comparara contra el texto reconocido por el pipeline.

## Estructura

- `src/recongrafica`: codigo del pipeline.
- `tests`: pruebas unitarias y end-to-end con imagen sintetica.
- `data/raw`: capturas originales.
- `data/ground_truth`: series reales de control.
- `data/anchors`: anclas manuales opcionales.
- `outputs`: resultados generados.
- `notebooks`: material academico/demos.
- `docs`: memoria tecnica.
