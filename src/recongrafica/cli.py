from __future__ import annotations

import argparse
import json
from pathlib import Path

from recongrafica.pipeline import run_extract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recongrafica", description="Reconstruye series temporales desde graficas Steam/CS2.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extrae una serie temporal desde una captura.")
    extract.add_argument("--image", required=True, help="Ruta de la imagen de entrada.")
    extract.add_argument("--out", required=True, help="Prefijo de salida, por ejemplo outputs/captura.")
    extract.add_argument("--truth", default=None, help="CSV opcional con columnas date,price.")
    extract.add_argument("--anchors", default=None, help="JSON opcional de anclas manuales.")
    extract.add_argument("--ocr-truth", default=None, help="JSON opcional con transcripciones OCR esperadas/predichas.")
    extract.add_argument("--lang", default="en", help="Idioma PaddleOCR principal. Por defecto: en.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "extract":
        result = run_extract(
            image_path=Path(args.image),
            out_prefix=Path(args.out),
            truth_csv=Path(args.truth) if args.truth else None,
            anchors_path=Path(args.anchors) if args.anchors else None,
            ocr_truth_json=Path(args.ocr_truth) if args.ocr_truth else None,
            languages=[args.lang],
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    parser.error("Comando no soportado")
    return 2
