from __future__ import annotations

import argparse
import json
from pathlib import Path

from recongrafica.pipeline import run_extract

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


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
    extract.add_argument(
        "--signal-quantile",
        type=float,
        default=0.5,
        help="Cuantil vertical del trazo usado como senal: 0.5 centro, 0.9 borde inferior. Por defecto: 0.5.",
    )
    extract.add_argument("--overwrite", action="store_true", help="Sobrescribe salidas existentes en vez de crear sufijo.")

    batch = subparsers.add_parser("batch", help="Procesa todas las imagenes de un directorio.")
    batch.add_argument("--raw-dir", default="data/raw", help="Directorio con capturas de entrada.")
    batch.add_argument("--out-dir", default="outputs", help="Directorio donde escribir las salidas.")
    batch.add_argument("--anchors-dir", default="data/anchors", help="Directorio con JSON de anclas por imagen.")
    batch.add_argument("--truth-dir", default=None, help="Directorio opcional con CSV ground truth por imagen.")
    batch.add_argument("--ocr-truth-dir", default=None, help="Directorio opcional con JSON de verdad OCR por imagen.")
    batch.add_argument("--lang", default="en", help="Idioma PaddleOCR principal. Por defecto: en.")
    batch.add_argument(
        "--signal-quantile",
        type=float,
        default=0.5,
        help="Cuantil vertical del trazo usado como senal: 0.5 centro, 0.9 borde inferior. Por defecto: 0.5.",
    )
    batch.add_argument("--use-ocr", action="store_true", help="Usa PaddleOCR cuando no exista JSON de anclas.")
    batch.add_argument("--recursive", action="store_true", help="Busca imagenes recursivamente dentro de raw-dir.")
    batch.add_argument("--overwrite", action="store_true", help="Sobrescribe salidas existentes en vez de crear sufijo.")
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
            signal_quantile=args.signal_quantile,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "batch":
        summary = run_batch_command(args)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 1 if summary["failed"] else 0
    parser.error("Comando no soportado")
    return 2


def run_batch_command(args: argparse.Namespace) -> dict:
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    anchors_dir = Path(args.anchors_dir)
    truth_dir = Path(args.truth_dir) if args.truth_dir else None
    ocr_truth_dir = Path(args.ocr_truth_dir) if args.ocr_truth_dir else None

    iterator = raw_dir.rglob("*") if args.recursive else raw_dir.iterdir()
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

    processed = []
    skipped = []
    failed = []
    ocr_disabled_reason = None
    for image_path in images:
        stem = image_path.stem
        anchors_path = anchors_dir / f"{stem}.json"
        if not anchors_path.exists():
            if not args.use_ocr:
                skipped.append({"image": str(image_path), "reason": f"No existe {anchors_path}"})
                continue
            if ocr_disabled_reason is not None:
                skipped.append({"image": str(image_path), "reason": ocr_disabled_reason})
                continue
            anchors_path = None

        truth_csv = truth_dir / f"{stem}.csv" if truth_dir else None
        if truth_csv is not None and not truth_csv.exists():
            truth_csv = None

        ocr_truth_json = ocr_truth_dir / f"{stem}.json" if ocr_truth_dir else None
        if ocr_truth_json is not None and not ocr_truth_json.exists():
            ocr_truth_json = None

        try:
            result = run_extract(
                image_path=image_path,
                out_prefix=out_dir / stem,
                truth_csv=truth_csv,
                anchors_path=anchors_path,
                ocr_truth_json=ocr_truth_json,
                languages=[args.lang],
                signal_quantile=args.signal_quantile,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - el batch debe continuar con el resto.
            error = str(exc)
            failed.append({"image": str(image_path), "error": error})
            if anchors_path is None and "PaddleOCR esta instalado" in error:
                ocr_disabled_reason = error
            continue
        processed.append({"image": str(image_path), **result})

    return {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "counts": {
            "images": len(images),
            "processed": len(processed),
            "skipped": len(skipped),
            "failed": len(failed),
        },
    }
