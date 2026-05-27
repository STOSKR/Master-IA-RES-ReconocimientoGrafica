from __future__ import annotations

from pathlib import Path
from typing import Any

from recongrafica.calibration import anchors_from_ocr, clean_anchors_for_reconstruction, reconstruct_series
from recongrafica.evaluation import evaluate_ocr_text, evaluate_series
from recongrafica.exporters import build_json_payload, export_csv, export_json, save_debug_image, save_mask
from recongrafica.image_io import load_image
from recongrafica.layout import detect_layout
from recongrafica.ocr import merge_ocr_results, read_manual_anchors, run_ocr_baselines
from recongrafica.signal import extract_price_signal


def run_extract(
    image_path: str | Path,
    out_prefix: str | Path,
    truth_csv: str | Path | None = None,
    anchors_path: str | Path | None = None,
    ocr_truth_json: str | Path | None = None,
    languages: list[str] | None = None,
    signal_quantile: float = 0.5,
    overwrite: bool = False,
) -> dict[str, Any]:
    image = load_image(image_path)
    layout = detect_layout(image)

    baselines = {}
    if anchors_path is not None:
        ocr_results = read_manual_anchors(anchors_path)
        baselines["manual"] = ocr_results
    else:
        baselines = run_ocr_baselines(image, layout, languages=languages)
        ocr_results = merge_ocr_results(baselines)

    anchors, rejected = anchors_from_ocr(ocr_results)
    anchors = clean_anchors_for_reconstruction(anchors)
    signal, mask = extract_price_signal(image, layout.plot_area, y_quantile=signal_quantile)
    series = reconstruct_series(signal, anchors)

    metrics = {
        "series": evaluate_series(series, truth_csv),
        "ocr": evaluate_ocr_text(ocr_truth_json, [result.text for result in ocr_results]),
        "processed": True,
    }

    out_prefix = Path(out_prefix)
    if not overwrite:
        out_prefix = _next_available_prefix(out_prefix)
    csv_path, json_path, debug_path, mask_path = _output_paths(out_prefix)

    export_csv(series, csv_path)
    payload = build_json_payload(
        image.shape,
        layout,
        ocr_results,
        anchors,
        rejected,
        signal,
        series,
        metrics,
        baselines,
    )
    export_json(payload, json_path)
    save_debug_image(image, layout, ocr_results, anchors, signal, debug_path)
    save_mask(mask, mask_path)

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "debug": str(debug_path),
        "mask": str(mask_path),
        "metrics": metrics,
        "points": len(series),
    }


def _output_paths(out_prefix: Path) -> tuple[Path, Path, Path, Path]:
    return (
        out_prefix.with_suffix(".csv"),
        out_prefix.with_suffix(".json"),
        out_prefix.with_name(out_prefix.name + "_debug").with_suffix(".png"),
        out_prefix.with_name(out_prefix.name + "_mask").with_suffix(".png"),
    )


def _next_available_prefix(out_prefix: Path) -> Path:
    if not any(path.exists() for path in _output_paths(out_prefix)):
        return out_prefix
    for index in range(1, 10_000):
        candidate = out_prefix.with_name(f"{out_prefix.name}_{index:03d}")
        if not any(path.exists() for path in _output_paths(candidate)):
            return candidate
    raise RuntimeError(f"No se encontro un nombre de salida libre para {out_prefix}")
