from datetime import date

from recongrafica.parsing import cer, parse_date, parse_price, wer
from recongrafica.evaluation import evaluate_ocr_text


def test_parse_price_accepts_euro_and_decimal_comma():
    assert parse_price("1,80€") == 1.8
    assert parse_price("EUR 2.35") == 2.35


def test_parse_date_accepts_common_axis_formats():
    assert parse_date("26/09/2025") == date(2025, 9, 26)
    assert parse_date("26-09-25") == date(2025, 9, 26)


def test_text_error_metrics():
    assert cer("abc", "abc") == 0
    assert cer("abc", "axc") == 1 / 3
    assert wer("hello world", "hello steam") == 0.5


def test_evaluate_ocr_text_uses_pipeline_predictions(tmp_path):
    payload = tmp_path / "ocr.json"
    payload.write_text('{"ocr_text": {"expected": ["abc"]}}', encoding="utf-8")

    metrics = evaluate_ocr_text(payload, ["axc"])

    assert metrics["pairs"] == 1
    assert metrics["cer"] == 1 / 3
