"""
OCR fallback for PDFs that were moved to data/pdf_butuh_ocr.

Pipeline:
  PDF in data/pdf_butuh_ocr
    -> render each page to image
    -> PaddleOCR
    -> save raw OCR text to data/raw_text_paddle
    -> reuse praproses_ringan_pasal metadata + pasal chunker
    -> merge/replace those sources into processed_chunks_ringan_pasal.json

Examples:
    python paddle_ocr_extract.py --limit 1
    python paddle_ocr_extract.py --device gpu:0
    python paddle_ocr_extract.py --pdf data/pdf_butuh_ocr/5PP2021.pdf
    python paddle_ocr_extract.py --merge-output data/processed_chunks_ringan_pasal_paddle_merged.json
"""

import argparse
import json
import re
import sys
import tempfile
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fitz

import praproses_ringan_pasal as prep


DEFAULT_OCR_PDF_DIR = Path("data") / "pdf_butuh_ocr"
DEFAULT_RAW_TEXT_DIR = Path("data") / "raw_text_paddle"
DEFAULT_CHUNKS_OUTPUT = Path("data") / "processed_chunks_paddle_ocr.json"
DEFAULT_BASE_JSON = Path("data") / "processed_chunks_ringan_pasal.json"
DEFAULT_MERGE_OUTPUT = Path("data") / "processed_chunks_ringan_pasal.json"
DEFAULT_REPORT = Path("data") / "paddle_ocr_report.json"


def iter_pdf_files(pdf_dir: Path) -> Iterable[Path]:
    yield from sorted(pdf_dir.glob("*.pdf"))


def install_lightweight_langchain_splitter_shim() -> None:
    """Avoid PaddleX importing transformers/torch just to satisfy a LangChain shim."""
    if "langchain_text_splitters" in sys.modules:
        return

    module = types.ModuleType("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass

    module.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules["langchain_text_splitters"] = module


def install_lightweight_modelscope_shim() -> None:
    """Keep PaddleX import from loading ModelScope, which imports torch."""
    if "modelscope" in sys.modules:
        return

    modelscope = types.ModuleType("modelscope")

    def snapshot_download(*args, **kwargs):
        raise RuntimeError("ModelScope disabled in paddle_ocr_extract.py; use Paddle/BOS cached models.")

    modelscope.snapshot_download = snapshot_download

    hub = types.ModuleType("modelscope.hub")
    errors = types.ModuleType("modelscope.hub.errors")

    class HTTPError(Exception):
        pass

    class NotExistError(Exception):
        pass

    errors.HTTPError = HTTPError
    errors.NotExistError = NotExistError
    hub.errors = errors
    modelscope.hub = hub

    sys.modules["modelscope"] = modelscope
    sys.modules["modelscope.hub"] = hub
    sys.modules["modelscope.hub.errors"] = errors


def load_paddle_ocr(
    lang: str,
    engine: str,
    det_model: str,
    rec_model: str,
    rec_batch_size: int,
    device: str,
):
    try:
        import paddle
    except ImportError as exc:
        raise RuntimeError("paddle belum terinstall. Install requirements Paddle dulu.") from exc

    if device == "auto":
        device = "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
    if device.startswith("gpu") and not paddle.device.is_compiled_with_cuda():
        raise RuntimeError(
            f"Device diminta {device}, tapi Paddle yang terinstall CPU build. "
            "Install requirements-paddle-gpu.txt dulu."
        )
    paddle.device.set_device(device)
    print(f"Paddle device: {paddle.device.get_device()}", flush=True)

    install_lightweight_langchain_splitter_shim()
    install_lightweight_modelscope_shim()
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "paddleocr belum terinstall. Jalankan:\n"
            "  python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n"
            "  python -m pip install paddleocr"
        ) from exc

    init_attempts = [
        {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_detection_model_name": det_model,
            "text_recognition_model_name": rec_model,
            "text_recognition_batch_size": rec_batch_size,
            "engine": engine,
        },
        {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_recognition_batch_size": rec_batch_size,
            "engine": engine,
        },
        {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": lang,
            "use_angle_cls": False,
        },
        {"lang": lang},
    ]
    last_error = None
    for kwargs in init_attempts:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:  # PaddleOCR has changed init kwargs across versions.
            last_error = exc
    raise RuntimeError(f"Gagal inisialisasi PaddleOCR: {last_error}") from last_error


def render_page_to_png(page: fitz.Page, output_path: Path, dpi: int) -> None:
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pix.save(str(output_path))


def flatten_ocr_texts(obj) -> List[Tuple[str, float]]:
    texts: List[Tuple[str, float]] = []

    def add_text(text, score=0.0) -> None:
        if isinstance(text, str) and text.strip():
            try:
                score_f = float(score)
            except Exception:
                score_f = 0.0
            texts.append((text.strip(), score_f))

    def walk(value) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            if "rec_texts" in value:
                scores = value.get("rec_scores") or []
                for idx, text in enumerate(value.get("rec_texts") or []):
                    score = scores[idx] if idx < len(scores) else 0.0
                    add_text(text, score)
                return
            if "rec_text" in value:
                add_text(value.get("rec_text"), value.get("rec_score", 0.0))
                return
            if "res" in value:
                walk(value["res"])
                return
            for item in value.values():
                walk(item)
            return
        if isinstance(value, (list, tuple)):
            # PaddleOCR 2.x commonly returns [box, (text, score)].
            if len(value) == 2 and isinstance(value[1], (list, tuple)) and value[1] and isinstance(value[1][0], str):
                score = value[1][1] if len(value[1]) > 1 else 0.0
                add_text(value[1][0], score)
                return
            for item in value:
                walk(item)
            return
        if hasattr(value, "json"):
            try:
                walk(value.json)
                return
            except Exception:
                pass
        if hasattr(value, "res"):
            try:
                walk(value.res)
                return
            except Exception:
                pass

    walk(obj)
    return texts


def ocr_image(ocr, image_path: Path) -> Tuple[str, Dict]:
    if hasattr(ocr, "predict"):
        result = ocr.predict(str(image_path))
    elif hasattr(ocr, "ocr"):
        result = ocr.ocr(str(image_path), cls=False)
    else:
        raise RuntimeError("Objek PaddleOCR tidak punya method predict/ocr.")

    pairs = flatten_ocr_texts(result)
    page_text = "\n".join(text for text, _score in pairs)
    scores = [score for _text, score in pairs if score > 0]
    return page_text, {
        "line_count": len(pairs),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
    }


def extract_text_paddle(pdf_path: Path, ocr, dpi: int, max_pages: Optional[int]) -> Tuple[str, Dict]:
    page_texts = []
    page_infos = []
    with fitz.open(pdf_path) as doc, tempfile.TemporaryDirectory(prefix="paddle_ocr_") as tmp:
        tmp_dir = Path(tmp)
        page_total = len(doc) if max_pages is None else min(len(doc), max_pages)
        for page_index in range(page_total):
            page = doc[page_index]
            image_path = tmp_dir / f"page_{page_index + 1:04d}.png"
            render_page_to_png(page, image_path, dpi)
            text, info = ocr_image(ocr, image_path)
            text = prep.clean_extracted_text(text)
            page_texts.append(text)
            info["page"] = page_index + 1
            info["char_count"] = len(text)
            page_infos.append(info)
            print(
                f"    page {page_index + 1}/{page_total}: {len(text)} chars, "
                f"{info['line_count']} lines, avg_score={info['avg_score']}",
                flush=True,
            )

    combined = prep.clean_extracted_text("\n\n".join(t for t in page_texts if t))
    return combined, {
        "page_count": len(page_infos),
        "page_infos": page_infos,
        "extractor_counts": {"paddleocr": len(page_infos)},
    }


def chunks_from_text(pdf_path: Path, text: str, extraction_info: Dict) -> Tuple[List[Dict], Dict]:
    quality = prep.extraction_quality(text)
    metadata = prep.extract_metadata(pdf_path, text)
    metadata.update({
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "extractor": "paddleocr",
        "extraction_page_count": extraction_info["page_count"],
        "extraction_counts": json.dumps(extraction_info["extractor_counts"], ensure_ascii=False),
        "extraction_char_count": quality["char_count"],
        "extraction_word_count": quality["word_count"],
        "extraction_pasal_count": quality["pasal_count"],
        "extraction_noise_ratio": quality["noise_ratio"],
        "extraction_ocr_artifact_count": quality["ocr_artifact_count"],
    })
    chunks = prep.structural_chunking_per_pasal(text, metadata)
    return chunks, quality


def replace_sources(base_chunks: List[Dict], new_chunks: List[Dict]) -> List[Dict]:
    new_sources = {c.get("metadata", {}).get("source_file", "") for c in new_chunks}
    kept = [c for c in base_chunks if c.get("metadata", {}).get("source_file", "") not in new_sources]
    return kept + new_chunks


def process_pdf(pdf_path: Path, args: argparse.Namespace, ocr) -> Tuple[List[Dict], Dict]:
    raw_path = args.raw_text_dir / f"{pdf_path.stem}.txt"
    meta_path = args.raw_text_dir / f"{pdf_path.stem}.meta.json"
    source_size = pdf_path.stat().st_size
    source_mtime = int(pdf_path.stat().st_mtime)
    with fitz.open(pdf_path) as doc:
        pdf_page_count = len(doc)
    expected_pages = min(pdf_page_count, args.max_pages) if args.max_pages else pdf_page_count

    cache_valid = False
    if raw_path.exists() and meta_path.exists() and not args.force:
        try:
            cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cache_valid = (
                cache_meta.get("source_size") == source_size
                and cache_meta.get("source_mtime") == source_mtime
                and cache_meta.get("ocr_pages") == expected_pages
                and cache_meta.get("pdf_page_count") == pdf_page_count
            )
        except Exception:
            cache_valid = False

    if cache_valid:
        print(f"Memakai raw OCR cache: {raw_path}")
        text = raw_path.read_text(encoding="utf-8")
        extraction_info = {
            "page_count": expected_pages,
            "extractor_counts": {"paddleocr_cache": expected_pages},
        }
    else:
        print(f"OCR Paddle: {pdf_path.name}", flush=True)
        text, extraction_info = extract_text_paddle(pdf_path, ocr, args.dpi, args.max_pages)
        args.raw_text_dir.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(text, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "source_file": str(pdf_path),
                    "source_size": source_size,
                    "source_mtime": source_mtime,
                    "pdf_page_count": pdf_page_count,
                    "ocr_pages": expected_pages,
                    "max_pages": args.max_pages,
                    "dpi": args.dpi,
                    "det_model": args.det_model,
                    "rec_model": args.rec_model,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    chunks, quality = chunks_from_text(pdf_path, text, extraction_info)
    report = {
        "source_file": pdf_path.name,
        "raw_text_path": str(raw_path),
        "chunk_count": len(chunks),
        "quality": quality,
    }
    print(
        f"  -> {len(chunks)} chunks | chars={quality['char_count']} "
        f"pasal={quality['pasal_count']} noise={quality['noise_ratio']} "
        f"artifacts={quality['ocr_artifact_count']}",
        flush=True,
    )
    return chunks, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR for PDFs that need OCR and merge chunks into JSON.")
    parser.add_argument("--ocr-pdf-dir", type=Path, default=DEFAULT_OCR_PDF_DIR)
    parser.add_argument("--pdf", type=Path, help="Process one PDF instead of the whole OCR folder.")
    parser.add_argument("--raw-text-dir", type=Path, default=DEFAULT_RAW_TEXT_DIR)
    parser.add_argument("--chunks-output", type=Path, default=DEFAULT_CHUNKS_OUTPUT)
    parser.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    parser.add_argument("--merge-output", type=Path, default=DEFAULT_MERGE_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--lang", default="en", help="PaddleOCR recognition language. Use 'en' for Indonesian Latin text.")
    parser.add_argument("--engine", default="paddle", choices=["paddle", "transformers"])
    parser.add_argument("--det-model", default="PP-OCRv5_mobile_det")
    parser.add_argument("--rec-model", default="en_PP-OCRv5_mobile_rec")
    parser.add_argument("--rec-batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="Paddle device: auto, gpu:0, gpu:1, or cpu. Auto uses GPU when Paddle GPU build is installed.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int, help="Debug: OCR only first N pages per PDF.")
    parser.add_argument("--limit", type=int, help="Debug: OCR only first N PDFs.")
    parser.add_argument("--force", action="store_true", help="Ignore raw_text_paddle cache and OCR again.")
    parser.add_argument("--no-merge", action="store_true", help="Only write PaddleOCR chunks, do not merge into base JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdfs = [args.pdf] if args.pdf else list(iter_pdf_files(args.ocr_pdf_dir))
    if args.limit is not None:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit(f"Tidak ada PDF di {args.ocr_pdf_dir}")

    ocr = load_paddle_ocr(
        args.lang,
        args.engine,
        args.det_model,
        args.rec_model,
        args.rec_batch_size,
        args.device,
    )

    all_new_chunks: List[Dict] = []
    reports = []
    for pdf_path in pdfs:
        chunks, report = process_pdf(pdf_path, args, ocr)
        all_new_chunks.extend(chunks)
        reports.append(report)

    args.chunks_output.parent.mkdir(parents=True, exist_ok=True)
    args.chunks_output.write_text(json.dumps(all_new_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_merge:
        base_chunks = json.loads(args.base_json.read_text(encoding="utf-8")) if args.base_json.exists() else []
        merged = replace_sources(base_chunks, all_new_chunks)
        args.merge_output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        merged_count = len(merged)
    else:
        merged_count = None

    args.report.write_text(json.dumps({
        "processed_pdfs": [str(p) for p in pdfs],
        "new_chunks": len(all_new_chunks),
        "merged_output": None if args.no_merge else str(args.merge_output),
        "merged_count": merged_count,
        "reports": reports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"PaddleOCR chunks: {len(all_new_chunks)} -> {args.chunks_output}")
    if not args.no_merge:
        print(f"Merged chunks: {merged_count} -> {args.merge_output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
