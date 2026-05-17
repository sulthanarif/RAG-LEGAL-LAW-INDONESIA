"""
Finalize processed legal chunks before inserting into ChromaDB.

This is a conservative post-processing pass:
  - repair document metadata from raw extracted text/filename,
  - rebuild display/citation/embedding headers,
  - trim appendix text that was accidentally attached to the last Pasal,
  - flag low-quality sources for review,
  - write a quality report for anything still suspicious.

It does not rewrite legal substance with an LLM.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import praproses_ringan_pasal as prep


DEFAULT_INPUT = Path("data") / "processed_chunks_ringan_pasal.json"
DEFAULT_OUTPUT = Path("data") / "processed_chunks_ringan_pasal_chroma_ready.json"
DEFAULT_REPORT = Path("data") / "processed_chunks_ringan_pasal_chroma_report.json"
DEFAULT_RAW_TEXT_DIR = Path("data") / "raw_text_ringan"


def source_file(chunk: Dict) -> str:
    return chunk.get("metadata", {}).get("source_file", "")


def load_raw_text(raw_text_dir: Path, source: str) -> str:
    raw_path = raw_text_dir / f"{Path(source).stem}.txt"
    if raw_path.exists():
        return raw_path.read_text(encoding="utf-8")
    return ""


def repair_metadata_for_source(source: str, sample_meta: Dict, raw_text_dir: Path) -> Dict:
    raw_text = load_raw_text(raw_text_dir, source)
    source_path = Path(sample_meta.get("source_path") or source)
    if raw_text:
        repaired = prep.extract_metadata(source_path, raw_text)
    else:
        nomor, year = prep.extract_nomor_year_from_filename(source)
        repaired = {
            key: sample_meta.get(key)
            for key in [
                "regulation_type",
                "regulation_hierarchy",
                "nomor",
                "tentang",
                "year",
                "publication_year",
                "active_status",
                "source_file",
                "source_path",
                "extractor",
                "chunk_level",
            ]
        }
        if nomor:
            repaired["nomor"] = nomor
        if year:
            repaired["year"] = year
            repaired["publication_year"] = year

    for key in [
        "extractor",
        "chunk_level",
        "extraction_page_count",
        "extraction_counts",
        "extraction_char_count",
        "extraction_word_count",
        "extraction_pasal_count",
        "extraction_noise_ratio",
        "extraction_ocr_artifact_count",
    ]:
        if key in sample_meta:
            repaired[key] = sample_meta[key]
    return repaired


def section_context_from_meta(meta: Dict) -> prep.SectionContext:
    return prep.SectionContext(
        bab=meta.get("bab", ""),
        bab_title=meta.get("bab_title", ""),
        bagian=meta.get("bagian", ""),
        bagian_title=meta.get("bagian_title", ""),
        paragraf=meta.get("paragraf", ""),
        paragraf_title=meta.get("paragraf_title", ""),
    )


def trim_attached_appendix(text: str) -> Tuple[str, bool]:
    trimmed = prep.strip_after_appendix(text)
    return trimmed, trimmed != text.strip()


def split_long_text(text: str, max_words: int, overlap_words: int) -> List[str]:
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return [text]

    units = [unit.strip() for unit in re.split(r"(?<=[.;:])\s+|\n+", text) if unit.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_count = 0

    def flush_current() -> None:
        nonlocal current, current_count
        if current:
            chunks.append(" ".join(current).strip())
            if overlap_words > 0:
                overlap = re.findall(r"\S+", chunks[-1])[-overlap_words:]
                current = [" ".join(overlap)] if overlap else []
                current_count = len(overlap)
            else:
                current = []
                current_count = 0

    for unit in units:
        unit_words = re.findall(r"\S+", unit)
        if len(unit_words) > max_words:
            flush_current()
            step = max(1, max_words - overlap_words)
            for start in range(0, len(unit_words), step):
                part = unit_words[start : start + max_words]
                if part:
                    chunks.append(" ".join(part).strip())
            current = []
            current_count = 0
            continue

        if current_count and current_count + len(unit_words) > max_words:
            flush_current()
        current.append(unit)
        current_count += len(unit_words)

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def should_quarantine(meta: Dict, max_noise_ratio: float, max_ocr_artifact_count: int) -> Tuple[bool, List[str]]:
    reasons = []
    if meta.get("extraction_noise_ratio", 0) > max_noise_ratio:
        reasons.append(f"noise_ratio>{max_noise_ratio}: {meta.get('extraction_noise_ratio')}")
    if meta.get("extraction_ocr_artifact_count", 0) > max_ocr_artifact_count:
        reasons.append(f"ocr_artifact_count>{max_ocr_artifact_count}: {meta.get('extraction_ocr_artifact_count')}")
    return bool(reasons), reasons


def suspicious_reasons(chunk: Dict) -> List[str]:
    meta = chunk.get("metadata", {})
    reasons = []
    if meta.get("nomor") in ("", "Unknown", None) or meta.get("year", 0) in (0, None):
        reasons.append("unknown_nomor_or_year")
    if meta.get("tentang") in ("", "Unknown", None):
        reasons.append("unknown_tentang")
    if re.search(r"\bPasal\s+[LI]\b", meta.get("pasal_id", "")):
        reasons.append("suspicious_pasal_id")
    if len(chunk.get("text", "")) > 12000:
        reasons.append("very_long_chunk")
    return reasons


def finalize_chunks(args: argparse.Namespace) -> Dict:
    chunks = json.loads(args.input.read_text(encoding="utf-8"))
    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for chunk in chunks:
        by_source[source_file(chunk)].append(chunk)

    repaired_by_source = {
        src: repair_metadata_for_source(src, items[0].get("metadata", {}), args.raw_text_dir)
        for src, items in by_source.items()
    }

    output = []
    review_sources = {}
    dropped_sources = {}
    suspicious = []
    appendix_trim_count = 0
    dropped_chunks = 0

    for chunk in chunks:
        src = source_file(chunk)
        old_meta = chunk.get("metadata", {})
        source_meta = repaired_by_source[src]
        needs_review, q_reasons = should_quarantine(source_meta, args.max_noise_ratio, args.max_ocr_artifact_count)
        if needs_review:
            review_sources.setdefault(src, {"source_file": src, "reasons": q_reasons, "chunk_count": 0})
            review_sources[src]["chunk_count"] += 1
        if needs_review and args.drop_quarantined:
            dropped_sources.setdefault(src, {"source_file": src, "reasons": q_reasons, "chunk_count": 0})
            dropped_sources[src]["chunk_count"] += 1
            dropped_chunks += 1
            continue

        content, trimmed = trim_attached_appendix(chunk.get("text", ""))
        content = prep.clean_extracted_text(content)
        if trimmed:
            appendix_trim_count += 1
        if not content.strip():
            continue

        pasal_id = old_meta.get("pasal_id", "")
        ctx = section_context_from_meta(old_meta)
        parts = split_long_text(content, args.max_words, args.overlap_words)

        for part_index, part in enumerate(parts, start=1):
            part_pasal_id = pasal_id if len(parts) == 1 else f"{pasal_id} bagian {part_index}"
            kind = old_meta.get("chunk_kind", "pasal") if len(parts) == 1 else "pasal_split"
            token_count = prep.estimate_token_count(part)
            meta = prep.build_chunk_metadata(source_meta, ctx, pasal_id, kind, old_meta.get("chunk_index", 0), token_count)
            meta["quality_status"] = "needs_review" if needs_review else "ok"
            meta["quality_reasons"] = json.dumps(q_reasons, ensure_ascii=False) if q_reasons else ""
            if len(parts) > 1:
                meta["part_index"] = part_index
                meta["part_count"] = len(parts)
            display_text = prep.build_display_text(source_meta, part_pasal_id, part, ctx)
            citation_text = prep.build_citation_text(source_meta, part_pasal_id)
            id_seed = "::".join(
                [
                    src,
                    str(chunk.get("id", "")),
                    str(old_meta.get("chunk_index", "")),
                    pasal_id,
                    str(part_index),
                    hashlib.sha1(part.encode("utf-8")).hexdigest()[:16],
                ]
            )

            finalized = {
                **chunk,
                "id": hashlib.sha1(id_seed.encode("utf-8")).hexdigest(),
                "text": part,
                "display_text": display_text,
                "embedding_text": display_text,
                "citation_text": citation_text,
                "metadata": meta,
            }
            reasons = suspicious_reasons(finalized)
            if reasons:
                suspicious.append({"source_file": src, "pasal_id": part_pasal_id, "reasons": reasons})
            output.append(finalized)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "input_chunks": len(chunks),
        "output_chunks": len(output),
        "dropped_chunks": dropped_chunks,
        "dropped_sources": list(dropped_sources.values()),
        "review_sources": list(review_sources.values()),
        # Backward-compatible names. These are only populated when --drop-quarantined is used.
        "quarantined_chunks": dropped_chunks,
        "quarantined_sources": list(dropped_sources.values()),
        "appendix_trimmed_chunks": appendix_trim_count,
        "suspicious_chunks": suspicious,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize legal chunks before ChromaDB ingestion.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--raw-text-dir", type=Path, default=DEFAULT_RAW_TEXT_DIR)
    parser.add_argument("--max-noise-ratio", type=float, default=prep.MAX_NOISE_RATIO)
    parser.add_argument("--max-ocr-artifact-count", type=int, default=prep.MAX_OCR_ARTIFACT_COUNT)
    parser.add_argument("--max-words", type=int, default=450)
    parser.add_argument("--overlap-words", type=int, default=60)
    parser.add_argument(
        "--drop-quarantined",
        action="store_true",
        help="Drop sources whose extraction quality exceeds noise/artifact thresholds. Default keeps them and flags quality_status=needs_review.",
    )
    parser.add_argument(
        "--keep-quarantined",
        action="store_true",
        help="Deprecated compatibility flag. The default already keeps review-quality sources.",
    )
    return parser.parse_args()


def main() -> None:
    report = finalize_chunks(parse_args())
    print(f"Input chunks: {report['input_chunks']}")
    print(f"Output chunks: {report['output_chunks']}")
    print(f"Review sources: {len(report['review_sources'])}")
    print(f"Dropped sources: {len(report['dropped_sources'])}")
    print(f"Suspicious chunks: {len(report['suspicious_chunks'])}")
    print(f"Appendix-trimmed chunks: {report['appendix_trimmed_chunks']}")


if __name__ == "__main__":
    main()
