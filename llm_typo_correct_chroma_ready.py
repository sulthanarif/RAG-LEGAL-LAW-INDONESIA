"""
LLM-assisted typo correction for Chroma-ready legal chunks.

This script corrects OCR/extraction typos in chunk text while preserving legal
substance, article numbers, paragraph numbers, and metadata. It writes a new
JSON file so the original Chroma-ready artifact stays untouched.

Examples:
    python llm_typo_correct_chroma_ready.py --provider transformers --model Qwen/Qwen2.5-7B-Instruct --limit 10
    python llm_typo_correct_chroma_ready.py --provider ollama --model qwen2.5:7b-instruct --limit 10
    python llm_typo_correct_chroma_ready.py --provider gemini --model gemini-2.5-flash --limit 10
    python llm_typo_correct_chroma_ready.py --provider openai-compatible --model Qwen/Qwen2.5-7B-Instruct
    python llm_typo_correct_chroma_ready.py --provider transformers --model Qwen/Qwen2.5-7B-Instruct --device cuda
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from tqdm.auto import tqdm

import praproses_ringan_pasal as prep
from praproses_ringan_pasal import SectionContext


DEFAULT_INPUT = Path("data") / "processed_chunks_ringan_pasal_chroma_ready.json"
DEFAULT_OUTPUT = Path("data") / "processed_chunks_ringan_pasal_chroma_ready_typo_corrected.json"
DEFAULT_CACHE = Path("data") / "llm_typo_correction_cache.jsonl"


SYSTEM_PROMPT = """Anda adalah editor teks hukum Indonesia.
Tugas Anda HANYA memperbaiki typo OCR dan spasi yang rusak.

Aturan wajib:
- Jangan meringkas.
- Jangan menambah fakta, pasal, ayat, atau frasa baru.
- Jangan menghapus substansi hukum.
- Pertahankan nomor Pasal, ayat, huruf, angka, persen, tanggal, dan nama peraturan.
- Perbaiki typo OCR umum seperti o/o -> %, olo -> %, 2o14 -> 2014, PRES IDEN -> PRESIDEN.
- Jika ragu, pertahankan teks asli.
- Balas teks hasil koreksi saja, tanpa markdown, tanpa komentar.
"""


def compact_citation(meta: Dict[str, Any]) -> str:
    reg_type = str(meta.get("regulation_type") or "Aturan").strip()
    nomor = str(meta.get("nomor") or "").strip()
    year = str(meta.get("publication_year") or meta.get("year") or "").strip()
    pasal = str(meta.get("pasal_id") or "").strip()
    parts = [reg_type]
    if nomor and nomor.lower() != "unknown":
        parts.append(f"No. {nomor}")
    if year and year != "0":
        parts.append(f"Tahun {year}")
    citation = " ".join(parts).strip()
    return f"{citation}, {pasal}" if pasal else citation


def section_context_from_meta(meta: Dict[str, Any]) -> SectionContext:
    return SectionContext(
        bab=meta.get("bab", "") or "",
        bab_title=meta.get("bab_title", "") or "",
        bagian=meta.get("bagian", "") or "",
        bagian_title=meta.get("bagian_title", "") or "",
        paragraf=meta.get("paragraf", "") or "",
        paragraf_title=meta.get("paragraf_title", "") or "",
    )


def rebuild_display_text(meta: Dict[str, Any], text: str) -> str:
    source_meta = {
        "regulation_type": meta.get("regulation_type") or "Aturan",
        "nomor": meta.get("nomor") or "Unknown",
        "publication_year": meta.get("publication_year") or meta.get("year") or 0,
    }
    return prep.build_display_text(
        source_meta,
        meta.get("pasal_id", "") or "",
        text,
        section_context_from_meta(meta),
    )


def clean_model_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^\s*(?:Hasil koreksi|Teks hasil koreksi)\s*:\s*", "", text, flags=re.IGNORECASE)
    return prep.clean_extracted_text(text)


def prompt_for_chunk(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata", {})
    return (
        f"SITASI: {compact_citation(meta)}\n"
        "TEKS YANG HARUS DIKOREKSI:\n"
        f"{chunk.get('text', '')}"
    )


def load_cache(path: Path) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[item["cache_key"]] = item["corrected_text"]
    return cache


def append_cache(path: Path, cache_key: str, chunk_id: str, corrected_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({
            "cache_key": cache_key,
            "chunk_id": chunk_id,
            "corrected_text": corrected_text,
        }, ensure_ascii=False) + "\n")


def cache_key_for(chunk: Dict[str, Any], model: str, provider: str) -> str:
    payload = json.dumps({
        "provider": provider,
        "model": model,
        "id": chunk.get("id"),
        "text": chunk.get("text", ""),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def call_gemini(model: str, prompt: str, max_new_tokens: int, temperature: float, timeout: int) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY atau GOOGLE_API_KEY untuk provider gemini.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_new_tokens,
        },
    }
    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini error {response.status_code}: {response.text[:1000]}")
    data = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def call_ollama(model: str, prompt: str, max_new_tokens: int, temperature: float, timeout: int) -> str:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    payload = {
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_new_tokens,
        },
    }
    response = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text[:1000]}")
    return response.json().get("response", "")


def call_openai_compatible(model: str, prompt: str, max_new_tokens: int, temperature: float, timeout: int) -> str:
    base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not base_url:
        raise RuntimeError("Set OPENAI_COMPATIBLE_BASE_URL untuk provider openai-compatible.")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    response = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI-compatible error {response.status_code}: {response.text[:1000]}")
    return response.json()["choices"][0]["message"]["content"]


class TransformersCaller:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = args.model
        device = args.device
        load_in_4bit = args.load_in_4bit or (args.auto_4bit_7b and re.search(r"(^|[-_/])7b($|[-_/])", model, re.IGNORECASE))
        if load_in_4bit:
            print(f"Loading {model} in 4-bit.", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        quantization_config = None
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("Install bitsandbytes untuk --load-in-4bit.") from exc
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        if args.torch_dtype == "auto":
            dtype = "auto"
        elif args.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if device.startswith("cuda") else None,
            "trust_remote_code": True,
            "quantization_config": quantization_config,
        }
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        if not device.startswith("cuda"):
            self.model.to(device)

    def __call__(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def correct_text(args: argparse.Namespace, prompt: str, transformers_caller: Optional[TransformersCaller]) -> str:
    if args.provider == "gemini":
        return call_gemini(args.model, prompt, args.max_new_tokens, args.temperature, args.timeout)
    if args.provider == "ollama":
        return call_ollama(args.model, prompt, args.max_new_tokens, args.temperature, args.timeout)
    if args.provider == "openai-compatible":
        return call_openai_compatible(args.model, prompt, args.max_new_tokens, args.temperature, args.timeout)
    if args.provider == "transformers":
        if transformers_caller is None:
            raise RuntimeError("Transformers caller belum diinisialisasi.")
        return transformers_caller(prompt, args.max_new_tokens, args.temperature)
    raise ValueError(f"Provider tidak dikenal: {args.provider}")


def should_process(chunk: Dict[str, Any], args: argparse.Namespace) -> bool:
    if args.only_needs_review and chunk.get("metadata", {}).get("quality_status") != "needs_review":
        return False
    text = chunk.get("text", "")
    if len(text.split()) < args.min_words:
        return False
    return True


def update_chunk(chunk: Dict[str, Any], corrected_text: str) -> Dict[str, Any]:
    meta = dict(chunk.get("metadata", {}))
    corrected_text = clean_model_output(corrected_text)
    display_text = rebuild_display_text(meta, corrected_text)
    citation_text = compact_citation(meta)
    updated = dict(chunk)
    updated["text"] = corrected_text
    updated["display_text"] = display_text
    updated["embedding_text"] = display_text
    updated["citation_text"] = citation_text
    meta["citation_text"] = citation_text
    meta["llm_typo_corrected"] = True
    updated["metadata"] = meta
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct OCR typos in Chroma-ready chunks using a small LLM.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--provider", choices=["gemini", "ollama", "openai-compatible", "transformers"], default="transformers")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load local Transformers model in 4-bit to save VRAM.")
    parser.add_argument("--no-auto-4bit-7b", dest="auto_4bit_7b", action="store_false", help="Disable automatic 4-bit loading for 7B models.")
    parser.set_defaults(auto_4bit_7b=True)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn-implementation", default="sdpa", help="Use sdpa or flash_attention_2 when available.")
    parser.add_argument("--max-new-tokens", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--only-needs-review", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Allow output path to equal input path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input == args.output and not args.overwrite:
        raise SystemExit("Output sama dengan input. Pakai --overwrite kalau memang mau menimpa.")
    if args.max_new_tokens >= 100000:
        print("Warning: max_new_tokens=100000 belum tentu didukung model/provider. Script tetap meneruskan nilai ini.")

    chunks = json.loads(args.input.read_text(encoding="utf-8"))
    cache = load_cache(args.cache)
    transformers_caller = TransformersCaller(args) if args.provider == "transformers" else None

    output: List[Dict[str, Any]] = []
    processed = 0
    corrected = 0
    failed = 0

    for idx, chunk in enumerate(tqdm(chunks, desc="LLM typo correction")):
        if idx < args.start:
            output.append(chunk)
            continue
        if args.limit is not None and processed >= args.limit:
            output.append(chunk)
            continue

        if not should_process(chunk, args):
            output.append(chunk)
            continue

        processed += 1
        key = cache_key_for(chunk, args.model, args.provider)
        try:
            if key in cache:
                corrected_text = cache[key]
            else:
                corrected_text = correct_text(args, prompt_for_chunk(chunk), transformers_caller)
                corrected_text = clean_model_output(corrected_text)
                append_cache(args.cache, key, chunk.get("id", ""), corrected_text)
            output.append(update_chunk(chunk, corrected_text))
            corrected += 1
        except Exception as exc:
            failed += 1
            fallback = dict(chunk)
            meta = dict(fallback.get("metadata", {}))
            meta["llm_typo_corrected"] = False
            meta["llm_typo_error"] = str(exc)[:500]
            fallback["metadata"] = meta
            output.append(fallback)
            print(f"\nFailed chunk {idx} {chunk.get('id')}: {exc}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Input chunks: {len(chunks)}")
    print(f"Processed: {processed}")
    print(f"Corrected: {corrected}")
    print(f"Failed: {failed}")
    print(f"Output: {args.output}")
    print(f"Cache: {args.cache}")


if __name__ == "__main__":
    main()
