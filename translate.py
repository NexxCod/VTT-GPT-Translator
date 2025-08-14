# translate_vtt_gpt5mini.py
# Requisitos: Python 3.9+, pip install openai python-dotenv
# Uso: python translate.py input.vtt [salida.vtt]
# utiliazar .env para OPENAI_API_KEY y OPENAI_MODEL_NAME (opcional, por defecto gpt-5-mini)

import os, re, sys, json, time, random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from openai import OpenAI

# ===================== CONFIG =====================
PRIMARY_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5-mini")
FALLBACK_MODEL = "gpt-5"          # fallback si falla mini
BATCH_SIZE = 36                         # cues por lote (pequeño = más estable)
MAX_CHARS_PER_BATCH = 3500             # caracteres por lote
MAX_COMPLETION_TOKENS = 8000           # salida máxima por llamada
MAX_RETRIES = 3                        # reintentos por lote antes de dividir
BASE_BACKOFF = 1.5                     # segundos de backoff inicial
AUTO_INJECT_WEBVTT = True              # agrega WEBVTT si falta
INTER_BATCH_SLEEP = (0.15, 0.30)         # jitter entre lotes (evita rate limit)
CLIENT_TIMEOUT = 60.0                  # timeout de red por llamada
# ==================================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=CLIENT_TIMEOUT)

VTT_TIMESTAMP = re.compile(r"^\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s-->\s(\d{2}:\d{2}:\d{2}\.\d{3})(.*)$")
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

@dataclass
class Cue:
    idx: int
    header: str
    start: str
    end: str
    settings: str
    text_lines: List[str]

def parse_vtt(path: str) -> Tuple[List[str], List[Cue]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    if not lines or not lines[0].strip().startswith("WEBVTT"):
        if AUTO_INJECT_WEBVTT:
            lines = ["WEBVTT", ""] + lines
        else:
            raise ValueError("Archivo VTT inválido: falta encabezado WEBVTT.")

    header_lines: List[str] = []
    cues: List[Cue] = []
    i, n = 0, len(lines)

    while i < n and not VTT_TIMESTAMP.match(lines[i]):
        header_lines.append(lines[i])
        i += 1

    cue_idx = 0
    while i < n:
        cue_header = ""
        if lines[i] and not VTT_TIMESTAMP.match(lines[i]):
            cue_header = lines[i]
            i += 1

        while i < n and not VTT_TIMESTAMP.match(lines[i]):
            i += 1
        if i >= n:
            break

        m = VTT_TIMESTAMP.match(lines[i])
        if not m:
            i += 1
            continue

        start, end, settings = m.group(1), m.group(2), (m.group(3) or "")
        i += 1

        text_lines = []
        while i < n and lines[i] != "":
            if VTT_TIMESTAMP.match(lines[i]):
                break
            text_lines.append(lines[i])
            i += 1

        if i < n and lines[i] == "":
            i += 1

        cues.append(Cue(cue_idx, cue_header, start, end, settings, text_lines))
        cue_idx += 1

    return header_lines, cues

def build_batches(cues: List[Cue], batch_size: int = BATCH_SIZE, max_chars: int = MAX_CHARS_PER_BATCH):
    batch = []
    count_chars = 0
    for c in cues:
        joined = "\n".join(c.text_lines)
        length = len(joined)
        item = {"i": c.idx, "t": joined}
        # Si el cue está vacío, igual se envía
        if (len(batch) >= batch_size) or (count_chars + length > max_chars):
            if batch:
                yield batch
            batch, count_chars = [], 0
        batch.append(item)
        count_chars += length
    if batch:
        yield batch

SYS_PROMPT = """Eres un traductor EN->ES para subtítulos WebVTT en un contexto médico/educativo.
El contenido puede incluir términos como hemorragia, muerte neuronal o metáforas (p.ej., estrangulación) con fines docentes.
No moderes ni omitas: tu tarea es traducir fielmente.
Responde SIEMPRE con SOLO un objeto JSON válido (sin texto extra, sin explicaciones).
Formato EXACTO: {"segments":[{"i":123,"t":"<texto_traducido>"}]}
Reglas:
- Español neutro y conciso.
- No cambies puntuación innecesariamente.
- Conserva saltos de línea dentro de cada cue.
- Conserva etiquetas inline (<i>, <b>, <u>, <c>, <font>, &nbsp;, etc.).
- NO incluyas timestamps ni IDs.
"""

def extract_json_obj(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Respuesta vacía del modelo.")
    # 1) intento directo
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) fences tipo ```json ... ```
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
    # 3) regex { ... }
    m = JSON_BLOCK_RE.search(text)
    if m:
        candidate = m.group(0).strip()
        return json.loads(candidate)
    raise ValueError("No se pudo extraer un JSON válido de la respuesta del modelo.")

def call_model(model_name: str, prompt: str) -> str:
    """Llama a Chat Completions y devuelve el 'content' (string). Sin temperature, con max_completion_tokens y timeout."""
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content":
                "Contexto: traducción de subtítulos médicos con fines educativos. "
                "No moderar ni omitir; solo traducir.\n"
                + prompt
            },
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        timeout=CLIENT_TIMEOUT,
    )
    choice = resp.choices[0]
    fr = getattr(choice, "finish_reason", None)
    content = choice.message.content or ""
    if not content:
        raise ValueError(f"Respuesta vacía del modelo. finish_reason={fr}")
    return content

def _translate_batch_core(batch: List[Dict], model_name: str) -> Dict[int, str]:
    user_payload = {
        "task": "translate_webvtt_segment_list_en_to_es",
        "notes": "Mantener saltos de línea y etiquetas inline. No inventar contenido.",
        "segments": batch,
    }
    prompt = (
        "Traduce al español los siguientes segmentos de subtítulos. "
        "Devuelve SOLO un JSON válido con el mismo índice 'i' y el texto traducido en 't'.\n\n"
        + json.dumps(user_payload, ensure_ascii=False)
    )
    content = call_model(model_name, prompt)
    data = extract_json_obj(content)
    if "segments" not in data or not isinstance(data["segments"], list):
        raise ValueError("JSON sin 'segments' válido.")
    mapping = {int(seg["i"]): seg["t"] for seg in data["segments"]}
    expected = [x["i"] for x in batch]
    missing = [i for i in expected if i not in mapping]
    if missing:
        raise ValueError(f"Faltan índices en la respuesta del modelo: {missing}")
    return mapping

def translate_batch_resilient(batch: List[Dict], model_name: str = PRIMARY_MODEL) -> Dict[int, str]:
    """Traduce un lote con reintentos, fallback y división adaptativa."""
    # 1) reintentos con el modelo primario
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _translate_batch_core(batch, model_name)
        except Exception as e:
            print(f"[{model_name}] Reintento {attempt}/{MAX_RETRIES} fallido: {e}", file=sys.stderr)
            time.sleep(BASE_BACKOFF * attempt + random.uniform(0, 0.5))

    # 2) fallback a modelo secundario
    if model_name != FALLBACK_MODEL:
        print(f"[{model_name}] Usando fallback → {FALLBACK_MODEL}", file=sys.stderr)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return _translate_batch_core(batch, FALLBACK_MODEL)
            except Exception as e:
                print(f"[{FALLBACK_MODEL}] Reintento {attempt}/{MAX_RETRIES} fallido: {e}", file=sys.stderr)
                time.sleep(BASE_BACKOFF * attempt + random.uniform(0, 0.5))

    # 3) si aún falla y el lote tiene >1 elemento, dividir en 2 y resolver recursivo
    if len(batch) > 1:
        mid = len(batch) // 2
        left = batch[:mid]
        right = batch[mid:]
        print(f"[split] Dividiendo lote en {len(left)} + {len(right)}", file=sys.stderr)
        left_map = translate_batch_resilient(left, model_name=model_name)
        right_map = translate_batch_resilient(right, model_name=model_name)
        left_map.update(right_map)
        return left_map

    # 4) si falla con 1 solo cue, elevar error
    raise RuntimeError("Fallo persistente al traducir el lote, incluso tras fallback y división.")

def write_vtt(path_out: str, header_lines: List[str], cues: List[Cue], translations: Dict[int, str]):
    with open(path_out, "w", encoding="utf-8") as f:
        for hl in header_lines:
            f.write(hl + "\n")
        if header_lines and header_lines[-1] != "":
            f.write("\n")
        for c in cues:
            if c.header:
                f.write(c.header + "\n")
            f.write(f"{c.start} --> {c.end}{c.settings}\n")
            t = translations.get(c.idx, "\n".join(c.text_lines))
            t = (t or "").replace("\r\n", "\n").replace("\r", "\n")
            if t != "":
                f.write(t + "\n")
            f.write("\n")

def main():
    if len(sys.argv) < 2:
        print("Uso: python translate_vtt_gpt5mini.py <archivo.vtt> [salida.vtt]")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) >= 3 else os.path.splitext(src)[0] + ".es.vtt"

    header, cues = parse_vtt(src)
    if not cues:
        print("No se encontraron cues en el archivo.")
        sys.exit(1)

    translations: Dict[int, str] = {}
    total = len(cues)
    done = 0

    batches = list(build_batches(cues, batch_size=BATCH_SIZE, max_chars=MAX_CHARS_PER_BATCH))

    for batch in batches:
        mapping = translate_batch_resilient(batch, model_name=PRIMARY_MODEL)
        translations.update(mapping)
        done += len(batch)
        print(f"Traducidos {done}/{total} cues...", file=sys.stderr)
        time.sleep(random.uniform(*INTER_BATCH_SLEEP))

    write_vtt(dst, header, cues, translations)
    print(f"Listo: {dst}")

if __name__ == "__main__":
    main()
