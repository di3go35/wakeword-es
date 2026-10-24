#!/usr/bin/env python3
"""Calcula características (mel + embedding de openWakeWord) de un lote de clips.

Se ejecuta con el intérprete del pipeline (Python 3.11 + openwakeword 0.6.0 + onnxruntime):

  ~/.hermes/tools/venv-wakeword/.venv311/bin/python pipeline/extraer_caracteristicas.py \
      --lista ~/datos-cv/lotes/lote_000.txt --audio-dir ~/datos-cv/clips \
      --salida ~/datos-cv/features --lote-id 000

Comportamiento:
  - Decodifica cada clip a PCM 16 kHz mono con ffmpeg (los MP3 del corpus no son WAV).
  - Reanudable: salta los clips cuyas características ya existen en `--salida`.
  - Deja un manifiesto JSONL (`<salida>/manifiesto_<lote>.jsonl`) con una fila por clip:
    ruta, segundos, forma del vector y hash del audio decodificado.
  - No borra el audio: de eso se encarga quien llama, cuando el lote está verificado.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

SR = 16000

# Un clip más corto que esto no produce una ventana de embedding válida: el modelo mel
# espera un tensor 4-D y con audios de pocas decenas de milisegundos la entrada sale con
# rango 2 y ONNX aborta ("Invalid rank for input"). Esos clips se descartan con motivo
# declarado en vez de contarse como error del pipeline. Medido en el lote 001: 5 de 179.924.
MIN_SEGUNDOS = 0.30
DIM_EMBEDDING = 96


def buscar_ffmpeg(preferido: str = "ffmpeg") -> str:
    """ffmpeg del sistema o el que vive en el toolchain local."""
    from shutil import which

    if which(preferido):
        return preferido
    for cand in sorted(Path.home().glob(".hermes/tools/ffmpeg-*/**/ffmpeg")):
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "no encuentro ffmpeg: ni en PATH ni en ~/.hermes/tools/ffmpeg-*"
    )


def decodificar(ffmpeg: str, ruta: Path) -> bytes:
    """MP3/WAV → PCM 16 kHz mono 16 bits."""
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(ruta), "-ac", "1", "-ar", str(SR),
         "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló en {ruta.name}: {proc.stderr.decode()[:200]}")
    return proc.stdout


def procesar_lote(
    lista: Path,
    audio_dir: Path,
    salida: Path,
    lote_id: str,
    max_clips: int | None = None,
    verbose: bool = True,
) -> dict:
    import numpy as np
    from openwakeword.utils import AudioFeatures

    salida.mkdir(parents=True, exist_ok=True)
    manifiesto = salida / f"manifiesto_{lote_id}.jsonl"
    hechos = {json.loads(l)["clip"] for l in manifiesto.read_text().splitlines()} \
        if manifiesto.exists() else set()

    ffmpeg = buscar_ffmpeg()
    af = AudioFeatures(inference_framework="onnx")
    extraer = getattr(af, "_get_embeddings", None) or getattr(af, "get_features")

    pendientes = [
        n.strip() for n in lista.read_text(encoding="utf-8").splitlines() if n.strip()
    ]
    if max_clips is not None:
        pendientes = pendientes[:max_clips]

    hechos_ahora = saltados = errores = descartados = 0
    segundos_totales = 0.0
    t0 = time.time()
    descartes = salida / f"descartes_{lote_id}.jsonl"
    for nombre in pendientes:
        if nombre in hechos:
            saltados += 1
            continue
        origen = audio_dir / nombre
        destino = salida / f"{Path(nombre).stem}.npy"
        try:
            pcm = decodificar(ffmpeg, origen)
            audio = np.frombuffer(pcm, dtype=np.int16)
            segundos = len(audio) / SR
            if segundos < MIN_SEGUNDOS:
                with descartes.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"clip": nombre,
                                        "motivo": f"audio demasiado corto ({segundos:.3f} s)"},
                                       ensure_ascii=False) + "\n")
                descartados += 1
                continue
            feats = np.asarray(extraer(audio), dtype=np.float32)
            if feats.ndim != 2 or feats.shape[1] != DIM_EMBEDDING or feats.shape[0] == 0:
                with descartes.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"clip": nombre,
                                        "motivo": f"forma inesperada {list(feats.shape)}"},
                                       ensure_ascii=False) + "\n")
                descartados += 1
                continue
            np.save(destino, feats)
            with manifiesto.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "clip": nombre,
                    "segundos": round(segundos, 3),
                    "forma": list(feats.shape),
                    "sha256_audio": hashlib.sha256(pcm).hexdigest(),
                    "archivo": destino.name,
                }, ensure_ascii=False) + "\n")
            hechos_ahora += 1
            segundos_totales += segundos
            if verbose and hechos_ahora % 25 == 0:
                ritmo = segundos_totales / (time.time() - t0)
                print(f"  {hechos_ahora} clips | {segundos_totales/3600:.2f} h de audio "
                      f"| {ritmo:.1f}x tiempo real", flush=True)
        except Exception as e:  # un clip corrupto no debe tumbar el lote
            errores += 1
            print(f"  ERROR {nombre}: {type(e).__name__}: {e}", file=sys.stderr)

    dt = time.time() - t0
    return {
        "lote": lote_id,
        "clips_en_lista": len(pendientes),
        "procesados": hechos_ahora,
        "saltados_ya_existentes": saltados,
        "descartados": descartados,
        "errores": errores,
        "horas_audio": round(segundos_totales / 3600.0, 4),
        "segundos_cpu": round(dt, 2),
        "factor_tiempo_real": round(segundos_totales / dt, 2) if dt > 0 else None,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lista", required=True, help="lote_NNN.txt con un clip por línea")
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--lote-id", required=True)
    p.add_argument("--max-clips", type=int, default=None,
                   help="limita el número de clips (para pruebas)")
    args = p.parse_args(argv)

    resumen = procesar_lote(
        Path(args.lista), Path(args.audio_dir), Path(args.salida), args.lote_id,
        max_clips=args.max_clips,
    )
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0 if resumen["procesados"] > 0 or resumen["saltados_ya_existentes"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
