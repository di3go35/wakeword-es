#!/usr/bin/env python3
"""Genera los positivos sintéticos de la frase congelada con Kokoro-82M (Apache-2.0).

Por qué Kokoro y no otra: es la única TTS grande cuyo audio generado es comercialmente libre.
Su límite es la diversidad (pocas voces españolas), así que la variación se fabrica aquí:
entonación por puntuación, velocidad, tono y ganancia. El test NO se genera con este motor
(disjunción por motor TTS), así que el modelo nunca ve el timbre con el que se le evalúa.

Salida: WAV 16 kHz mono PCM16 + manifiesto JSONL reanudable (si el archivo existe, se salta).

Uso:
  python3 pipeline/generar_positivos.py --salida ~/datos-cv/positivos/kokoro \
      --por-voz 600 --semilla 20261024
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FRASE = "hey celeste"
FRECUENCIA_KOKORO = 24_000
FRECUENCIA_SALIDA = 16_000
VOCES = ("ef_dora", "em_alex", "em_santa")

# La puntuación cambia la prosodia del sintetizador: es variación gratis y realista.
VARIANTES = (
    "hey celeste",
    "Hey, Celeste.",
    "hey... celeste",
    "¡hey celeste!",
    "hey Celeste",
    "Hey celeste.",
)


def a_16k(audio: np.ndarray) -> np.ndarray:
    if FRECUENCIA_KOKORO == FRECUENCIA_SALIDA:
        return audio
    return resample_poly(audio, FRECUENCIA_SALIDA, FRECUENCIA_KOKORO).astype(np.float32)


def perturbar_tono(audio: np.ndarray, factor: float) -> np.ndarray:
    """Remuestreo racional: cambia tono y duración juntos, como la aumentación clásica de KWS."""
    if abs(factor - 1.0) < 1e-3:
        return audio
    from fractions import Fraction

    frac = Fraction(factor).limit_denominator(200)
    return resample_poly(audio, frac.numerator, frac.denominator).astype(np.float32)


def normalizar_pico(audio: np.ndarray, objetivo: float) -> np.ndarray:
    pico = float(np.max(np.abs(audio))) if audio.size else 0.0
    if pico < 1e-6:
        return audio
    return (audio * (objetivo / pico)).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--salida", required=True)
    p.add_argument("--por-voz", type=int, default=600)
    p.add_argument("--semilla", type=int, default=20261024)
    p.add_argument("--frase", default=FRASE)
    args = p.parse_args(argv)

    from kokoro import KPipeline
    from misaki.espeak import EspeakG2P

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    manifiesto = salida / "manifiesto.jsonl"
    rng = random.Random(args.semilla)

    # Kokoro trae espeak-ng empaquetado vía espeakng-loader (no hace falta root), pero su
    # español por defecto es el de ESPAÑA: "celeste" -> ˈθelˈeste, con distinción.
    # Para un wakeword latinoamericano hay que forzar la voz es-419: -> selˈeste (seseo).
    tuberia = KPipeline(lang_code="e")
    tuberia.g2p = EspeakG2P(language="es-419")
    total = 0
    with manifiesto.open("a", encoding="utf-8") as mf:
        for voz in VOCES:
            for i in range(args.por_voz):
                nombre = f"{voz}_{i:05d}.wav"
                destino = salida / nombre
                if destino.exists():
                    continue
                variante = rng.choice(VARIANTES)
                velocidad = round(rng.uniform(0.85, 1.15), 3)
                factor_tono = round(rng.uniform(0.94, 1.06), 4)
                ganancia = round(rng.uniform(0.35, 0.75), 3)

                try:
                    trozos = [a for _, _, a in tuberia(variante, voice=voz, speed=velocidad)]
                except Exception as e:  # voz inexistente en este checkpoint
                    print(f"AVISO: se omite la voz {voz}: {e}", flush=True)
                    break
                audio = np.concatenate([t.numpy() for t in trozos]).astype(np.float32)
                audio = a_16k(perturbar_tono(audio, factor_tono))
                audio = normalizar_pico(audio, ganancia)
                # Silencio corto: el modelo tiene que aprender a activar también con margen.
                audio = np.concatenate([
                    np.zeros(int(0.05 * FRECUENCIA_SALIDA), dtype=np.float32), audio,
                    np.zeros(int(0.10 * FRECUENCIA_SALIDA), dtype=np.float32),
                ])

                sf.write(destino, audio, FRECUENCIA_SALIDA, subtype="PCM_16")
                mf.write(json.dumps({
                    "archivo": nombre,
                    "voz": voz,
                    "variante": variante,
                    "velocidad": velocidad,
                    "factor_tono": factor_tono,
                    "segundos": round(len(audio) / FRECUENCIA_SALIDA, 3),
                    "sha256": hashlib.sha256(destino.read_bytes()).hexdigest(),
                }, ensure_ascii=False) + "\n")
                mf.flush()
                total += 1
                if total % 50 == 0:
                    print(f"{total} clips generados ({voz})", flush=True)

    print(f"LISTO: {total} clips nuevos en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
