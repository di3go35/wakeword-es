#!/usr/bin/env python3
"""Genera positivos con Chatterbox (motor distinto de Kokoro) para medir recall.

Por qué: los positivos de entrenamiento son de Kokoro. Un modelo puede aprender la frase *de ese
motor* y fallar con cualquier otra voz. Este conjunto es el control de disjunción por motor TTS:
si el modelo acierta aquí, generaliza entre sintetizadores; si falla, lo que aprendió es la huella
de Kokoro.

Chatterbox es multilingüe (23 idiomas, español incluido) y su voz por defecto se puede sustituir
por una muestra de referencia (`--referencia`), lo que permite acercarse a una voz humana real.

Uso:
  python3 pipeline/generar_positivos_chatterbox.py --salida ~/datos-cv/positivos/chatterbox \
      --n 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

FRASE = "hey celeste"
VARIANTES = ("hey celeste", "Hey, Celeste.", "hey... celeste", "¡Hey Celeste!", "Hey celeste.")
SR_CHATTERBOX = 24_000
SR_SALIDA = 16_000


def a_16k(audio: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    if SR_CHATTERBOX == SR_SALIDA:
        return audio
    return resample_poly(audio, SR_SALIDA, SR_CHATTERBOX).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--salida", required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--semilla", type=int, default=20261025)
    p.add_argument("--referencia", default=None,
                   help="WAV de referencia para clonar una voz (opcional)")
    p.add_argument("--frase", default=FRASE)
    p.add_argument("--pad-min", type=float, default=2.2)
    p.add_argument("--pad-max", type=float, default=3.2)
    args = p.parse_args(argv)

    import torch

    torch.set_num_threads(4)
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    manifiesto = salida / "manifiesto.jsonl"
    rng = random.Random(args.semilla)

    print("cargando Chatterbox multilingüe en CPU (la primera vez descarga el modelo)...")
    modelo = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    parametros_voz = {"audio_prompt_path": args.referencia} if args.referencia else {}

    generados = 0
    with manifiesto.open("a", encoding="utf-8") as mf:
        for i in range(args.n):
            nombre = f"cb_{i:04d}.wav"
            destino = salida / nombre
            if destino.exists():
                continue
            variante = args.frase if i == 0 else rng.choice(VARIANTES)
            exageracion = round(rng.uniform(0.25, 0.85), 2)
            cfg = round(rng.uniform(0.25, 0.7), 2)
            temperatura = round(rng.uniform(0.55, 0.9), 2)

            wav = modelo.generate(
                variante, language_id="es", exaggeration=exageracion,
                cfg_weight=cfg, temperature=temperatura, **parametros_voz,
            )
            audio = wav.squeeze().numpy().astype(np.float32)
            pico = float(np.max(np.abs(audio))) or 1.0
            audio = (audio / pico * 0.6).astype(np.float32)
            audio = a_16k(audio)

            objetivo = rng.uniform(args.pad_min, args.pad_max)
            falta = max(0.0, objetivo - len(audio) / SR_SALIDA)
            adelante = rng.uniform(0.15, 0.55) * falta
            audio = np.concatenate([
                np.zeros(int(adelante * SR_SALIDA), dtype=np.float32), audio,
                np.zeros(int((falta - adelante) * SR_SALIDA), dtype=np.float32),
            ])

            sf.write(destino, audio, SR_SALIDA, subtype="PCM_16")
            mf.write(json.dumps({
                "archivo": nombre, "motor": "chatterbox", "variante": variante,
                "exageracion": exageracion, "cfg": cfg, "temperatura": temperatura,
                "referencia": args.referencia,
                "segundos": round(len(audio) / SR_SALIDA, 3),
                "sha256": hashlib.sha256(destino.read_bytes()).hexdigest(),
            }, ensure_ascii=False) + "\n")
            mf.flush()
            generados += 1
            print(f"  {generados}/{args.n} ({round(len(audio) / SR_SALIDA, 2)} s)", flush=True)

    print(f"LISTO: {generados} clips nuevos en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
