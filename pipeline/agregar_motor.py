#!/usr/bin/env python3
"""Añade un segundo (o tercer) motor TTS a los positivos de entrenamiento.

Motivo medido: una cabeza entrenada solo con Kokoro da recall 1,000 con las tres voces de Kokoro
—incluida una nunca vista— y **0,067 con Chatterbox** (434 clips). Generaliza a voces, no a motores.
Para aprender la frase y no el sintetizador, los positivos tienen que venir de varios motores.

Reglas que se mantienen del pipeline:
  - Una ventana es positiva solo si contiene la frase completa (si no, enseña media frase).
  - El reparto train/validación es por clip, nunca por ventana: ventanas del mismo clip en los dos
    lados inflarían la validación.

Uso:
  python3 pipeline/agregar_motor.py --features ~/datos-cv/features_cb_kaggle \
      --manifiesto ~/datos-cv/positivos/chatterbox_kaggle/manifiesto.jsonl \
      --audio ~/datos-cv/positivos/chatterbox_kaggle \
      --salida ~/datos-cv/entrenamiento --prefijo cb --val-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

VENTANA = 16
DIM = 96
HOP_S = 0.08
CONTEXTO_S = 2.04


def extensión_frase(wav: Path) -> tuple[float, float] | None:
    import soundfile as sf

    audio, sr = sf.read(wav, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    activo = np.flatnonzero(np.abs(audio) > 1e-4)
    if activo.size == 0:
        return None
    return float(activo[0]) / sr, float(activo[-1] + 1) / sr


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--manifiesto", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--prefijo", required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--val-list", default=None,
                   help="archivo con los clips que van a validación (para respetar un split ya hecho)")
    p.add_argument("--voz-val", default=None,
                   help="sufijo de voz que va a validación (p. ej. em_santa); el resto a train")
    p.add_argument("--semilla", type=int, default=20261025)
    args = p.parse_args(argv)

    rng = random.Random(args.semilla)
    features, audio_dir = Path(args.features), Path(args.audio)
    filas = [json.loads(l) for l in Path(args.manifiesto).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if args.val_list:
        # Se compara por el clip de origen: los aumentados heredan el split de su original.
        permitidos = set(Path(args.val_list).read_text().split())
        en_val = {f["archivo"] for f in filas
                  if Path(f["archivo"]).stem.split("_aug")[0] + ".wav" in permitidos}
        n_val = len(en_val)
    elif args.voz_val:
        en_val = {f["archivo"] for f in filas if args.voz_val in f["archivo"]}
        n_val = len(en_val)
    else:
        rng.shuffle(filas)
        n_val = max(1, int(len(filas) * args.val_frac))
        en_val = set(f["archivo"] for f in filas[:n_val])

    train, val, sin_span = [], [], []
    for fila in filas:
        npy = features / (Path(fila["archivo"]).stem + ".npy")
        if not npy.exists():
            continue
        datos = np.load(npy, mmap_mode="r")
        if datos.shape[0] < VENTANA or datos.shape[1] != DIM:
            continue
        span = extensión_frase(audio_dir / fila["archivo"])
        if span is None:
            sin_span.append(fila["archivo"])
            continue
        t0, t1 = span
        destino = val if fila["archivo"] in en_val else train
        for k in range(datos.shape[0] - VENTANA + 1):
            if HOP_S * k <= t0 and CONTEXTO_S + HOP_S * k >= t1:
                destino.append(np.array(datos[k:k + VENTANA], dtype=np.float16))

    salida = Path(args.salida)
    for nombre, datos in ((f"{args.prefijo}_pos_train", train), (f"{args.prefijo}_pos_val", val)):
        if not datos:
            raise SystemExit(f"sin ventanas para {nombre}")
        ruta = salida / f"{nombre}.npy"
        np.save(ruta, np.stack(datos))
        print(f"{nombre}: {len(datos)} ventanas -> {ruta.name}")

    if not args.val_list and not args.voz_val:
        (salida / f"{args.prefijo}_val.txt").write_text("\n".join(sorted(en_val)) + "\n",
                                                        encoding="utf-8")
    print(f"clips: {len(filas)} | a validación: {n_val} (lista en {args.prefijo}_val.txt) | "
          f"sin señal: {len(sin_span)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
