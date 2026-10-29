#!/usr/bin/env python3
"""Mide el efecto de la compuerta de energía sobre falsos positivos y sobre recall lejano.

Una sola pasada por clip: se puntúa una vez y se evalúa con varios pisos de energía, para poder
elegir el piso con las dos cifras a la vista (lo que quita y lo que cuesta) en lugar de a ojo.

Uso:
  python3 pipeline/medir_con_compuerta.py --modelo ~/datos-cv/entrenamiento/cabeza_fewshot.onnx \
      --features ~/datos-cv/features --tsv ~/datos-cv/validado_con_duraciones.tsv \
      --clips ~/datos-cv/test/clips.txt --audio-dir ~/datos-cv/test/clips --umbral 0.9
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import antidisparo  # noqa: E402

VENTANA = 16
DIM = 96


def puntuar_clip(sesion, datos: np.ndarray, lote: int = 2048) -> np.ndarray:
    n = datos.shape[0]
    puntajes = np.zeros(n, dtype=np.float32)
    if n < VENTANA:
        return puntajes
    ventanas = np.lib.stride_tricks.sliding_window_view(datos, (VENTANA, DIM))[:, 0]
    for i in range(0, len(ventanas), lote):
        trozo = np.ascontiguousarray(ventanas[i:i + lote], dtype=np.float32)
        salida = sesion.run(None, {"input": trozo})[0].ravel()
        p = 1 / (1 + np.exp(-salida))
        puntajes[i + VENTANA - 1:i + VENTANA - 1 + len(p)] = p
    return puntajes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modelo", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--tsv", default=None, help="para agrupar por condición de acento")
    p.add_argument("--clips", required=True)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--umbral", type=float, default=0.9)
    p.add_argument("--consecutivos", type=int, default=3)
    p.add_argument("--enfriamiento", type=float, default=1.0)
    p.add_argument("--pisos", default="-60,-50,-45,-40,-35",
                   help="pisos de energía en dBFS a comparar")
    args = p.parse_args(argv)

    pisos = [float(x) for x in args.pisos.split(",")]
    sesion = ort.InferenceSession(args.modelo, providers=["CPUExecutionProvider"])
    features, audio_dir = Path(args.features), Path(args.audio_dir)
    clips = [l.strip() for l in Path(args.clips).read_text().splitlines() if l.strip()]

    condicion, horas = {}, defaultdict(float)
    if args.tsv:
        with Path(args.tsv).open(encoding="utf-8", newline="") as f:
            import planificar_lotes as pl
            for fila in csv.DictReader(f, delimiter="\t"):
                nombre = (fila.get("path") or "").strip()
                if nombre in set(clips):
                    grupo = pl.etiqueta_canonica(fila.get("accents"))
                    condicion[nombre] = grupo
                    horas[grupo] += float(fila.get("duration") or 0.0) / 3600

    eventos = {piso: defaultdict(int) for piso in pisos}
    eventos["sin"] = defaultdict(int)
    energias_disparo: list[float] = []
    sin_audio = 0

    for i, clip in enumerate(clips, 1):
        npy = features / (Path(clip).stem + ".npy")
        wav = audio_dir / clip
        if not npy.exists() or not wav.exists():
            sin_audio += 1
            continue
        datos = np.load(npy)
        if datos.shape[1] != DIM:
            sin_audio += 1
            continue
        audio, sr = sf.read(wav, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != antidisparo.SR:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, antidisparo.SR, sr).astype(np.float32)

        puntajes = puntuar_clip(sesion, datos)
        grupo = condicion.get(clip, "sin condición")

        eventos["sin"][grupo] += antidisparo.contar_eventos(
            puntajes, args.umbral, args.consecutivos, args.enfriamiento)
        for piso in pisos:
            gateado = antidisparo.aplicar_compuerta(
                puntajes, antidisparo.energia_por_frame(audio), piso)
            ev = antidisparo.contar_eventos(gateado, args.umbral, args.consecutivos,
                                            args.enfriamiento)
            eventos[piso][grupo] += ev
        # energía de los frames donde el modelo sin compuerta dispararía
        if (puntajes > args.umbral).any():
            energias = antidisparo.energia_por_frame(audio)
            idx = np.flatnonzero(puntajes > args.umbral)
            idx = idx[idx < len(energias)]
            energias_disparo.extend(energias[idx].tolist())
        if i % 5000 == 0:
            print(f"  {i} clips...", flush=True)

    total_h = sum(horas.values()) if horas else float(len(clips))
    print(f"\n{'piso':>6} {'eventos':>9} {'FA/h':>8}")
    for clave in ["sin"] + pisos:
        total = sum(eventos[clave].values())
        hora = f"{clave:>6}"
        etiqueta = "sin" if clave == "sin" else f"{clave:.0f}"
        print(f"{etiqueta:>6} {total:>9} {total / total_h:>8.3f}" if total_h else "")
    if energias_disparo:
        e = np.array(energias_disparo)
        print(f"\nenergía de los frames que disparan (sin compuerta): "
              f"p5 {np.percentile(e, 5):.1f} · mediana {np.median(e):.1f} · "
              f"p95 {np.percentile(e, 95):.1f} dBFS")
    print(f"clips sin audio o sin características: {sin_audio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
