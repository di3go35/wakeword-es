#!/usr/bin/env python3
"""Calibra el umbral para el entorno del usuario.

Fase 6.2 del plan. Un umbral único no sirve para todos los entornos: en una oficina ruidosa los
falsos positivos suben y en una casa silenciosa bajar. El calibrador toma unos minutos de audio del
**entorno real donde va a vivir el dispositivo** y devuelve el umbral que cumple un objetivo de falsos
positivos por hora, junto con el margen que queda.

Uso típico (la app graba 10 min sin hablar y llama a esto):

  python3 pipeline/calibrar.py --features ~/datos-cv/features --clips mi_entorno.txt \\
      --horas 0.1667 --objetivo 0.5

El mismo cálculo se hace en el dispositivo con las características ya extraídas: solo necesita la
lista de puntuaciones por frame y la duración.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
import antidisparo  # noqa: E402

VENTANA = 16
DIM = 96


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modelo", default=str(Path.home() / "datos-cv/entrenamiento/cabeza_fewshot.onnx"))
    p.add_argument("--features", required=True)
    p.add_argument("--clips", required=True, help="lista de clips del entorno, sin hablar la frase")
    p.add_argument("--horas", type=float, required=True, help="duración total de esos clips")
    p.add_argument("--objetivo", type=float, default=0.5, help="FA/h máximo aceptable")
    p.add_argument("--consecutivos", type=int, default=3)
    p.add_argument("--salida", default=None)
    args = p.parse_args(argv)

    sesion = ort.InferenceSession(args.modelo, providers=["CPUExecutionProvider"])
    features = Path(args.features)
    puntajes: list[np.ndarray] = []
    for clip in Path(args.clips).read_text().splitlines():
        if not clip.strip():
            continue
        npy = features / (Path(clip).stem + ".npy")
        if not npy.exists():
            continue
        datos = np.load(npy)
        if datos.shape[0] < VENTANA or datos.shape[1] != DIM:
            continue
        w = np.lib.stride_tricks.sliding_window_view(datos, (VENTANA, DIM))[:, 0]
        salida = sesion.run(None, {"input": np.ascontiguousarray(w, dtype=np.float32)})[0].ravel()
        prob = 1 / (1 + np.exp(-salida))
        marco = np.zeros(datos.shape[0], dtype=np.float32)
        marco[VENTANA - 1:VENTANA - 1 + len(prob)] = prob
        puntajes.append(marco)

    if not puntajes:
        raise SystemExit("sin puntuaciones: revisa --features y --clips")

    candidatos = np.round(np.arange(0.30, 0.991, 0.01), 2)
    filas = []
    for u in candidatos:
        eventos = sum(antidisparo.contar_eventos(s, float(u), args.consecutivos) for s in puntajes)
        filas.append((float(u), eventos, eventos / args.horas))

    aptos = [f for f in filas if f[2] <= args.objetivo]
    print(f"{'umbral':>7} {'eventos':>8} {'FA/h':>8}")
    for u, ev, fa in filas[::5]:
        print(f"{u:>7.2f} {ev:>8} {fa:>8.3f}")

    if aptos:
        elegido = min(aptos, key=lambda f: f[0])   # el más bajo que cumple: máximo recall
        print(f"\numbral recomendado: {elegido[0]:.2f} "
              f"({elegido[1]} eventos en {args.horas:.2f} h = {elegido[2]:.3f} FA/h)")
        nota = ("En un entorno más ruidoso el umbral sube; si el recomendado se acerca a 0,95 el "
                "entorno no es apto para este modelo con el recall actual.")
    else:
        elegido = max(filas, key=lambda f: f[0])
        print(f"\nNINGÚN umbral cumple {args.objetivo} FA/h en este entorno "
              f"(el mejor es {elegido[0]:.2f} con {elegido[2]:.3f} FA/h)")
        nota = "Entorno demasiado ruidoso para el objetivo: hace falta más dato o otro modelo."

    print(nota)
    if args.salida:
        Path(args.salida).write_text(json.dumps({
            "horas": args.horas, "objetivo_fa_h": args.objetivo,
            "umbral_recomendado": elegido[0] if aptos else None,
            "fa_h_en_ese_umbral": round(elegido[2], 4),
            "curva": [{"umbral": u, "eventos": ev, "fa_h": round(fa, 4)} for u, ev, fa in filas],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"escrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
