#!/usr/bin/env python3
"""Minado de negativos difíciles: los trozos de habla real que el modelo ya confunde con la frase.

Por qué existe: el FA/h bajo en promedio puede esconder un puñado de ventanas que se disparan con
fuerza. Reentrenar con ellas como negativas es el paso que en la literatura llevó 91/140 a 1/140
falsos positivos por activación.

Regla de oro: se mina **solo con hablantes de entrenamiento**. Si se minara con los de validación o
test, el modelo aprendería a no fallar exactamente donde se le mide.

Salida: `neg_dificiles.npy` (float16) + `neg_dificiles.jsonl` con clip, hablante y puntuación.

Uso:
  python3 pipeline/minar_negativos.py --modelo ~/datos-cv/entrenamiento/cabeza.onnx \
      --features ~/datos-cv/features --tsv ~/datos-cv/validado_con_duraciones.tsv \
      --split ~/datos-cv/split.json --salida ~/datos-cv/entrenamiento --n 8000 --umbral 0.10
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort

VENTANA = 16
DIM = 96


def hablantes_permitidos(tsv: Path, split: Path, excluir_pct: float, semilla: int) -> set[str]:
    """Hablantes de train menos los de validación (10% apartado al preparar el conjunto)."""
    datos = json.loads(split.read_text(encoding="utf-8"))
    train = sorted(set(datos["train"]))
    rng = random.Random(semilla)
    rng.shuffle(train)
    n_val = max(1, round(len(train) * excluir_pct / 100))
    return set(train[n_val:])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modelo", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--tsv", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--umbral", type=float, default=0.10)
    p.add_argument("--max-clips", type=int, default=60_000)
    p.add_argument("--validacion-pct", type=float, default=10.0)
    p.add_argument("--semilla", type=int, default=20261024)
    args = p.parse_args(argv)

    permitidos = hablantes_permitidos(Path(args.split), Path(args.split),
                                      args.validacion_pct, args.semilla)
    with Path(args.tsv).open(encoding="utf-8", newline="") as f:
        hablante_de = {(fila.get("path") or "").strip(): (fila.get("client_id") or "").strip()
                       for fila in csv.DictReader(f, delimiter="\t")}

    features = Path(args.features)
    candidatos: list[tuple[Path, str]] = []
    for mani in sorted(features.glob("manifiesto_*.jsonl")):
        for linea in mani.open(encoding="utf-8"):
            if not linea.strip():
                continue
            fila = json.loads(linea)
            if "archivo" not in fila:
                continue
            clip = Path(fila.get("clip") or fila.get("ruta") or "").name
            if hablante_de.get(clip) in permitidos:
                candidatos.append((features / fila["archivo"], clip))

    rng = random.Random(args.semilla)
    rng.shuffle(candidatos)
    candidatos = candidatos[:args.max_clips]
    print(f"clips de hablantes de entrenamiento a revisar: {len(candidatos)}")

    sesion = ort.InferenceSession(args.modelo, providers=["CPUExecutionProvider"])
    acumulado: list[tuple[float, np.ndarray, str, int]] = []
    procesados = 0
    for npy, clip in candidatos:
        try:
            datos = np.load(npy)
        except Exception:
            continue
        if datos.shape[0] < VENTANA or datos.shape[1] != DIM:
            continue
        ventanas = np.lib.stride_tricks.sliding_window_view(datos, (VENTANA, DIM))[:, 0]
        salida = sesion.run(None, {"input": np.ascontiguousarray(ventanas, dtype=np.float32)})[0]
        prob = 1 / (1 + np.exp(-salida.ravel()))
        for i, pr in enumerate(prob):
            if pr > args.umbral:
                acumulado.append((float(pr), ventanas[i].astype(np.float16), clip, i))
        procesados += 1
        if procesados % 5000 == 0:
            print(f"  {procesados} clips | candidatas acumuladas: {len(acumulado)}", flush=True)

    if not acumulado:
        print("NINGUNA ventana superó el umbral: el modelo no tiene negativos difíciles que minar")
        return 0

    acumulado.sort(key=lambda x: -x[0])
    seleccion = acumulado[:args.n]
    salida = Path(args.salida)
    np.save(salida / "neg_dificiles.npy", np.stack([s[1] for s in seleccion]))
    with (salida / "neg_dificiles.jsonl").open("w", encoding="utf-8") as f:
        for pr, _, clip, i in seleccion:
            f.write(json.dumps({"clip": clip, "ventana": i, "puntuacion": round(pr, 4)}) + "\n")

    print(f"ventanas sobre {args.umbral}: {len(acumulado)} | guardadas las {len(seleccion)} peores")
    print(f"puntuación de la peor guardada: {seleccion[-1][0]:.4f}")
    print(f"escrito en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
