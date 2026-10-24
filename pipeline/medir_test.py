#!/usr/bin/env python3
"""Mide activaciones por hora sobre el test congelado de habla real.

Este es el número honesto del proyecto: **cuántas veces se dispara el modelo sobre 53 h de habla
real de 3.247 hablantes que no vio nunca**, por condición de acento. La AUC de validación no sirve
para esto: con positivos sintéticos y negativos humanos, un AUC de 1,0 puede significar solo que el
modelo distingue "voz de TTS" de "voz humana".

Protocolo:
  - Ventana de 16 frames, solapada (una puntuación por frame, como en streaming).
  - Capa anti-disparo mínima: N frames consecutivos sobre el umbral y tiempo de enfriamiento.
  - Un evento por activación; FA/h = eventos / horas de la condición.

Uso:
  python3 pipeline/medir_test.py --modelo ~/datos-cv/entrenamiento/cabeza.onnx \
      --features ~/datos-cv/features --tsv ~/datos-cv/validado_con_duraciones.tsv \
      --clips ~/datos-cv/test/clips.txt --umbral 0.5 --frames-consecutivos 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

VENTANA = 16
DIM = 96


def condicion_por_clip(tsv: Path, permitidos: set[str]) -> tuple[dict[str, str], dict[str, float]]:
    """clip -> condición de acento, y horas por condición (solo de los clips del test)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import planificar_lotes as pl

    condicion: dict[str, str] = {}
    horas: dict[str, float] = defaultdict(float)
    with tsv.open(encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f, delimiter="\t"):
            nombre = (fila.get("path") or "").strip()
            if nombre not in permitidos:
                continue
            grupo = pl.etiqueta_canonica(fila.get("accents"))
            condicion[nombre] = grupo
            horas[grupo] += float(fila.get("duration") or 0.0) / 3600
    return condicion, horas


def puntuar_clip(sesion, datos: np.ndarray, umbral: float, consecutivos: int,
                 enfriamiento_s: float) -> tuple[int, float]:
    """Devuelve (eventos, puntuación máxima) de un clip."""
    n = datos.shape[0]
    if n < VENTANA:
        return 0, 0.0
    ventanas = np.lib.stride_tricks.sliding_window_view(datos, (VENTANA, DIM))[:, 0]
    puntajes = np.zeros(n, dtype=np.float32)
    lote = 4096
    for i in range(0, len(ventanas), lote):
        trozo = np.ascontiguousarray(ventanas[i:i + lote], dtype=np.float32)
        salida = sesion.run(None, {"input": trozo})[0].ravel()
        prob = 1 / (1 + np.exp(-salida))
        puntajes[i + VENTANA - 1:i + VENTANA - 1 + len(prob)] = prob

    eventos, racha, ultimo_frame = 0, 0, -10**9
    frames_enfriamiento = max(1, int(enfriamiento_s * 12.4))
    for i, p in enumerate(puntajes):
        if p > umbral:
            racha += 1
            if racha >= consecutivos and (i - ultimo_frame) > frames_enfriamiento:
                eventos += 1
                ultimo_frame = i
                racha = 0
        else:
            racha = 0
    return eventos, float(puntajes.max())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modelo", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--tsv", required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--umbral", type=float, default=0.5)
    p.add_argument("--frames-consecutivos", type=int, default=3)
    p.add_argument("--enfriamiento", type=float, default=1.0)
    p.add_argument("--salida", default=None)
    args = p.parse_args(argv)

    permitidos = {l.strip() for l in Path(args.clips).read_text().splitlines() if l.strip()}
    condicion, horas = condicion_por_clip(Path(args.tsv), permitidos)
    print(f"clips del test: {len(permitidos)} | con condición asignada: {len(condicion)}")
    print(f"horas totales: {sum(horas.values()):.2f}")

    sesion = ort.InferenceSession(args.modelo, providers=["CPUExecutionProvider"])
    features = Path(args.features)

    eventos: dict[str, int] = defaultdict(int)
    clips_por_cond: dict[str, int] = defaultdict(int)
    max_global = 0.0
    faltantes = 0
    for i, clip in enumerate(sorted(permitidos), 1):
        npy = features / (Path(clip).stem + ".npy")
        if not npy.exists():
            faltantes += 1
            continue
        datos = np.load(npy)
        if datos.shape[1] != DIM:
            faltantes += 1
            continue
        ev, maximo = puntuar_clip(sesion, datos, args.umbral, args.frames_consecutivos,
                                  args.enfriamiento)
        grupo = condicion.get(clip, "sin condición")
        eventos[grupo] += ev
        clips_por_cond[grupo] += 1
        max_global = max(max_global, maximo)
        if i % 5000 == 0:
            print(f"  {i} clips procesados...", flush=True)

    total_eventos = sum(eventos.values())
    total_horas = sum(horas.get(g, 0.0) for g in eventos)
    print(f"\n{'condición':<45} {'clips':>7} {'horas':>7} {'eventos':>8} {'FA/h':>8}")
    for grupo in sorted(eventos, key=lambda g: -horas.get(g, 0.0)):
        h = horas.get(grupo, 0.0)
        print(f"{grupo[:44]:<45} {clips_por_cond[grupo]:>7} {h:>7.2f} {eventos[grupo]:>8} "
              f"{(eventos[grupo] / h if h else float('nan')):>8.3f}")
    print(f"\nTOTAL: {total_eventos} eventos en {total_horas:.2f} h = "
          f"{total_eventos / total_horas:.3f} FA/h" if total_horas else "")
    print(f"clips sin características: {faltantes} | puntuación máxima vista: {max_global:.4f}")

    if args.salida:
        Path(args.salida).write_text(json.dumps({
            "modelo": args.modelo, "umbral": args.umbral,
            "frames_consecutivos": args.frames_consecutivos,
            "enfriamiento_s": args.enfriamiento,
            "total_eventos": total_eventos, "total_horas": round(total_horas, 3),
            "fa_por_hora": round(total_eventos / total_horas, 4) if total_horas else None,
            "por_condicion": {
                g: {"clips": clips_por_cond[g], "horas": round(horas.get(g, 0.0), 3),
                    "eventos": eventos[g],
                    "fa_por_hora": round(eventos[g] / horas[g], 4) if horas.get(g) else None}
                for g in eventos
            },
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"escrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
