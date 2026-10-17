#!/usr/bin/env python3
"""Genera un JSONL de puntajes SINTÉTICO para autoprobar el banco de evaluación.

Esto no es un resultado: es un fixture determinista para verificar que las métricas
(FA/h, recall, criterio, silencio) hacen lo que dicen. Cualquier número que salga de
aquí y no esté etiquetado como autotest es un error de proceso.

  --perfil bueno  genera un fixture que debe CUMPLIR el criterio
  --perfil malo   genera un fixture que debe FALLAR (demuestra que el criterio
                  es ejecutable y no decorativo)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CONDICIONES = ("limpio", "tv", "lejano")
PASO_S = 0.08          # resolución de frame de un detector real (80 ms)
PASO_NEG_S = 1.0       # resolución reducida solo para mantener el fixture liviano


def frames_positivo(rng: random.Random, detecta: bool) -> list[list[float]]:
    """Un positivo: la frase dura ~0.4 s; el score sube a un plateau alto y decae."""
    inicio = 1.0 + rng.random() * 0.2
    frames = []
    t = 0.0
    while t < 3.0:
        score = 0.005 + rng.random() * 0.025
        if detecta and inicio <= t <= inicio + 0.4:
            score = 0.97 + rng.random() * 0.02
        elif detecta and inicio - 0.16 <= t < inicio:
            score = 0.40 + rng.random() * 0.20
        frames.append([round(t, 3), round(min(score, 0.999), 3)])
        t += PASO_S
    return frames


def frames_negativo(rng: random.Random, dur_s: float, picos: int) -> list[list[float]]:
    """Audio sin la frase: fondo bajo con algunos picos que se van a contar como FA."""
    frames = []
    t = 0.0
    en_pico = 0.0
    while t < dur_s:
        score = 0.005 + rng.random() * 0.025
        if en_pico > 0.0:
            score = 0.50 + rng.random() * 0.30
            en_pico -= PASO_NEG_S
        elif picos > 0 and rng.random() < picos / (dur_s / PASO_NEG_S):
            en_pico = 0.5
            score = 0.50 + rng.random() * 0.30
            picos -= 1
        frames.append([round(t, 3), round(min(score, 0.999), 3)])
        t += PASO_NEG_S
    return frames


def generar(perfil: str, semilla: int = 7) -> list[dict]:
    rng = random.Random(semilla)
    if perfil == "bueno":
        # 20 clips por condición; el criterio exige >=95% en cada una.
        fallos_por_condicion = {"limpio": 0, "tv": 1, "lejano": 1}
        duracion_neg = 600.0
        picos_neg = 1
    elif perfil == "malo":
        fallos_por_condicion = {"limpio": 2, "tv": 6, "lejano": 9}
        duracion_neg = 600.0
        picos_neg = 6
    else:
        raise SystemExit(f"perfil desconocido: {perfil}")

    registros = []
    for condicion, fallos in fallos_por_condicion.items():
        for i in range(20):
            detecta = i >= fallos
            registros.append({
                "id": f"pos_{condicion}_{i:02d}",
                "tipo": "positivo",
                "condicion": condicion,
                "hablante": f"h{(i % 7) + 1:02d}",
                "motor_tts": "chatterbox" if i % 2 == 0 else "kokoro",
                "frames": frames_positivo(rng, detecta),
            })

    for i in range(3):
        registros.append({
            "id": f"neg_cv_{i:02d}",
            "tipo": "negativo",
            "condicion": "-",
            "fuente": "autotest_fixture",
            "dur_s": duracion_neg,
            "frames": frames_negativo(rng, duracion_neg, picos_neg),
        })

    registros.append({
        "id": "sil_00",
        "tipo": "silencio",
        "condicion": "silencio_digital",
        "dur_s": 1800.0,
        "frames": [[round(t, 3), 0.0] for t in
                   (i * PASO_NEG_S for i in range(int(1800.0 / PASO_NEG_S)))],
    })
    return registros


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--perfil", choices=("bueno", "malo"), required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--semilla", type=int, default=7)
    args = p.parse_args()

    registros = generar(args.perfil, args.semilla)
    with Path(args.salida).open("w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"autotest '{args.perfil}': {len(registros)} registros -> {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
