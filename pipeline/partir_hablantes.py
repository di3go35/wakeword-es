#!/usr/bin/env python3
"""Partición disjunta por hablante del corpus validado, estratificada por acento.

Por qué existe: si un mismo hablante aparece en los negativos de entrenamiento y en el test,
la métrica se infla y deja de decir nada sobre el mundo real. Esta partición se hace **antes
de entrenar** y se congela con su hash, igual que el test.

Regla:
  - Se agrupa por `client_id` (hablante), nunca por clip.
  - Dentro de cada acento canónico se retiene un porcentaje de hablantes para test.
  - La semilla es fija: la partición es reproducible.
  - Se informan las horas resultantes por lado para poder decidir con números.

Uso:
  python3 pipeline/partir_hablantes.py --tsv ~/datos-cv/validado_con_duraciones.tsv \
      --salida ~/datos-cv/split.json --test-pct 15 --semilla 20261018
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import planificar_lotes as pl  # etiqueta_canonica / SIN_DECLARAR


def partir(tsv: Path, test_pct: float, semilla: int, horas_test_min: float = 0.0) -> dict:
    if not 0 < test_pct < 100:
        raise ValueError("test-pct debe estar entre 0 y 100")

    por_acento: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with tsv.open(encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f, delimiter="\t")
        for fila in lector:
            hablante = (fila.get("client_id") or "").strip()
            if not hablante:
                continue
            grupo = pl.etiqueta_canonica(fila.get("accents"))
            segundos = float(fila.get("duration") or 0.0)
            por_acento[grupo][hablante] += segundos

    rng = random.Random(semilla)
    test_hablantes: list[str] = []
    train_hablantes: list[str] = []
    detalle: dict[str, dict] = {}
    for grupo in sorted(por_acento, key=lambda g: -sum(por_acento[g].values())):
        hablantes = sorted(por_acento[grupo])
        rng.shuffle(hablantes)
        n = len(hablantes)
        n_test = max(1, round(n * test_pct / 100))
        # Un acento con pocas horas necesita más hablantes en test para que su FA/h tenga
        # resolución suficiente: con 3 h, un solo falso positivo ya mueve la métrica 0,33/h.
        if horas_test_min > 0:
            acumulado = 0.0
            for k in range(n_test, n + 1):
                acumulado = sum(por_acento[grupo][h] for h in hablantes[:k]) / 3600
                if acumulado >= horas_test_min:
                    n_test = k
                    break
            else:
                n_test = n
        # Nunca más del 40% de los hablantes de un acento al test: si el acento tiene pocas
        # horas, es preferible una condición con poca resolución que quedarse sin negativos
        # de entrenamiento para ese acento.
        n_test = min(n_test, max(1, int(n * 0.4)))
        test, train = hablantes[:n_test], hablantes[n_test:]
        test_hablantes.extend(test)
        train_hablantes.extend(train)
        horas = lambda hs: round(sum(por_acento[grupo][h] for h in hs) / 3600, 2)  # noqa: E731
        detalle[grupo] = {
            "hablantes_train": len(train), "horas_train": horas(train),
            "hablantes_test": len(test), "horas_test": horas(test),
        }

    horas_test = sum(d["horas_test"] for d in detalle.values())
    horas_train = sum(d["horas_train"] for d in detalle.values())
    return {
        "semilla": semilla,
        "test_pct": test_pct,
        "regla": "disjunto por client_id, estratificado por acento canónico, tope del 40% de hablantes por acento en test",
        "horas_minimas_por_condicion": 5.0,
        "hablantes_test": len(test_hablantes),
        "hablantes_train": len(train_hablantes),
        "horas_test": round(horas_test, 2),
        "horas_train": round(horas_train, 2),
        "por_acento": detalle,
        "test": sorted(test_hablantes),
        "train": sorted(train_hablantes),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--test-pct", type=float, default=15.0)
    p.add_argument("--semilla", type=int, default=20261018)
    p.add_argument("--horas-test-min", type=float, default=6.0,
                   help="horas mínimas de test por acento, para que su FA/h tenga resolución")
    args = p.parse_args(argv)

    split = partir(Path(args.tsv), args.test_pct, args.semilla, args.horas_test_min)
    ruta = Path(args.salida)
    ruta.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    h = hashlib.sha256(ruta.read_bytes()).hexdigest()

    print(f"hablantes: {split['hablantes_train']} train / {split['hablantes_test']} test "
          f"({args.test_pct}% por acento, semilla {args.semilla})")
    print(f"horas: {split['horas_train']} train / {split['horas_test']} test")
    print(f"{'acento':<45} {'train':>18} {'test':>18}")
    for grupo, d in split["por_acento"].items():
        print(f"{grupo[:44]:<45} {d['hablantes_train']:>6} hab {d['horas_train']:>8.2f} h "
              f"{d['hablantes_test']:>6} hab {d['horas_test']:>8.2f} h")
    print("\ncondiciones con >=5 h de test (las demás se agregan como 'otros'):")
    for grupo, d in split["por_acento"].items():
        if d["horas_test"] >= 5.0:
            print(f"  {grupo[:44]:<45} {d['horas_test']:>6.2f} h test")
    print(f"\nsha256 de la partición: {h}")
    print(f"escrita en {ruta}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
