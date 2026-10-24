#!/usr/bin/env python3
"""Selecciona el audio del test a partir de la partición por hablante congelada.

Regla:
  - Solo hablantes de `test` en `split.json` (nunca de train).
  - Acentos con >=5 h de test: se toman hablantes completos hasta ~5 h por condición.
  - Acentos menores (los que no llegan a 5 h): se toman **todos** sus clips y se agregan
    como «otros», porque no soportan una condición propia.
  - El resultado es determinista (hablantes ordenados por client_id).

Salida:
  ~/datos-cv/test/clips.txt      lista de nombres de clip (para tar)
  ~/datos-cv/test/manifiesto.json  horas y hablantes por condición + sha256 de la selección

Uso:
  python3 pipeline/preparar_test.py --tsv ~/datos-cv/validado_con_duraciones.tsv \
      --split ~/datos-cv/split.json --salida ~/datos-cv/test --horas-por-condicion 5.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import planificar_lotes as pl

HORAS_MINIMAS_CONDICION = 5.0
TOPE_POR_HABLANTE = 3600.0  # segundos por hablante y condición


def seleccionar(tsv: Path, split: Path, horas_por_condicion: float) -> tuple[list[str], dict]:
    datos = json.loads(split.read_text(encoding="utf-8"))
    test = set(datos["test"])

    # (condición, hablante) -> [(clip, segundos)]
    por_condicion: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with tsv.open(encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f, delimiter="\t"):
            hablante = (fila.get("client_id") or "").strip()
            if hablante not in test:
                continue
            condicion = pl.etiqueta_canonica(fila.get("accents"))
            clip = (fila.get("path") or "").strip()
            if not clip:
                continue
            por_condicion[condicion][hablante].append((clip, float(fila.get("duration") or 0.0)))

    horas_disponibles = {
        c: sum(s for clips in hs.values() for _, s in clips) / 3600
        for c, hs in por_condicion.items()
    }

    seleccion: list[str] = []
    detalle: dict[str, dict] = {}
    for condicion in sorted(por_condicion, key=lambda c: -horas_disponibles[c]):
        hablantes = por_condicion[condicion]
        disponible = horas_disponibles[condicion]
        grupo = condicion if disponible >= HORAS_MINIMAS_CONDICION else "otros"
        objetivo = horas_por_condicion if grupo != "otros" else None

        acumulado = 0.0  # segundos
        clips_usados = 0
        hablantes_usados = 0
        for hablante in sorted(hablantes):
            if objetivo is not None and acumulado >= objetivo * 3600:
                break
            # Tope por hablante: sin él, un hablante prolífico puede llenar toda la condición
            # y la métrica acabaría midiendo 4 voces en vez de un acento.
            del_hablante = 0.0
            for clip, segundos in hablantes[hablante]:
                if del_hablante >= TOPE_POR_HABLANTE:
                    break
                seleccion.append(clip)
                del_hablante += segundos
                acumulado += segundos
                clips_usados += 1
            hablantes_usados += 1

        d = detalle.setdefault(
            grupo, {"horas": 0.0, "hablantes": 0, "clips": 0, "disponible_h": 0.0}
        )
        d["horas"] = round(d["horas"] + acumulado / 3600, 2)
        d["hablantes"] += hablantes_usados
        d["clips"] += clips_usados
        d["disponible_h"] = round(d["disponible_h"] + disponible, 2)

    manifiesto = {
        "regla": ("hablantes de test únicamente; condiciones con >=5 h toman hablantes "
                  "completos hasta el objetivo; los acentos menores se agregan como 'otros'"),
        "horas_por_condicion": horas_por_condicion,
        "clips": len(seleccion),
        "horas_totales": round(sum(d["horas"] for d in detalle.values()), 2),
        "por_condicion": detalle,
        "sha256_seleccion": hashlib.sha256("\n".join(seleccion).encode()).hexdigest(),
    }
    return seleccion, manifiesto


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--horas-por-condicion", type=float, default=5.0)
    args = p.parse_args(argv)

    seleccion, manifiesto = seleccionar(
        Path(args.tsv), Path(args.split), args.horas_por_condicion
    )
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "clips.txt").write_text("\n".join(seleccion) + "\n", encoding="utf-8")
    (salida / "manifiesto.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"clips: {manifiesto['clips']} | horas: {manifiesto['horas_totales']}")
    for grupo, d in manifiesto["por_condicion"].items():
        print(f"  {grupo[:44]:<45} {d['clips']:>6} clips {d['horas']:>6.2f} h "
              f"{d['hablantes']:>4} hablantes (disponible {d['disponible_h']} h)")
    print(f"sha256 de la selección: {manifiesto['sha256_seleccion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
