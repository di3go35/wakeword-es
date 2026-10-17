#!/usr/bin/env python3
"""Planifica los lotes de extracción a partir del `validated.tsv` de Common Voice.

Por qué existe: el corpus son ~52 GB comprimidos y ~597 h de audio validado. Extraer todo
de golpe no cabe en disco junto al archivo, así que se procesa por tandas (por defecto 100 h
cada una): se extraen esos clips, se calculan sus características y se borran. Este script
decide qué va en cada tanda y deja la lista de rutas para `tar -T`.

Uso:
  python3 pipeline/planificar_lotes.py --tsv ~/datos-cv/validated.tsv \
      --salida ~/datos-cv/lotes --horas-por-lote 100

  # solo un acento (p. ej. el andino-pacífico)
  python3 pipeline/planificar_lotes.py --tsv validated.tsv --salida lotes --acentos andino

Notas de honestidad de datos:
  - La duración por clip se toma de la columna `duration` del TSV si existe y es numérica.
  - Si no existe, se usa una estimación declarada (`--duracion-media`) y el informe lo dice
    explícitamente. No se inventa precisión: se reporta el modo usado.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DURACION_MEDIA_POR_DEFECTO = 5.5  # segundos, solo si el TSV no trae `duration`


def leer_tsv(ruta: str | Path) -> tuple[list[dict], list[str]]:
    """Devuelve (filas, columnas). Falla claro si el archivo no tiene la columna `path`."""
    with Path(ruta).open(encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f, delimiter="\t")
        columnas = list(lector.fieldnames or [])
        if "path" not in columnas:
            raise ValueError(
                f"{ruta}: no parece un TSV de Common Voice (columnas: {columnas})"
            )
        filas = [fila for fila in lector if (fila.get("path") or "").strip()]
    if not filas:
        raise ValueError(f"{ruta}: no contiene filas con `path`")
    return filas, columnas


def duraciones(
    filas: list[dict], columnas: list[str], duracion_media: float
) -> tuple[list[float], str]:
    """Duración por fila y el modo usado ('tsv' o 'estimada')."""
    if "duration" in columnas:
        valores = []
        validos = 0
        for fila in filas:
            try:
                valores.append(float(fila["duration"]))
                validos += 1
            except (TypeError, ValueError):
                valores.append(duracion_media)
        if validos == len(filas):
            return valores, "tsv"
        return valores, f"tsv ({validos}/{len(filas)} válidas, resto estimado)"
    return [duracion_media] * len(filas), "estimada"


def planificar(
    filas: list[dict],
    dur: list[float],
    horas_por_lote: float,
    acentos: set[str] | None = None,
) -> list[dict]:
    """Reparte las filas en lotes de como máximo `horas_por_lote` horas."""
    if horas_por_lote <= 0:
        raise ValueError("horas_por_lote debe ser > 0")
    limite = horas_por_lote * 3600.0

    lotes: list[dict] = []
    actual: dict = {"horas": 0.0, "clips": [], "por_acento": Counter()}
    for fila, segundos in zip(filas, dur):
        acento = (fila.get("accents") or "").strip()
        if acentos is not None and acento not in acentos:
            continue
        if actual["clips"] and actual["horas"] + segundos > limite:
            lotes.append(actual)
            actual = {"horas": 0.0, "clips": [], "por_acento": Counter()}
        actual["clips"].append(
            {"path": fila["path"].strip(), "accents": acento,
             "client_id": (fila.get("client_id") or "").strip(),
             "segundos": round(segundos, 3)}
        )
        actual["horas"] += segundos
        actual["por_acento"][acento or "-"] += 1
    if actual["clips"]:
        lotes.append(actual)

    for i, lote in enumerate(lotes):
        lote["indice"] = f"{i:03d}"
        lote["clips"] = lote["clips"]
        lote["por_acento"] = dict(lote["por_acento"].most_common())
        lote["horas"] = round(lote["horas"] / 3600.0, 3)
    return lotes


def escribir(lotes: list[dict], salida: str | Path, modo_duracion: str) -> dict:
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    for lote in lotes:
        lista = salida / f"lote_{lote['indice']}.txt"
        lista.write_text("\n".join(c["path"] for c in lote["clips"]) + "\n", encoding="utf-8")
    manifiesto = {
        "modo_duracion": modo_duracion,
        "lotes": [
            {
                "indice": l["indice"],
                "horas": l["horas"],
                "n_clips": len(l["clips"]),
                "por_acento": l["por_acento"],
                "lista": f"lote_{l['indice']}.txt",
            }
            for l in lotes
        ],
        "total_horas": round(sum(l["horas"] for l in lotes), 3),
        "total_clips": sum(len(l["clips"]) for l in lotes),
    }
    (salida / "manifiesto_lotes.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifiesto


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", required=True, help="ruta al validated.tsv")
    p.add_argument("--salida", required=True, help="directorio donde dejar los lotes")
    p.add_argument("--horas-por-lote", type=float, default=100.0)
    p.add_argument("--duracion-media", type=float, default=DURACION_MEDIA_POR_DEFECTO)
    p.add_argument("--acentos", default=None,
                   help="lista separada por comas; por defecto, todos")
    args = p.parse_args(argv)

    filas, columnas = leer_tsv(args.tsv)
    dur, modo = duraciones(filas, columnas, args.duracion_media)
    acentos = {a.strip() for a in args.acentos.split(",")} if args.acentos else None
    lotes = planificar(filas, dur, args.horas_por_lote, acentos)
    m = escribir(lotes, args.salida, modo)

    print(f"TSV: {args.tsv} ({len(filas)} clips, columnas: {', '.join(columnas)})")
    print(f"duración por clip: {modo}")
    if acentos:
        print(f"filtro de acentos: {', '.join(sorted(acentos))}")
    print(f"lotes: {len(m['lotes'])} | clips: {m['total_clips']} | horas: {m['total_horas']}")
    acumulado = Counter()
    for l in m["lotes"]:
        acumulado.update(l["por_acento"])
    print("clips por acento:", dict(acumulado.most_common()))
    print(f"escrito en {args.salida}/manifiesto_lotes.json y un lote_NNN.txt por tanda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
