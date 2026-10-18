#!/usr/bin/env python3
"""Une `validated.tsv` con `clip_durations.tsv` y reporta el corpus real.

Por qué existe: el `validated.tsv` de Common Voice **no trae duración**, así que cualquier
planificación por horas sería una estimación. `clip_durations.tsv` sí trae la duración por
clip, en **milisegundos**. Este paso las une, convierte a segundos y deja un TSV listo para
`planificar_lotes.py`, además de un informe de qué hay de verdad en el corpus (clips, horas,
hablantes y acentos) para que las decisiones se tomen sobre datos medidos.

Uso:
  python3 pipeline/preparar_validado.py \
      --dir ~/datos-cv/corpus/cv-corpus-27.0-2026-09-11/es \
      --salida ~/datos-cv/validado_con_duraciones.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def leer_duraciones(ruta: Path) -> dict[str, float]:
    """clip -> segundos. Common Voice nombra la columna `duration[ms]` y viene en milisegundos.

    Se aceptan `duration[ms]` y `duration` (por si el corpus cambia de esquema): si la
    columna declara milisegundos, se convierte; si no, se asume que ya viene en segundos.
    """
    out: dict[str, float] = {}
    with ruta.open(encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f, delimiter="\t")
        columnas = lector.fieldnames or []
        if "clip" not in columnas:
            raise ValueError(f"{ruta}: se esperaba la columna `clip` (columnas: {columnas})")
        if "duration[ms]" in columnas:
            columna, divisor = "duration[ms]", 1000.0
        elif "duration" in columnas:
            columna, divisor = "duration", 1.0
        else:
            raise ValueError(
                f"{ruta}: no encuentro la duración (columnas: {columnas}). "
                "Se esperaba `duration[ms]` o `duration`."
            )
        for fila in lector:
            try:
                out[fila["clip"].strip()] = float(fila[columna]) / divisor
            except (TypeError, ValueError):
                continue
    return out


def procesar(dir_corpus: Path, salida: Path) -> dict:
    validado = dir_corpus / "validated.tsv"
    duraciones = leer_duraciones(dir_corpus / "clip_durations.tsv")

    with validado.open(encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f, delimiter="\t")
        columnas = list(lector.fieldnames or [])
        filas = [f for f in lector if (f.get("path") or "").strip()]

    sin_duracion = [f["path"] for f in filas if f["path"] not in duraciones]
    por_acento: dict[str, dict] = defaultdict(
        lambda: {"clips": 0, "horas": 0.0, "hablantes": set()}
    )
    total_horas = 0.0
    with salida.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(
            f, fieldnames=columnas + ["duration"], delimiter="\t", extrasaction="ignore"
        )
        escritor.writeheader()
        for fila in filas:
            seg = duraciones.get(fila["path"])
            if seg is None:
                continue
            fila["duration"] = f"{seg:.3f}"
            escritor.writerow(fila)
            acento = (fila.get("accents") or "").strip() or "(sin acento declarado)"
            por_acento[acento]["clips"] += 1
            por_acento[acento]["horas"] += seg
            por_acento[acento]["hablantes"].add((fila.get("client_id") or "").strip())
            total_horas += seg

    informe = {
        "clips_validados": len(filas),
        "clips_sin_duracion": len(sin_duracion),
        "horas_totales": round(total_horas / 3600.0, 2),
        "duracion_media_s": round(total_horas / max(len(filas) - len(sin_duracion), 1), 3),
        "acentos": {
            k: {"clips": v["clips"], "horas": round(v["horas"] / 3600.0, 2),
                "hablantes": len(v["hablantes"])}
            for k, v in sorted(por_acento.items(), key=lambda kv: -kv[1]["clips"])
        },
        "salida": str(salida),
    }
    return informe


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True, help="directorio es/ con los .tsv del corpus")
    p.add_argument("--salida", required=True, help="TSV de salida con duraciones en segundos")
    args = p.parse_args(argv)

    informe = procesar(Path(args.dir), Path(args.salida))
    print(json.dumps(informe, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
