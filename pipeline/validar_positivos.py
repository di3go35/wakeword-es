#!/usr/bin/env python3
"""Valida clips TTS de una frase: descarta repeticiones y audios inutilizables.

Regla (decisión 35):
  - **Un solo segmento de voz** (los huecos de menos de 0,6 s se fusionan, para no confundir la
    pausa interna de "hey… celeste" con una frase repetida).
  - Duración de voz entre **0,9 y 2,0 s**: por debajo falta la frase, por encima está repetida.
  - Escribe siempre el motivo del descarte; el número que manda es el de válidos, no el pedido.

Uso:
  python3 pipeline/validar_positivos.py --audio <carpeta> --manifiesto <jsonl> \
      --salida <jsonl filtrado> [--copiar-a <carpeta>]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

UMBRAL_DB = -40.0
FRAME_S = 0.025
HOP_S = 0.010
FUSION_S = 0.60
VOZ_MIN_S = 0.90
VOZ_MAX_S = 2.00
# Rachas finas: "hey" y "celeste" son dos; una tercera delata una repetición truncada del tipo
# "hey celeste … hey cel", que con la fusión de 0,6 s se disfraza de una sola frase.
FUSION_FINA_S = 0.12
RACHA_MIN_S = 0.10
RACHAS_FINAS_MAX = 2


def segmentos_voz(x: np.ndarray, sr: int) -> list[tuple[float, float]]:
    n, hop = int(FRAME_S * sr), int(HOP_S * sr)
    if len(x) < n:
        return []
    tramas = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    db = 20 * np.log10(np.sqrt((tramas ** 2).mean(axis=1) + 1e-12))
    activo = db > UMBRAL_DB
    bruto, ini = [], None
    for i, a in enumerate(activo):
        if a and ini is None:
            ini = i
        elif not a and ini is not None:
            bruto.append([ini * hop / sr, i * hop / sr])
            ini = None
    if ini is not None:
        bruto.append([ini * hop / sr, len(activo) * hop / sr])
    fusion = []
    for s in bruto:
        if fusion and s[0] - fusion[-1][1] < FUSION_S:
            fusion[-1][1] = s[1]
        else:
            fusion.append(s)
    return [(a, b) for a, b in fusion if b - a >= 0.15]


def segmentos_voz_finas(x: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Rachas con fusión mínima: separa "hey" y "celeste" y expone las repeticiones pegadas."""
    n, hop = int(FRAME_S * sr), int(HOP_S * sr)
    if len(x) < n:
        return []
    tramas = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    db = 20 * np.log10(np.sqrt((tramas ** 2).mean(axis=1) + 1e-12))
    activo = db > UMBRAL_DB
    bruto, ini = [], None
    for i, a in enumerate(activo):
        if a and ini is None:
            ini = i
        elif not a and ini is not None:
            bruto.append([ini * hop / sr, i * hop / sr])
            ini = None
    if ini is not None:
        bruto.append([ini * hop / sr, len(activo) * hop / sr])
    fusion = []
    for s in bruto:
        if fusion and s[0] - fusion[-1][1] < FUSION_FINA_S:
            fusion[-1][1] = s[1]
        else:
            fusion.append(s)
    return [(a, b) for a, b in fusion if b - a >= RACHA_MIN_S]


def validar(audio: Path, nombre: str) -> tuple[bool, str, dict]:
    x, sr = sf.read(audio / nombre, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    segs = segmentos_voz(x, sr)
    finas = segmentos_voz_finas(x, sr)
    voz = sum(b - a for a, b in segs)
    info = {"segmentos": len(segs), "rachas_finas": len(finas), "voz_s": round(voz, 3),
            "clip_s": round(len(x) / sr, 3)}
    if len(segs) == 0:
        return False, "sin voz detectable", info
    if len(segs) > 1:
        return False, f"frase repetida ({len(segs)} segmentos de voz)", info
    if voz < VOZ_MIN_S:
        return False, f"voz demasiado corta ({voz:.2f} s)", info
    if voz > VOZ_MAX_S:
        return False, f"voz demasiado larga ({voz:.2f} s)", info
    if len(finas) > RACHAS_FINAS_MAX:
        return False, f"repetición truncada ({len(finas)} rachas de voz)", info
    return True, "", info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", required=True)
    p.add_argument("--manifiesto", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--copiar-a", default=None)
    args = p.parse_args(argv)

    audio = Path(args.audio)
    filas = [json.loads(l) for l in Path(args.manifiesto).read_text(
        encoding="utf-8").splitlines() if l.strip()]

    validos, descartes = [], []
    for fila in filas:
        if not (audio / fila["archivo"]).exists():
            descartes.append({"archivo": fila["archivo"], "motivo": "archivo ausente"})
            continue
        ok, motivo, info = validar(audio, fila["archivo"])
        if ok:
            validos.append({**fila, **info})
        else:
            descartes.append({"archivo": fila["archivo"], "motivo": motivo, **info})

    Path(args.salida).write_text(
        "\n".join(json.dumps(f, ensure_ascii=False) for f in validos) + "\n", encoding="utf-8"
    )
    if args.copiar_a:
        destino = Path(args.copiar_a)
        destino.mkdir(parents=True, exist_ok=True)
        for f in validos:
            (audio / f["archivo"]).replace(destino / f["archivo"])
        print(f"válidos movidos a {destino}")

    print(f"revisados: {len(filas)} | válidos: {len(validos)} | descartados: {len(descartes)}")
    motivos: dict[str, int] = {}
    for d in descartes:
        clave = d["motivo"].split("(")[0].strip()
        motivos[clave] = motivos.get(clave, 0) + 1
    for m, n in sorted(motivos.items(), key=lambda x: -x[1]):
        print(f"  {m}: {n}")
    if validos:
        v = np.array([f["voz_s"] for f in validos])
        print(f"duración de voz de los válidos: min {v.min():.2f} | mediana {np.median(v):.2f} "
              f"| max {v.max():.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
