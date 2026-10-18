#!/usr/bin/env python3
"""Verifica que las características de un lote estén completas y sean consistentes.

No basta con que existan archivos: hay que poder afirmar que **cada clip de la lista tiene
sus características**, que no sobran archivos huérfanos y que la forma del vector es la
esperada. Esta es la comprobación que convierte «creo que se procesó» en «está verificado».

Uso:
  python3 pipeline/verificar_features.py --lote 000 \
      --lista ~/datos-cv/lotes/lote_000.txt \
      --features ~/datos-cv/features --dim 96
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def verificar(lote: str, lista: Path, features: Path, dim: int = 96) -> dict:
    clips = [l.strip() for l in lista.read_text(encoding="utf-8").splitlines() if l.strip()]
    esperados = {Path(c).stem for c in clips}

    manifiestos = sorted(features.glob(f"manifiesto_{lote}*.jsonl"))
    if not manifiestos:
        raise FileNotFoundError(
            f"no hay manifiestos manifiesto_{lote}*.jsonl en {features}: el lote no se procesó"
        )

    vistos: dict[str, dict] = {}
    filas = 0
    horas = 0.0
    formas: dict[tuple, int] = {}
    for man in manifiestos:
        for linea in man.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            d = json.loads(linea)
            filas += 1
            horas += d["segundos"] / 3600.0
            clave = Path(d["clip"]).stem
            vistos[clave] = d
            formas[tuple(d["forma"])] = formas.get(tuple(d["forma"]), 0) + 1

    faltan = esperados - set(vistos)
    sobran = set(vistos) - esperados
    sin_npy = [c for c in vistos if not (features / f"{c}.npy").is_file()]
    dims_malas = {
        f: n for f, n in formas.items()
        if len(f) != 2 or f[-1] != dim
    }

    return {
        "lote": lote,
        "clips_en_lista": len(clips),
        "filas_en_manifiestos": filas,
        "clips_unicos": len(vistos),
        "manifiestos": [m.name for m in manifiestos],
        "horas": round(horas, 2),
        "formas_distintas": {str(k): v for k, v in sorted(formas.items(), key=lambda kv: -kv[1])[:5]},
        "faltan_en_manifiesto": len(faltan),
        "sobran_en_manifiesto": len(sobran),
        "sin_archivo_npy": len(sin_npy),
        "dims_fuera_de_lo_esperado": len(dims_malas),
        "completo": not faltan and not sobran and not sin_npy and not dims_malas,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lote", required=True)
    p.add_argument("--lista", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--dim", type=int, default=96)
    args = p.parse_args(argv)

    r = verificar(args.lote, Path(args.lista), Path(args.features), args.dim)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    if not r["completo"]:
        print("VERIFICACION FALLIDA: el lote no está completo", file=sys.stderr)
        return 1
    print("VERIFICADO: el lote está completo y consistente")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
