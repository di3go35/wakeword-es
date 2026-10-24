#!/usr/bin/env python3
"""Elige la frase de activación midiendo el corpus, no el gusto.

Criterio: una buena wake word es **rara en habla normal** y **fonéticamente distintiva**.
Aquí se mide, sobre las 1.082.797 frases de Common Voice es 27.0:

  1. `freq_bigrama`: veces por millón de palabras que aparece la frase completa.
  2. `freq_nombre`: veces por millón que aparece la palabra distintiva sola.
  3. `colisiones`: palabras del corpus que comparten el **esqueleto consonántico** de la
     palabra distintiva (proxy de "suena parecido a algo común"), ponderadas por frecuencia.
  4. `silabas`: sílabas de la frase completa. Se prefieren 3-4: con 2 hay demasiadas
     confusiones, con 5+ el usuario tarda en activar.

La puntuación penaliza frecuencia, colisiones y sílabas fuera de rango. Todo es un proxy
documentado: la comprobación final es el test congelado, no esta tabla.

Uso:
  python3 pipeline/elegir_frase.py --tsv .../validated_sentences.tsv \
      --salida ~/datos-cv/frase/candidatas.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

DISPARADORES = ("oye", "oiga", "hola", "ok")
NOMBRES = (
    "cóndor", "puma", "quipu", "chasqui", "inti", "wayra", "illari", "yaku",
    "sierra", "andes", "nazca", "tumi", "urpi", "wira", "killa", "sumaq",
)
PALABRAS_POR_MILLON = 1_000_000


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos, solo letras y espacios simples."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-zñ\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def silabas(palabra: str) -> int:
    """Aproximación: grupos de vocales."""
    return max(1, len(re.findall(r"[aeiouáéíóúü]+", palabra.lower())))


def esqueleto(palabra: str) -> str:
    """Consonantes normalizadas: proxy de cómo suena (qu→k, c→k, z→s, v→b, ll→l, h muda)."""
    p = normalizar(palabra)
    p = p.replace("qu", "k").replace("ll", "l").replace("ñ", "n")
    p = p.replace("ce", "se").replace("ci", "si")
    p = p.replace("c", "k").replace("z", "s").replace("v", "b").replace("h", "")
    consonantes = re.sub(r"[aeiou]", "", p)
    return re.sub(r"(.)\1+", r"\1", consonantes)


def cargar_corpus(ruta: Path) -> tuple[Counter, Counter, int, Counter]:
    """Devuelve (unigramas, bigramas, total de palabras, frecuencia por esqueleto)."""
    unigramas: Counter = Counter()
    bigramas: Counter = Counter()
    esqueletos: Counter = Counter()
    total = 0
    with ruta.open(encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f, delimiter="\t"):
            palabras = normalizar(fila.get("sentence") or "").split()
            if not palabras:
                continue
            total += len(palabras)
            unigramas.update(palabras)
            bigramas.update(zip(palabras, palabras[1:]))
            esqueletos.update(esqueleto(p) for p in palabras)
    return unigramas, bigramas, total, esqueletos


def evaluar(
    candidata: str, unigramas: Counter, bigramas: Counter, total: int, esqueletos: Counter
) -> dict:
    palabras = normalizar(candidata).split()
    if len(palabras) != 2:
        raise ValueError(f"la candidata debe tener dos palabras: {candidata!r}")
    disparador, nombre = palabras
    bigrama = bigramas.get((disparador, nombre), 0)
    freq_nombre = unigramas.get(nombre, 0)
    colisiones = esqueletos.get(esqueleto(nombre), 0)
    n_silabas = silabas(candidata)

    por_millon = lambda n: round(n * PALABRAS_POR_MILLON / total, 3)  # noqa: E731
    penal_silabas = 0.0 if 3 <= n_silabas <= 4 else (1.5 if n_silabas in (2, 5) else 3.0)
    puntuacion = (
        math.log1p(por_millon(bigrama))
        + 0.5 * math.log1p(por_millon(freq_nombre))
        + 0.7 * math.log1p(por_millon(colisiones))
        + penal_silabas
    )
    return {
        "frase": candidata,
        "disparador": disparador,
        "nombre": nombre,
        "silabas": n_silabas,
        "esqueleto": esqueleto(nombre),
        "freq_bigrama_por_millon": por_millon(bigrama),
        "freq_nombre_por_millon": por_millon(freq_nombre),
        "colisiones_por_millon": por_millon(colisiones),
        "penalizacion_silabas": penal_silabas,
        "puntuacion": round(puntuacion, 3),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args(argv)

    unigramas, bigramas, total, esqueletos = cargar_corpus(Path(args.tsv))
    print(f"corpus: {total:,} palabras | {len(unigramas):,} unigramas | "
          f"{len(bigramas):,} bigramas distintos")

    resultados = [
        evaluar(f"{d} {n}", unigramas, bigramas, total, esqueletos)
        for d in DISPARADORES for n in NOMBRES
    ]
    resultados.sort(key=lambda r: r["puntuacion"])

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({
        "corpus": {"palabras": total, "unigramas": len(unigramas), "bigramas": len(bigramas)},
        "regla": ("puntuación = log(1+freq_bigrama) + 0,5·log(1+freq_nombre) + "
                  "0,7·log(1+colisiones) + penalización por sílabas fuera de 3-4"),
        "candidatas": resultados,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'frase':<16} {'síl':>3} {'bigrama':>9} {'nombre':>9} {'colis.':>8} {'punt.':>7}")
    for r in resultados[:args.top]:
        print(f"{r['frase']:<16} {r['silabas']:>3} {r['freq_bigrama_por_millon']:>9.3f} "
              f"{r['freq_nombre_por_millon']:>9.2f} {r['colisiones_por_millon']:>8.2f} "
              f"{r['puntuacion']:>7.3f}")
    print(f"\nescrito en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
