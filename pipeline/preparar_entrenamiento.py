#!/usr/bin/env python3
"""Arma los tensores de entrenamiento a partir de características ya calculadas.

Qué hace y por qué así:

  - **Positivos**: ventanas de 16 frames de los clips sintéticos. Se parte **por voz**
    (`ef_dora`/`em_alex` a train, `em_santa` a validación) porque la pregunta del proyecto es si
    el modelo generaliza a una voz que nunca vio.
  - **Negativos**: ventanas de habla real (Common Voice), partidas **por hablante**. Los hablantes
    de test (`split.json`) no se tocan: si se usaran para elegir el modelo, el test dejaría de ser
    un test y pasaría a ser validación.
  - **Tercera partición**: de los hablantes de train se aparta un 10% como validación, para
    poder parar el entrenamiento sin mirar el test.

Salida: `pos_train.npy`, `pos_val.npy`, `neg_train.npy`, `neg_val.npy` en float16 (mitad de disco
que float32) más `resumen.json` con conteos y sha256.

Uso:
  python3 pipeline/preparar_entrenamiento.py \
      --split ~/datos-cv/split.json --tsv ~/datos-cv/validado_con_duraciones.tsv \
      --features ~/datos-cv/features --features-positivos ~/datos-cv/features_positivos \
      --salida ~/datos-cv/entrenamiento --neg-train 150000 --neg-val 25000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np

VENTANA = 16
DIM = 96
VOCES_TRAIN = ("ef_dora", "em_alex")
VOCES_VAL = ("em_santa",)


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def mapa_hablantes(tsv: Path) -> dict[str, str]:
    with tsv.open(encoding="utf-8", newline="") as f:
        return {
            (fila.get("path") or "").strip(): (fila.get("client_id") or "").strip()
            for fila in csv.DictReader(f, delimiter="\t")
            if fila.get("path")
        }


def ventanas_positivas(features_pos: Path, manifiesto: Path) -> dict[str, list[np.ndarray]]:
    """Ventanas de 16 frames por voz, desde el manifiesto de los clips sintéticos."""
    por_voz: dict[str, list[np.ndarray]] = {}
    with manifiesto.open(encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            fila = json.loads(linea)
            npy = features_pos / fila["archivo"]
            if not npy.exists():
                continue
            datos = np.load(npy, mmap_mode="r")
            if datos.shape[0] < VENTANA or datos.shape[1] != DIM:
                continue
            voz = fila["archivo"].split("_")[0] + "_" + fila["archivo"].split("_")[1]
            for inicio in range(datos.shape[0] - VENTANA + 1):
                por_voz.setdefault(voz, []).append(
                    np.array(datos[inicio:inicio + VENTANA], dtype=np.float16)
                )
    return por_voz


def ventanas_negativas(
    features: Path, manifiestos: list[Path], hablantes: dict[str, str],
    permitidos: set[str], cantidad: int, rng: random.Random,
) -> tuple[np.ndarray, dict]:
    """Una ventana al azar por clip, hasta `cantidad`, solo de hablantes permitidos."""
    candidatos: list[tuple[Path, str]] = []
    for mani in manifiestos:
        if not mani.exists():
            continue
        with mani.open(encoding="utf-8") as f:
            for linea in f:
                if not linea.strip():
                    continue
                fila = json.loads(linea)
                if "archivo" not in fila:      # fila de descarte: no aporta ventanas
                    continue
                clip = fila.get("clip") or fila.get("ruta") or ""
                clip = Path(clip).name
                if hablantes.get(clip) in permitidos:
                    candidatos.append((features / fila["archivo"], clip))
    rng.shuffle(candidatos)
    candidatos = candidatos[:cantidad]

    tensor = np.zeros((len(candidatos), VENTANA, DIM), dtype=np.float16)
    usados = 0
    for i, (npy, _) in enumerate(candidatos):
        try:
            datos = np.load(npy, mmap_mode="r")
        except Exception:
            continue
        if datos.shape[0] < VENTANA or datos.shape[1] != DIM:
            continue
        inicio = rng.randrange(0, datos.shape[0] - VENTANA + 1)
        tensor[usados] = datos[inicio:inicio + VENTANA]
        usados += 1
    return tensor[:usados], {"clips_candidatos": len(candidatos), "ventanas": usados}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True)
    p.add_argument("--tsv", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--features-positivos", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--neg-train", type=int, default=150_000)
    p.add_argument("--neg-val", type=int, default=25_000)
    p.add_argument("--validacion-pct", type=float, default=10.0)
    p.add_argument("--semilla", type=int, default=20261024)
    args = p.parse_args(argv)

    rng = random.Random(args.semilla)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_hab = set(split["train"])
    test_hab = set(split["test"])

    # Tercera partición: 10% de los hablantes de train, disjunta, para parar el entrenamiento.
    train_lista = sorted(train_hab)
    rng.shuffle(train_lista)
    n_val = max(1, round(len(train_lista) * args.validacion_pct / 100))
    val_hab = set(train_lista[:n_val])
    train_hab = set(train_lista[n_val:])

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    hablantes = mapa_hablantes(Path(args.tsv))
    features_pos = Path(args.features_positivos)
    manifiesto_pos = features_pos / "manifiesto_kokoro.jsonl"

    print("ventanas positivas...")
    por_voz = ventanas_positivas(features_pos, manifiesto_pos)
    pos_train = [v for voz in VOCES_TRAIN for v in por_voz.get(voz, [])]
    pos_val = [v for voz in VOCES_VAL for v in por_voz.get(voz, [])]
    print(f"  train: {len(pos_train)} ventanas de {', '.join(VOCES_TRAIN)}")
    for voz in VOCES_TRAIN + VOCES_VAL:
        print(f"    {voz}: {len(por_voz.get(voz, []))} ventanas")
    print(f"  val:   {len(pos_val)} ventanas de {', '.join(VOCES_VAL)}")

    manifiestos = sorted(Path(args.features).glob("manifiesto_*.jsonl"))
    if not manifiestos:
        raise SystemExit(f"no encuentro manifiesto_*.jsonl en {args.features}")

    print("ventanas negativas (train)...")
    neg_train, info_train = ventanas_negativas(
        Path(args.features), manifiestos, hablantes, train_hab, args.neg_train, rng
    )
    print("ventanas negativas (val)...")
    neg_val, info_val = ventanas_negativas(
        Path(args.features), manifiestos, hablantes, val_hab, args.neg_val, rng
    )

    # el test no se toca: se comprueba explícitamente
    assert not (val_hab & test_hab), "la validación no puede compartir hablantes con el test"
    assert not (train_hab & test_hab), "el entrenamiento no puede compartir hablantes con el test"

    if not pos_train or not pos_val:
        raise SystemExit(
            "faltan ventanas positivas: "
            f"{len(pos_train)} en train y {len(pos_val)} en validación. "
            "¿Están generadas todas las voces (incluida em_santa)?"
        )

    arrays = {
        "pos_train": np.stack(pos_train),
        "pos_val": np.stack(pos_val),
        "neg_train": neg_train,
        "neg_val": neg_val,
    }
    resumen = {
        "semilla": args.semilla,
        "ventana_frames": VENTANA,
        "dim": DIM,
        "voces_train": list(VOCES_TRAIN),
        "voces_val": list(VOCES_VAL),
        "hablantes_train": len(train_hab),
        "hablantes_val": len(val_hab),
        "hablantes_test_intactos": len(test_hab),
        "negativos": {"train": info_train, "val": info_val},
        "arrays": {},
    }
    for nombre, arr in arrays.items():
        ruta = salida / f"{nombre}.npy"
        np.save(ruta, arr)
        resumen["arrays"][nombre] = {
            "forma": list(arr.shape), "dtype": str(arr.dtype),
            "sha256": sha256(ruta), "mb": round(ruta.stat().st_size / 1e6, 1),
        }

    (salida / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{'array':<12} {'forma':<20} {'MB':>7}")
    for nombre, d in resumen["arrays"].items():
        print(f"{nombre:<12} {str(tuple(d['forma'])):<20} {d['mb']:>7}")
    print(f"\nescrito en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
