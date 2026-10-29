#!/usr/bin/env python3
"""Prueba few-shot: ¿basta con que el usuario grabe unas pocas muestras?

Contexto medido: un modelo entrenado solo con TTS da recall 0,000 con voz humana real (10 clips),
y ni normalizar el nivel ni estirar el tiempo lo cambian. La pregunta que decide el diseño del
producto es si **10 grabaciones del propio usuario** arreglan el problema.

Protocolo (validación cruzada dejando uno fuera):
  - Se aumentan las 10 grabaciones (canal, nivel, reverberación, velocidad) para que el modelo no
    memorice una sola toma.
  - En cada pliegue se entrena con los positivos TTS + los aumentados de **9** grabaciones y se mide
    sobre la grabación **dejada fuera**, que el modelo nunca vio.
  - Se reporta el recall medio: si es alto, el producto puede pedir 10 grabaciones al usuario; si es
    bajo, el problema es la arquitectura y no el dato.

Uso:
  python3 pipeline/probar_pocos_ejemplos.py --reales ~/datos-cv/positivos/diego \
      --aug ~/datos-cv/positivos/diego_aug --features-aug ~/datos-cv/features_diego_aug \
      --datos ~/datos-cv/entrenamiento
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

VENTANA = 16
DIM = 96
HOP_S = 0.08
CONTEXTO_S = 2.04


def ventanas_de(npy: Path) -> np.ndarray:
    datos = np.load(npy)
    if datos.shape[0] < VENTANA or datos.shape[1] != DIM:
        return np.zeros((0, VENTANA, DIM), dtype=np.float32)
    w = np.lib.stride_tricks.sliding_window_view(datos, (VENTANA, DIM))[:, 0]
    return w.astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reales", required=True)
    p.add_argument("--aug", required=True)
    p.add_argument("--features-aug", required=True)
    p.add_argument("--datos", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--semilla", type=int, default=20261026)
    p.add_argument("--neg-dificiles", default=None)
    p.add_argument("--peso-dificiles", type=float, default=3.0)
    args = p.parse_args(argv)

    datos = Path(args.datos)
    x_tts = np.load(datos / "pos_train.npy").astype(np.float32)
    x_neg = np.load(datos / "neg_train.npy").astype(np.float32)
    if args.neg_dificiles and Path(args.neg_dificiles).exists():
        x_dif = np.load(args.neg_dificiles).astype(np.float32)
        x_neg = np.concatenate([x_neg] + [x_dif] * max(1, int(args.peso_dificiles)))
        print(f"negativos difíciles en el pliegue: {len(x_dif)}")
    print(f"TTS: {len(x_tts)} ventanas | negativos reales: {len(x_neg)}")

    reales = sorted(Path(args.reales).glob("diego_*.wav"))
    feat_aug = Path(args.features_aug)
    por_clip: dict[str, list[np.ndarray]] = {}
    for npy in sorted(feat_aug.glob("*.npy")):
        original = npy.stem.split("_aug")[0] + ".wav"
        w = ventanas_de(npy)
        if len(w):
            por_clip.setdefault(original, []).append(w)
    print(f"grabaciones reales: {len(por_clip)} | ventanas aumentadas totales: "
          f"{sum(len(v) for v in por_clip.values())}")

    resultados = []
    for i, clip in enumerate(sorted(por_clip)):
        otros = [w for c, v in por_clip.items() if c != clip for w in v]
        x_pos = np.concatenate([x_tts] + otros) if otros else x_tts
        x = torch.from_numpy(np.concatenate([x_pos, x_neg]))
        y = torch.from_numpy(np.concatenate([
            np.ones(len(x_pos), dtype=np.float32), np.zeros(len(x_neg), dtype=np.float32)]))

        torch.manual_seed(args.semilla)
        modelo = nn.Sequential(nn.Flatten(), nn.Linear(VENTANA * DIM, 128), nn.ReLU(),
                               nn.Linear(128, 1))
        criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([len(x_neg) / len(x_pos)]))
        opt = torch.optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(args.epochs):
            perm = torch.randperm(len(x))
            for j in range(0, len(x) - 255, 256):
                idx = perm[j:j + 256]
                opt.zero_grad()
                criterio(modelo(x[idx]).squeeze(-1), y[idx]).backward()
                opt.step()

        modelo.eval()
        with torch.no_grad():
            pruebas = por_clip[clip]
            probs = [float(torch.sigmoid(modelo(torch.from_numpy(w)).squeeze(-1)).max())
                     for w in pruebas]
        resultados.append({"clip": clip, "variantes": len(probs),
                           "prob_max_media": round(float(np.mean(probs)), 4),
                           "prob_max_mediana": round(float(np.median(probs)), 4),
                           "detectado": bool(np.median(probs) > 0.5)})
        print(f"  {clip:<16} prob media {np.mean(probs):.4f} mediana {np.median(probs):.4f} "
              f"{'DETECTA' if np.median(probs) > 0.5 else 'no'}")

    recall = sum(r["detectado"] for r in resultados) / len(resultados)
    print(f"\nRECALL few-shot dejando uno fuera: {recall:.2%} ({sum(r['detectado'] for r in resultados)}"
          f" de {len(resultados)})")
    (datos / "fewshot.json").write_text(json.dumps({"recall": recall, "pliegues": resultados},
                                                   indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
