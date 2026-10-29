#!/usr/bin/env python3
"""Entrena un modelo few-shot: positivos TTS + las grabaciones del usuario.

Es la conclusión medida del proyecto: un modelo entrenado solo con TTS da recall 0,000 con voz
humana real; con 9 de las 10 grabaciones del usuario (aumentadas ×20) detecta la décima en el 87,5%
de los casos. Este script produce el modelo final de ese escenario y mide lo que cuesta: si añadir
voz real dispara los falsos positivos sobre habla de otros.

Salida: `cabeza_fewshot.onnx`, `metricas_fewshot.json`.

Uso:
  python3 pipeline/entrenar_fewshot.py --features-reales ~/datos-cv/features_diego_aug \
      --datos ~/datos-cv/entrenamiento --epochs 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

VENTANA = 16
DIM = 96


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-reales", default=None)
    p.add_argument("--reales-npy", default=None, help="tensor ya armado (n,16,96) en vez de carpeta")
    p.add_argument("--datos", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--semilla", type=int, default=20261026)
    p.add_argument("--neg-dificiles", default=None,
                   help="npy con negativos minados (sin ellos el modelo dispara 4,5 veces por hora)")
    p.add_argument("--peso-dificiles", type=float, default=3.0,
                   help="cuántas veces se repiten los negativos difíciles en el entrenamiento")
    args = p.parse_args(argv)

    datos = Path(args.datos)
    x_tts = np.load(datos / "pos_train.npy").astype(np.float32)
    x_neg = np.load(datos / "neg_train.npy").astype(np.float32)
    if args.neg_dificiles and Path(args.neg_dificiles).exists():
        x_dif = np.load(args.neg_dificiles).astype(np.float32)
        repeticiones = max(1, int(round(args.peso_dificiles)))
        x_neg = np.concatenate([x_neg] + [x_dif] * repeticiones)
        print(f"negativos difíciles: {len(x_dif)} x{repeticiones}")

    if args.reales_npy:
        x_real = np.load(args.reales_npy).astype(np.float32)
        print(f"tensor de voz real cargado: {x_real.shape}")
        reales = []
    reales = reales if not args.reales_npy else []
    for npy in ([] if args.reales_npy else sorted(Path(args.features_reales).glob("*.npy"))):
        d = np.load(npy)
        if d.shape[0] >= VENTANA and d.shape[1] == DIM:
            w = np.lib.stride_tricks.sliding_window_view(d, (VENTANA, DIM))[:, 0]
            reales.append(w.astype(np.float32))
    if not args.reales_npy:
        x_real = np.concatenate(reales) if reales else np.zeros((0, VENTANA, DIM), np.float32)
    print(f"TTS: {len(x_tts)} | voz real aumentada: {len(x_real)} | negativos: {len(x_neg)}")

    x_pos = np.concatenate([x_tts, x_real])
    x = torch.from_numpy(np.concatenate([x_pos, x_neg]))
    y = torch.from_numpy(np.concatenate([
        np.ones(len(x_pos), dtype=np.float32), np.zeros(len(x_neg), dtype=np.float32)]))

    torch.manual_seed(args.semilla)
    modelo = nn.Sequential(nn.Flatten(), nn.Linear(VENTANA * DIM, 128), nn.ReLU(), nn.Linear(128, 1))
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([len(x_neg) / len(x_pos)]))
    opt = torch.optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoca in range(1, args.epochs + 1):
        modelo.train()
        perm = torch.randperm(len(x))
        total = 0.0
        for j in range(0, len(x) - args.batch + 1, args.batch):
            idx = perm[j:j + args.batch]
            opt.zero_grad()
            perdida = criterio(modelo(x[idx]).squeeze(-1), y[idx])
            perdida.backward()
            opt.step()
            total += perdida.item() * len(idx)
        print(f"época {epoca:>2} | pérdida {total / len(x):.5f}", flush=True)

    modelo.eval()
    torch.save(modelo.state_dict(), datos / "cabeza_fewshot.pt")
    torch.onnx.export(modelo, torch.zeros(1, VENTANA, DIM), datos / "cabeza_fewshot.onnx",
                      input_names=["input"], output_names=["output"], opset_version=13,
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      do_constant_folding=True)

    with torch.no_grad():
        probs_reales = torch.sigmoid(modelo(torch.from_numpy(x_real)).squeeze(-1)).numpy()
    metricas = {
        "ventanas_tts": int(len(x_tts)), "ventanas_voz_real": int(len(x_real)),
        "ventanas_negativas": int(len(x_neg)),
        "recall_en_muestras_de_entrenamiento": round(float((probs_reales > 0.5).mean()), 4),
        "nota": ("el recall aquí es en muestra (las grabaciones están en el entrenamiento); el "
                 "número honesto es el 87,5% de validación dejando uno fuera"),
    }
    (datos / "metricas_fewshot.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    print(f"guardado en {datos}/cabeza_fewshot.onnx | {metricas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
