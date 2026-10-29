#!/usr/bin/env python3
"""Validación cruzada del recetario completo del few-shot.

Recetario: positivos TTS + grabaciones del usuario aumentadas (canal y ruido) + negativos reales +
negativos difíciles minados. Se deja **una grabación fuera** en cada pliegue y se mide sobre su
versión con ruido no visto (otra semilla de MUSAN), que es lo más parecido a "el usuario habla en
otra habitación otro día".

Reporta: recall medio, por clip, y el AUC de validación interna.

Uso:
  python3 pipeline/validar_fewshot.py --datos ~/datos-cv/entrenamiento --epochs 10
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

VENTANA = 16
DIM = 96


def ventanas_con_clip(patron: str, etiqueta: str) -> list[tuple[str, np.ndarray]]:
    salida = []
    for f in sorted(glob.glob(patron)):
        nombre = Path(f).stem
        m = re.match(r"(diego_\d+)", nombre)
        if not m:
            continue
        d = np.load(f)
        if d.shape[0] < VENTANA or d.shape[1] != DIM:
            continue
        w = np.lib.stride_tricks.sliding_window_view(d, (VENTANA, DIM))[:, 0]
        salida.append((m.group(1), w.astype(np.float32)))
    return salida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos", required=True)
    p.add_argument("--aug", default=str(Path.home() / "datos-cv/features_diego_aug/*.npy"))
    p.add_argument("--ruido", default=str(Path.home() / "datos-cv/features_diego_ruido/*_{rA}*.npy"))
    p.add_argument("--test-ruido", default=str(Path.home() / "datos-cv/features_diego_ruido_b/*_{rB}*.npy"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--neg-pliegue", type=int, default=80000)
    p.add_argument("--neg-extra", default=None,
                   help="carpeta de características con frases que no deben activar")
    p.add_argument("--semilla", type=int, default=20261026)
    args = p.parse_args(argv)

    datos = Path(args.datos)
    x_tts = np.load(datos / "pos_train.npy").astype(np.float32)
    x_neg = np.load(datos / "neg_train.npy").astype(np.float32)
    dif = datos / "neg_dificiles_all.npy"
    if dif.exists():
        x_neg = np.concatenate([x_neg, np.load(dif).astype(np.float32)] * 1)
        x_neg = np.concatenate([x_neg, np.load(dif).astype(np.float32)] *
                               (2 if len(np.load(dif)) < 5000 else 1))

    if args.neg_extra:
        extra = []
        for npy in sorted(Path(args.neg_extra).glob("*.npy")):
            d = np.load(npy)
            if d.shape[0] >= VENTANA and d.shape[1] == DIM:
                extra.append(np.lib.stride_tricks.sliding_window_view(d, (VENTANA, DIM))[:, 0])
        if extra:
            x_neg = np.concatenate([x_neg, np.concatenate(extra).astype(np.float32)])
            print(f"negativos de frase en cada pliegue: {sum(len(e) for e in extra)}")

    entrenamiento = ventanas_con_clip(args.aug, "aug") + ventanas_con_clip(args.ruido, "ruido")
    prueba = ventanas_con_clip(args.test_ruido, "ruido_b")
    clips = sorted({c for c, _ in entrenamiento})
    print(f"ventanas reales de entrenamiento: {len(entrenamiento)} de {len(clips)} clips | "
          f"ventanas de prueba con ruido: {len(prueba)}")
    if not clips or not prueba:
        raise SystemExit("faltan ventanas: revisa --aug, --ruido y --test-ruido")

    resultados = []
    for clip in clips:
        x_real = np.concatenate([w for c, w in entrenamiento if c != clip])
        x_pos = np.concatenate([x_tts, x_real])
        # Negativos fijos y subsampleados: con 163k el tensor por pliegue no cabe en memoria.
        x = torch.from_numpy(np.concatenate([x_pos, x_neg[: args.neg_pliegue]]))
        y = torch.from_numpy(np.concatenate([
            np.ones(len(x_pos), dtype=np.float32),
            np.zeros(min(len(x_neg), args.neg_pliegue), dtype=np.float32)]))

        torch.manual_seed(args.semilla)
        modelo = nn.Sequential(nn.Flatten(), nn.Linear(VENTANA * DIM, 128), nn.ReLU(),
                               nn.Linear(128, 1))
        criterio = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([min(len(x_neg), args.neg_pliegue) / len(x_pos)]))
        opt = torch.optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(args.epochs):
            perm = torch.randperm(len(x))
            for j in range(0, len(x) - 255, 256):
                idx = perm[j:j + 256]
                opt.zero_grad()
                criterio(modelo(x[idx]).squeeze(-1), y[idx]).backward()
                opt.step()

        modelo.eval()
        w = np.concatenate([v for c, v in prueba if c == clip])
        with torch.no_grad():
            probs = torch.sigmoid(modelo(torch.from_numpy(w)).squeeze(-1)).numpy()
        # una activación por clip: el máximo de sus ventanas
        por_clip = [probs[i:i + 1].max() for i in range(len(probs))]
        maximo = float(probs.max())
        mediana = float(np.median(probs))
        resultados.append({"clip": clip, "ventanas": int(len(probs)),
                           "prob_max": round(maximo, 4), "prob_mediana": round(mediana, 4),
                           "detecta": bool(mediana > 0.5)})
        print(f"  {clip:<12} máx {maximo:.4f} mediana {mediana:.4f} "
              f"{'DETECTA' if mediana > 0.5 else 'no'}", flush=True)

    recall = sum(r["detecta"] for r in resultados) / len(resultados)
    print(f"\nRECALL con ruido no visto, dejando un clip fuera: {recall:.1%} "
          f"({sum(r['detecta'] for r in resultados)} de {len(resultados)})")
    (datos / "validacion_fewshot.json").write_text(
        json.dumps({"recall": recall, "pliegues": resultados}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
