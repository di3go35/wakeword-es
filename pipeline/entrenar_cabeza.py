#!/usr/bin/env python3
"""Entrena la cabeza clasificadora del candidato C sobre características congeladas.

Arquitectura: la misma familia que openWakeWord — 16 frames × 96 dimensiones aplanados, una capa
oculta pequeña y una salida. Nada de CNN aquí: el objetivo de esta corrida es tener una **línea
base medible** contra el test congelado, no el modelo final.

Lo importante del script no es la red, es el protocolo:
  - Los positivos de validación son de **otra voz** (`em_santa`), nunca vistas en entrenamiento.
  - Los negativos de validación son de **otros hablantes**, disjuntos del train y del test.
  - El test no aparece aquí. Ni una vez.
  - El umbral se elige en validación con un FPR objetivo, no "a ojo".

Salida: `cabeza.pt`, `cabeza.onnx`, `metricas.json`.

Uso:
  python3 pipeline/entrenar_cabeza.py --datos ~/datos-cv/entrenamiento --epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

VENTANA = 16
DIM = 96


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC por rangos (Mann-Whitney), sin sklearn."""
    orden = np.argsort(s, kind="mergesort")
    rangos = np.empty_like(orden, dtype=np.float64)
    rangos[orden] = np.arange(1, len(s) + 1)
    # empates: promedio de rangos
    s_ord = s[orden]
    i = 0
    while i < len(s_ord):
        j = i
        while j + 1 < len(s_ord) and s_ord[j + 1] == s_ord[i]:
            j += 1
        if j > i:
            medio = (rangos[orden[i]] + rangos[orden[j]]) / 2
            for k in range(i, j + 1):
                rangos[orden[k]] = medio
        i = j + 1
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    suma_pos = rangos[y == 1].sum()
    return float((suma_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def tpr_a_fpr(y: np.ndarray, s: np.ndarray, fpr_objetivo: float) -> tuple[float, float]:
    """Devuelve (TPR, umbral) en el punto donde el FPR queda por debajo del objetivo."""
    negativos = np.sort(s[y == 0])
    if len(negativos) == 0:
        return float("nan"), float("nan")
    indice = max(0, min(len(negativos) - 1, int(np.floor((1 - fpr_objetivo) * len(negativos)))))
    umbral = float(negativos[indice])
    positivos = s[y == 1]
    return float((positivos > umbral).mean()), umbral


class Cabeza(nn.Module):
    """1536 -> 128 -> 1. Pequeña a propósito: el techo lo pone el embedding congelado."""

    def __init__(self, ocultas: int = 128) -> None:
        super().__init__()
        self.red = nn.Sequential(
            nn.Flatten(),
            nn.Linear(VENTANA * DIM, ocultas),
            nn.ReLU(),
            nn.Linear(ocultas, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x)


def cargar(ruta: Path) -> np.ndarray:
    return np.load(ruta).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ocultas", type=int, default=128)
    p.add_argument("--paciencia", type=int, default=6)
    p.add_argument("--semilla", type=int, default=20261024)
    args = p.parse_args(argv)

    torch.manual_seed(args.semilla)
    np.random.seed(args.semilla)
    datos = Path(args.datos)

    x_pos = cargar(datos / "pos_train.npy")
    x_neg = cargar(datos / "neg_train.npy")
    v_pos = cargar(datos / "pos_val.npy")
    v_neg = cargar(datos / "neg_val.npy")
    print(f"train: {len(x_pos)} pos / {len(x_neg)} neg | val: {len(v_pos)} pos / {len(v_neg)} neg")

    x = torch.from_numpy(np.concatenate([x_pos, x_neg]))
    y = torch.from_numpy(np.concatenate([
        np.ones(len(x_pos), dtype=np.float32), np.zeros(len(x_neg), dtype=np.float32)
    ]))
    xv = torch.from_numpy(np.concatenate([v_pos, v_neg]))
    yv = torch.from_numpy(np.concatenate([
        np.ones(len(v_pos), dtype=np.float32), np.zeros(len(v_neg), dtype=np.float32)
    ]))

    cargador = DataLoader(TensorDataset(x, y), batch_size=args.batch, shuffle=True,
                          num_workers=0, drop_last=True)
    modelo = Cabeza(args.ocultas)
    peso_pos = torch.tensor([len(x_neg) / max(1, len(x_pos))], dtype=torch.float32)
    criterio = nn.BCEWithLogitsLoss(pos_weight=peso_pos)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)

    mejor_auc, mejor_estado, sin_mejora = -1.0, None, 0
    historia = []
    for epoca in range(1, args.epochs + 1):
        inicio = time.time()
        modelo.train()
        perdida_suma = 0.0
        for xb, yb in cargador:
            optimizador.zero_grad()
            perdida = criterio(modelo(xb).squeeze(-1), yb)
            perdida.backward()
            optimizador.step()
            perdida_suma += perdida.item() * len(xb)

        modelo.eval()
        with torch.no_grad():
            logits_val = modelo(xv).squeeze(-1).numpy()
        prob_val = 1 / (1 + np.exp(-logits_val))
        yv_np = yv.numpy()
        a = auc(yv_np, prob_val)
        tpr1, umbral1 = tpr_a_fpr(yv_np, prob_val, 0.01)
        tpr01, _ = tpr_a_fpr(yv_np, prob_val, 0.001)
        historia.append({"epoca": epoca, "perdida_train": round(perdida_suma / len(x), 5),
                         "auc_val": round(a, 5), "tpr_fpr1": round(tpr1, 4),
                         "tpr_fpr01": round(tpr01, 4), "segundos": round(time.time() - inicio, 1)})
        print(f"época {epoca:>2} | pérdida {perdida_suma / len(x):.5f} | AUC {a:.5f} | "
              f"TPR@1%FPR {tpr1:.3f} | TPR@0,1%FPR {tpr01:.3f} | {time.time() - inicio:.0f}s")

        if a > mejor_auc + 1e-5:
            mejor_auc, sin_mejora = a, 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                print(f"sin mejora en {args.paciencia} épocas: paro")
                break

    modelo.load_state_dict(mejor_estado)
    modelo.eval()

    with torch.no_grad():
        prob_val = 1 / (1 + np.exp(-modelo(xv).squeeze(-1).numpy()))
    tpr1, umbral1 = tpr_a_fpr(yv.numpy(), prob_val, 0.01)

    torch.save(modelo.state_dict(), datos / "cabeza.pt")
    ejemplo = torch.zeros(1, VENTANA, DIM)
    torch.onnx.export(
        modelo, ejemplo, datos / "cabeza.onnx", input_names=["input"], output_names=["output"],
        opset_version=13, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )

    metricas = {
        "arquitectura": f"Flatten(16x96) -> Dense({args.ocultas}) ReLU -> Dense(1)",
        "parametros": int(sum(p.numel() for p in modelo.parameters())),
        "epocas_entrenadas": len(historia),
        "mejor_auc_val": round(mejor_auc, 5),
        "tpr_a_fpr_1pct": round(tpr1, 4),
        "umbral_logit_fpr1": round(umbral1, 5),
        "ventana_frames": VENTANA,
        "historia": historia,
        "positivos_val_de_otra_voz": True,
        "test_usado": False,
    }
    (datos / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"\nmejor AUC val: {mejor_auc:.5f} | TPR@1%FPR: {tpr1:.3f}")
    print(f"parámetros: {metricas['parametros']} | guardado en {datos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
