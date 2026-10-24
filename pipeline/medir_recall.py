#!/usr/bin/env python3
"""Mide recall de la frase sobre un conjunto de positivos (p. ej. otro motor TTS).

Mismo protocolo que la medición de falsos positivos: ventana de 16 frames, capa anti-disparo
(N frames consecutivos sobre el umbral), y un clip cuenta como detectado si produce al menos un
evento. Así el recall y el FA/h son comparables: mismo umbral, misma capa.

Se reporta la curva completa de umbrales, no un único punto: el umbral se elige después mirando
el compromiso con el FA/h medido sobre habla real.

Uso:
  python3 pipeline/medir_recall.py --modelo ~/datos-cv/entrenamiento/cabeza.onnx \
      --features ~/datos-cv/features_chatterbox --manifiesto ~/datos-cv/positivos/chatterbox/manifiesto.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

VENTANA = 16
DIM = 96


def puntuar(sesion, datos: np.ndarray) -> np.ndarray:
    """Probabilidad por frame (la ventana que termina en ese frame)."""
    n = datos.shape[0]
    puntajes = np.zeros(n, dtype=np.float32)
    if n < VENTANA:
        return puntajes
    ventanas = np.lib.stride_tricks.sliding_window_view(datos, (VENTANA, DIM))[:, 0]
    lote = 4096
    for i in range(0, len(ventanas), lote):
        trozo = np.ascontiguousarray(ventanas[i:i + lote], dtype=np.float32)
        salida = sesion.run(None, {"input": trozo})[0].ravel()
        p = 1 / (1 + np.exp(-salida))
        puntajes[i + VENTANA - 1:i + VENTANA - 1 + len(p)] = p
    return puntajes


def hay_evento(puntajes: np.ndarray, umbral: float, consecutivos: int,
               enfriamiento_s: float = 1.0) -> bool:
    racha, ultimo = 0, -10**9
    frames_enfriamiento = max(1, int(enfriamiento_s * 12.4))
    for i, p in enumerate(puntajes):
        if p > umbral:
            racha += 1
            if racha >= consecutivos and (i - ultimo) > frames_enfriamiento:
                return True
        else:
            racha = 0
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modelo", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--manifiesto", required=True)
    p.add_argument("--consecutivos", type=int, default=3)
    p.add_argument("--audio", default=None,
                   help="carpeta de WAV para validar el clip (detecta repetición de Chatterbox)")
    p.add_argument("--span-max", type=float, default=3.5,
                   help="segundos de voz máximos razonables para una frase de dos palabras")
    p.add_argument("--salida", default=None)
    args = p.parse_args(argv)

    sesion = ort.InferenceSession(args.modelo, providers=["CPUExecutionProvider"])
    features = Path(args.features)

    maximos: list[float] = []
    puntajes_por_clip: list[np.ndarray] = []
    descartes: list[dict] = []
    sin_features = 0
    for linea in Path(args.manifiesto).read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        fila = json.loads(linea)
        npy = features / (Path(fila["archivo"]).stem + ".npy")
        if not npy.exists():
            sin_features += 1
            continue
        if args.audio:
            # Chatterbox a veces entra en repetición y devuelve 5-7 s para una frase de dos
            # palabras. Puntuar eso mediría el fallo del TTS, no el del wakeword.
            import soundfile as sf

            wav = Path(args.audio) / fila["archivo"]
            if wav.exists():
                audio, sr = sf.read(wav, dtype="float32", always_2d=False)
                activo = np.flatnonzero(np.abs(audio) > 1e-4)
                span = (activo[-1] - activo[0] + 1) / sr if activo.size else 0.0
                if span > args.span_max or span < 0.6:
                    descartes.append({"archivo": fila["archivo"], "span_s": round(span, 2),
                                      "motivo": f"duración de voz fuera de rango ({span:.2f} s)"})
                    continue
        datos = np.load(npy)
        if datos.shape[1] != DIM:
            sin_features += 1
            continue
        puntajes = puntuar(sesion, datos)
        puntajes_por_clip.append(puntajes)
        maximos.append(float(puntajes.max()))

    if not maximos:
        raise SystemExit("no hay clips puntuados: revisa --features y --manifiesto")

    umbrales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"clips puntuados: {len(maximos)} | descartados: {len(descartes)} | "
          f"sin características: {sin_features}")
    for d in descartes[:5]:
        print(f"  descartado {d['archivo']}: {d['motivo']}")
    print(f"prob. máxima: mediana {np.median(maximos):.4f} | min {min(maximos):.4f} | "
          f"max {max(maximos):.4f}")
    print(f"\n{'umbral':>7} {'recall':>8} {'detectados':>11}")
    curva = {}
    for u in umbrales:
        detectados = sum(1 for s in puntajes_por_clip if hay_evento(s, u, args.consecutivos))
        curva[u] = detectados / len(puntajes_por_clip)
        print(f"{u:>7.2f} {curva[u]:>8.3f} {detectados:>11}")

    if args.salida:
        Path(args.salida).write_text(json.dumps({
            "modelo": args.modelo, "clips": len(maximos),
            "consecutivos": args.consecutivos,
            "prob_max_mediana": round(float(np.median(maximos)), 4),
            "prob_max_min": round(min(maximos), 4),
            "prob_max_max": round(max(maximos), 4),
            "recall_por_umbral": {f"{u:.2f}": round(r, 4) for u, r in curva.items()},
            "descartes": descartes,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"escrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
