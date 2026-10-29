#!/usr/bin/env python3
"""Aumentación de canal sobre los positivos sintéticos: lo que hace que un TTS suene a humano.

El problema medido: un modelo entrenado con TTS acierta 98-99% con otros TTS y **0% con una voz
humana real** (10 grabaciones, micrófono de laptop). El embedding está congelado, así que la única
forma de moverlo es mover el audio. Aquí se le aplica a cada positivo lo que diferencia a una
grabación real de una síntesis limpia:

  - **Ganancia**: de -30 a -6 dBFS de pico (una laptop graba a -40 RMS; un móvil, a -20).
  - **Respuesta de micrófono**: paso alto (80-160 Hz), paso bajo (5-8 kHz) y una inclinación
    espectral aleatoria, que es la diferencia de timbre entre micrófonos baratos y un TTS.
  - **Reverberación**: convolución con una respuesta al impulso sintética (RT60 0,15-0,7 s) al
    10-35% de mezcla, para simular sala.
  - **Ruido de fondo** (opcional): recortes de MUSAN a distintas SNR.

Cada clip genera `--variantes` versiones. Se escribe un manifiesto con los parámetros de cada una
para poder auditar cualquier muestra.

Uso:
  python3 pipeline/aumentar_positivos.py --audio ~/datos-cv/positivos/kokoro_pad \
      --salida ~/datos-cv/positivos/aug_kokoro --variantes 3 --musan ~/datos/musan/musan
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

SR = 16_000


def respuesta_impulso(rng: random.Random, rt60: float) -> np.ndarray:
    """RIR sintética: ruido con decaimiento exponencial (Schroeder)."""
    n = int(rt60 * SR)
    ruido = np.random.RandomState(rng.randrange(1 << 30)).randn(n)
    env = np.exp(-6.9 * np.arange(n) / n)
    rir = ruido * env
    return rir / (np.abs(rir).sum() + 1e-9)


def ecualizar(x: np.ndarray, sr: int, corte_alto: float, corte_bajo: float,
              inclinacion: float, rng: random.Random) -> np.ndarray:
    """Paso alto + paso bajo + inclinación espectral suave (perfil de micrófono)."""
    sos_hp = butter(2, corte_alto / (sr / 2), btype="highpass", output="sos")
    sos_lp = butter(2, min(0.99, corte_bajo / (sr / 2)), btype="lowpass", output="sos")
    y = sosfilt(sos_lp, sosfilt(sos_hp, x))
    if abs(inclinacion) > 0.5:
        # Inclinación espectral: un polo (lfilter) y mezcla con la señal, no sosfilt, que
        # espera secciones de segundo orden (shape n,6) y aquí es de primer orden.
        fc = rng.choice([300.0, 800.0, 2000.0])
        alpha = np.exp(-2 * np.pi * fc / sr)
        graves = lfilter([1 - alpha], [1, -alpha], y)
        gan = 10 ** (inclinacion / 20)
        y = y + (gan - 1) * (graves - y) * 0.5
    return y.astype(np.float32)


def sumar_ruido(x: np.ndarray, ruido: np.ndarray, snr_db: float) -> np.ndarray:
    """Mezcla ruido a la SNR pedida, calculada sobre la energía de la señal."""
    if len(ruido) < len(x):
        ruido = np.tile(ruido, int(np.ceil(len(x) / len(ruido))))
    r = ruido[:len(x)]
    e_sig = np.mean(x[np.abs(x) > 1e-4] ** 2) if np.any(np.abs(x) > 1e-4) else np.mean(x ** 2)
    e_rui = np.mean(r ** 2) + 1e-12
    escala = np.sqrt(max(e_sig, 1e-9) / (e_rui * 10 ** (snr_db / 10)))
    return (x + r * escala).astype(np.float32)


def cargar_ruidos(musan: Path | None, limite: int = 200) -> list[np.ndarray]:
    """Recortes de 5 s de MUSAN (noise + music). Si no hay MUSAN, devuelve lista vacía."""
    if not musan or not musan.exists():
        return []
    import subprocess

    archivos = sorted(musan.rglob("*.wav"))
    if not archivos:
        return []
    salida = []
    for ruta in archivos[:limite]:
        try:
            x, sr = sf.read(ruta, dtype="float32", frames=SR * 5, always_2d=False)
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly

            x = resample_poly(x, SR, sr).astype(np.float32)
        if len(x) > SR:
            salida.append(x)
    return salida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--variantes", type=int, default=3)
    p.add_argument("--musan", default=None)
    p.add_argument("--semilla", type=int, default=20261026)
    args = p.parse_args(argv)

    rng = random.Random(args.semilla)
    origen = Path(args.audio)
    destino = Path(args.salida)
    destino.mkdir(parents=True, exist_ok=True)

    ruidos = cargar_ruidos(Path(args.musan) if args.musan else None)
    print(f"ruidos de fondo disponibles: {len(ruidos)}")

    archivos = sorted(origen.glob("*.wav"))
    manifiesto = destino / "manifiesto.jsonl"
    hechos = 0
    with manifiesto.open("w", encoding="utf-8") as mf:
        for ruta in archivos:
            x, sr = sf.read(ruta, dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
                continue
            for v in range(args.variantes):
                y = x.copy()
                rt60 = rng.uniform(0.15, 0.70)
                mojado = rng.uniform(0.10, 0.35)
                y = (1 - mojado) * y + mojado * fftconvolve(y, respuesta_impulso(rng, rt60))[:len(y)]
                y = ecualizar(y, sr, rng.uniform(80, 160), rng.uniform(5000, 8000),
                              rng.uniform(-4, 4), rng)
                snr = None
                if ruidos and rng.random() < 0.85:
                    snr = rng.uniform(-5, 20)
                    y = sumar_ruido(y, rng.choice(ruidos), snr)
                pico = np.abs(y).max() or 1.0
                objetivo = 10 ** (rng.uniform(-30, -6) / 20)
                y = (y / pico * objetivo).astype(np.float32)

                nombre = f"{ruta.stem}_aug{v}.wav"
                sf.write(destino / nombre, y, sr, subtype="PCM_16")
                mf.write(json.dumps({"archivo": nombre, "origen": ruta.name, "rt60": round(rt60, 3),
                                     "mojado": round(mojado, 3), "snr_db": None if snr is None
                                     else round(snr, 1),
                                     "pico_dbfs": round(float(objetivo), 4)}) + "\n")
                hechos += 1
            if hechos % 500 == 0:
                print(f"  {hechos} clips aumentados", flush=True)
    print(f"LISTO: {hechos} clips en {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
