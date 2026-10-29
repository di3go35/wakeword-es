#!/usr/bin/env python3
"""Capa anti-disparo: compuerta de energía + frames consecutivos + enfriamiento.

Es la fase 6.1 del plan. Tres piezas, en este orden:

1. **Compuerta de energía (RMS)**: si la energía del último tramo de audio está por debajo de un
   piso, la puntuación se fuerza a 0. Sirve para el caso que el plan señala como síntoma clásico:
   en silencio el mel normaliza ruido térmico y el modelo "oye" algo que no existe.
2. **Frames consecutivos**: una activación exige N frames seguidos sobre el umbral, no un pico.
3. **Enfriamiento**: tras una activación, un tiempo mínimo antes de contar otra.

El riesgo declarado en el plan: un piso demasiado alto **rompe la detección en campo lejano**, donde
la señal llega atenuada. Por eso el módulo se mide en las dos direcciones: falsos positivos que quita
y recall que cuesta.

Alineación: el frame k del embedding cubre el audio [0,08·k, 2,04 + 0,08·k]. Para la compuerta se usa
la energía de la ventana más reciente, [0,08·k, 0,08·(k+1)], que es lo que un detector en streaming
tiene disponible en ese instante.
"""

from __future__ import annotations

import numpy as np

SR = 16_000
HOP_S = 0.08


def energia_por_frame(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """RMS en dBFS por cada 80 ms de audio, alineado con los frames del embedding."""
    hop = int(HOP_S * sr)
    if len(audio) < hop:
        return np.zeros(0, dtype=np.float32)
    n = (len(audio) // hop) * hop
    tramas = audio[:n].reshape(-1, hop).astype(np.float64)
    rms = np.sqrt((tramas ** 2).mean(axis=1) + 1e-12)
    return (20 * np.log10(rms)).astype(np.float32)


def aplicar_compuerta(puntajes: np.ndarray, energias: np.ndarray, piso_db: float) -> np.ndarray:
    """Fuerza a 0 las puntuaciones cuyo frame no tiene energía suficiente."""
    salida = puntajes.copy()
    n = min(len(salida), len(energias))
    if n == 0:
        return salida
    silenciosos = energias[:n] < piso_db
    salida[:n][silenciosos] = 0.0
    return salida


def piso_adaptativo(energias: np.ndarray, margen_db: float = 8.0, piso_min: float = -60.0,
                    ventana_frames: int = 37) -> np.ndarray:
    """Piso por frame: el suelo de ruido local más un margen.

    Un piso fijo de -45 dBFS funciona en una habitación silenciosa y **rompe** la detección cuando
    el clip entero viene bajo (medido: recall en ruido 100% → 66%). El piso correcto depende del
    ruido de fondo: en silencio el suelo está en -70 y la compuerta debe cortar ahí; en una sala con
    ventilador el suelo está en -35 y cortar en -45 silenciaría la propia voz.

    Se estima el suelo local como el percentil 10 dentro de una ventana deslizante de ~3 s y se le
    suma un margen. Nunca baja de `piso_min` (silencio digital).
    """
    if len(energias) == 0:
        return energias
    v = min(ventana_frames, len(energias))
    ventanas = np.lib.stride_tricks.sliding_window_view(
        np.pad(energias, (v // 2, v - 1 - v // 2), mode="edge"), v)
    suelo = np.percentile(ventanas, 10, axis=1)
    return np.maximum(suelo + margen_db, piso_min).astype(np.float32)


def aplicar_compuerta_adaptativa(puntajes: np.ndarray, energias: np.ndarray,
                                 margen_db: float = 8.0, piso_min: float = -60.0) -> np.ndarray:
    pisos = piso_adaptativo(energias, margen_db, piso_min)
    salida = puntajes.copy()
    n = min(len(salida), len(pisos))
    if n:
        salida[:n][energias[:n] < pisos[:n]] = 0.0
    return salida


def contar_eventos(puntajes: np.ndarray, umbral: float, consecutivos: int = 3,
                   enfriamiento_s: float = 1.0, fps: float = 12.4) -> int:
    """Cuenta activaciones con la regla de frames consecutivos y enfriamiento."""
    eventos, racha, ultimo = 0, 0, -10**9
    frames_enfriamiento = max(1, int(enfriamiento_s * fps))
    for i, p in enumerate(puntajes):
        if p > umbral:
            racha += 1
            if racha >= consecutivos and (i - ultimo) > frames_enfriamiento:
                eventos += 1
                ultimo = i
                racha = 0
        else:
            racha = 0
    return eventos


def evaluar_clip(puntajes: np.ndarray, audio: np.ndarray | None, umbral: float,
                 consecutivos: int = 3, enfriamiento_s: float = 1.0,
                 piso_db: float | None = None) -> tuple[int, float, float]:
    """Devuelve (eventos, probabilidad máxima, energía mínima de los frames que dispararon)."""
    energias = energia_por_frame(audio) if audio is not None else None
    if energias is not None and piso_db is not None:
        puntajes = aplicar_compuerta(puntajes, energias, piso_db)
    eventos = contar_eventos(puntajes, umbral, consecutivos, enfriamiento_s)
    maximo = float(puntajes.max()) if len(puntajes) else 0.0
    minimo = float(energias.min()) if energias is not None and len(energias) else 0.0
    return eventos, maximo, minimo
