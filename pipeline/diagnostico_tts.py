#!/usr/bin/env python3
"""Control negativo: ¿el modelo detecta la frase o detecta que el audio es de TTS?

Si un modelo entrenado con positivos sintéticos puntúa alto también en **otra frase dicha por la
misma voz**, no aprendió la frase: aprendió la huella del sintetizador. Es el fallo más común y el
más silencioso de los wakewords entrenados con TTS, porque todas las métricas de validación salen
perfectas.

Se genera un conjunto de frases de control con las mismas voces, se calculan sus características
con el mismo extractor y se puntúan con la cabeza entrenada.

Uso:
  python3 pipeline/diagnostico_tts.py --modelo ~/datos-cv/entrenamiento/cabeza.onnx
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

FRASES_CONTROL = ("hola mundo", "buenos días", "la casa azul", "mañana temprano")
VOCES = ("ef_dora", "em_alex", "em_santa")
REPETICIONES = 3
FEAT_DIM = 96


def sintetizar(salida: Path) -> list[Path]:
    from kokoro import KPipeline
    from misaki.espeak import EspeakG2P

    tuberia = KPipeline(lang_code="e")
    tuberia.g2p = EspeakG2P(language="es-419")
    salida.mkdir(parents=True, exist_ok=True)
    rutas = []
    for frase in FRASES_CONTROL:
        for voz in VOCES:
            for i in range(REPETICIONES):
                for velocidad in (0.92, 1.0, 1.08):
                    trozos = [a for _, _, a in tuberia(frase, voice=voz, speed=velocidad)]
                    audio = np.concatenate([t.numpy() for t in trozos]).astype(np.float32)
                    objetivo = 2.6 * 24_000
                    falta = max(0, int(objetivo - len(audio)))
                    audio = np.concatenate([np.zeros(falta // 3, dtype=np.float32), audio,
                                            np.zeros(falta - falta // 3, dtype=np.float32)])
                    # separador "__" entre frase y metadatos: con "_" simple el nombre no se
                    # puede volver a parsear (la frase tiene espacios)
                    ruta = salida / (frase.replace(" ", "-") + f"__{voz}__{i}__{velocidad}.wav")
                    sf.write(ruta, audio, 24_000, subtype="PCM_16")
                    rutas.append(ruta)
    return rutas


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modelo", required=True)
    p.add_argument("--dir", default=str(Path.home() / "datos-cv/diagnostico"))
    p.add_argument("--extractor", default=str(Path(__file__).parent / "extraer_caracteristicas.py"))
    args = p.parse_args(argv)

    base = Path(args.dir)
    audio_dir, feat_dir = base / "audio", base / "features"
    print("sintetizando frases de control con las mismas voces...")
    rutas = sintetizar(audio_dir)
    print(f"  {len(rutas)} clips de control")

    lista = base / "lista.txt"
    lista.write_text("\n".join(r.name for r in rutas) + "\n", encoding="utf-8")
    print("calculando características...")
    subprocess.run([sys.executable, args.extractor, "--lista", str(lista),
                    "--audio-dir", str(audio_dir), "--salida", str(feat_dir),
                    "--lote-id", "diagnostico"],
                   check=True, capture_output=True)

    import onnxruntime as ort

    sesion = ort.InferenceSession(args.modelo, providers=["CPUExecutionProvider"])
    ventana = 16
    resultados: dict[str, list[float]] = {}
    for ruta in rutas:
        npy = feat_dir / (ruta.stem + ".npy")
        if not npy.exists():
            continue
        datos = np.load(npy)
        if datos.shape[0] < ventana or datos.shape[1] != FEAT_DIM:
            continue
        w = np.lib.stride_tricks.sliding_window_view(datos, (ventana, FEAT_DIM))[:, 0]
        salida = sesion.run(None, {"input": np.ascontiguousarray(w, dtype=np.float32)})[0].ravel()
        prob = 1 / (1 + np.exp(-salida))
        frase = ruta.stem.split("__")[0].replace("-", " ")
        resultados.setdefault(frase, []).append(float(prob.max()))

    print(f"\n{'frase de control':<20} {'clips':>6} {'prob media':>11} {'prob máx':>9} {'>0,5':>6}")
    resumen = {}
    for frase, valores in sorted(resultados.items()):
        alto = sum(1 for v in valores if v > 0.5)
        resumen[frase] = {"clips": len(valores), "media": round(float(np.mean(valores)), 4),
                          "max": round(float(np.max(valores)), 4), "sobre_umbral": alto}
        print(f"{frase:<20} {len(valores):>6} {np.mean(valores):>11.4f} "
              f"{np.max(valores):>9.4f} {alto:>6}")

    (base / "resultado.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"\nescrito en {base / 'resultado.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
