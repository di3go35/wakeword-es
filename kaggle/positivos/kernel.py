"""Genera positivos de 'hey celeste' con Chatterbox en la GPU de Kaggle.

Se ejecuta como *script kernel* (`kaggle kernels push`), no localmente. Todo lo que produce va a
/kaggle/working/ para poder descargarlo con `kaggle kernels output`.

Por qué en Kaggle: en el VPS un clip tarda ~3 min en CPU; en una T4 tarda 1-2 s. Eso permite
cientos de clips con variación de prosodia en lugar de decenas.

Salida:
  /kaggle/working/clips/cb_XXXX.wav   WAV 16 kHz mono
  /kaggle/working/clips/manifiesto.jsonl
"""

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Chatterbox descarta ~72% de sus salidas en español (frase repetida, demasiado larga o
# corta) y los parámetros no lo cambian: se pide 4x para quedarse con ~800 limpios.
TOTAL = int(os.environ.get("TOTAL_CLIPS", "3000"))
FRASE = "hey celeste"
# Sin puntos suspensivos: la variante "hey... celeste" era la que más empujaba a Chatterbox a
# repetir la frase (132 de 434 clips salieron duplicados).
VARIANTES = ("hey celeste", "Hey, Celeste.", "¡Hey Celeste!", "Hey celeste.")
TEMPERATURA = (0.45, 0.75)   # más baja = menos divagación
SR_SALIDA = 16_000
SEMILLA = 20261025


def segmentos_voz(x, sr):
    """Segmentos con voz: RMS por tramas de 25 ms, umbral -40 dBFS, fusiona huecos < 0,6 s."""
    import numpy as np

    n, hop = int(0.025 * sr), int(0.010 * sr)
    if len(x) < n:
        return []
    t = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    db = 20 * np.log10(np.sqrt((t ** 2).mean(axis=1) + 1e-12))
    act, segs, ini = db > -40.0, [], None
    for i, a in enumerate(act):
        if a and ini is None:
            ini = i
        elif not a and ini is not None:
            segs.append([ini * hop / sr, i * hop / sr])
            ini = None
    if ini is not None:
        segs.append([ini * hop / sr, len(act) * hop / sr])
    fus = []
    for s in segs:
        if fus and s[0] - fus[-1][1] < 0.60:
            fus[-1][1] = s[1]
        else:
            fus.append(s)
    return [(a, b) for a, b in fus if b - a >= 0.15]


def instalar() -> None:
    """chatterbox-tts no viene en la imagen de Kaggle.

    Trampa de la imagen: trae `torchvision` compilado contra otra versión de `torch`, y al mover
    `torch` el operador `torchvision::nms` deja de existir. Eso rompe el import de `transformers`
    (`Could not import module 'LlamaModel'`) y el kernel muere antes de generar nada.
    En el VPS no pasa porque allí torchvision no está instalado: aquí se desinstala.
    """
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "chatterbox-tts", "soundfile"],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"],
                   check=False, capture_output=True)


def main() -> int:
    instalar()

    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    print("GPU disponible:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # comprobación explícita: si el entorno está roto, que falle con un mensaje legible
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except Exception as e:
        print("FALLO AL IMPORTAR CHATTERBOX:", repr(e))
        raise

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    modelo = ChatterboxMultilingualTTS.from_pretrained(device=dispositivo)
    print(f"modelo cargado en {time.time() - t0:.0f}s")

    salida = Path("/kaggle/working/clips")
    salida.mkdir(parents=True, exist_ok=True)
    manifiesto = salida / "manifiesto.jsonl"
    descartes = salida / "descartes.jsonl"
    dfile = descartes.open("w", encoding="utf-8")   # un Path no tiene .write()
    rng = random.Random(SEMILLA)

    generados = 0
    descartados = 0
    with manifiesto.open("w", encoding="utf-8") as mf:
        for i in range(TOTAL):
            variante = FRASE if i == 0 else rng.choice(VARIANTES)
            exageracion = round(rng.uniform(0.25, 0.85), 2)
            cfg = round(rng.uniform(0.25, 0.7), 2)
            temperatura = round(rng.uniform(*TEMPERATURA), 2)

            inicio = time.time()
            wav = modelo.generate(variante, language_id="es", exaggeration=exageracion,
                                  cfg_weight=cfg, temperature=temperatura)
            audio = wav.squeeze().cpu().numpy().astype("float32")

            # Validación estricta: UNA sola frase (fusionando huecos < 0,6 s) y 0,9-2,0 s de voz.
            # Contar clips no sirve: lo que manda es cuántos sirven.
            segs = segmentos_voz(audio, modelo.sr)
            voz = sum(b - a for a, b in segs)
            motivo = ""
            if len(segs) == 0:
                motivo = "sin voz detectable"
            elif len(segs) > 1:
                motivo = f"frase repetida ({len(segs)} segmentos)"
            elif voz < 0.9:
                motivo = f"voz demasiado corta ({voz:.2f} s)"
            elif voz > 2.0:
                motivo = f"voz demasiado larga ({voz:.2f} s)"
            if motivo:
                dfile.write(json.dumps({"variante": variante, "temperatura": temperatura,
                                        "motivo": motivo}, ensure_ascii=False) + "\n")
                dfile.flush()
                descartados += 1
                continue

            audio = audio / (float(np.max(np.abs(audio))) or 1.0) * 0.6
            audio = resample_poly(audio, SR_SALIDA, modelo.sr).astype("float32")
            objetivo = rng.uniform(2.2, 3.2)
            falta = max(0.0, objetivo - len(audio) / SR_SALIDA)
            adelante = rng.uniform(0.15, 0.55) * falta
            audio = np.concatenate([
                np.zeros(int(adelante * SR_SALIDA), dtype="float32"), audio,
                np.zeros(int((falta - adelante) * SR_SALIDA), dtype="float32"),
            ])

            nombre = f"cb_{generados:04d}.wav"
            sf.write(salida / nombre, audio, SR_SALIDA, subtype="PCM_16")
            mf.write(json.dumps({
                "archivo": nombre, "motor": "chatterbox", "variante": variante,
                "exageracion": exageracion, "cfg": cfg, "temperatura": temperatura,
                "segundos": round(len(audio) / SR_SALIDA, 3),
                "voz_s": round(voz, 3),
                "rachas_finas": len(segs),
            }, ensure_ascii=False) + "\n")
            mf.flush()
            generados += 1
            if generados % 25 == 0:
                print(f"{generados} válidos ({descartados} descartados) | "
                      f"{time.time() - inicio:.1f}s el último", flush=True)

    dfile.close()
    print(f"LISTO: {generados} válidos de {TOTAL} pedidos, {descartados} descartados")
    print(f"tasa de descarte: {descartados / TOTAL:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
