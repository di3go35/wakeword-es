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

TOTAL = int(os.environ.get("TOTAL_CLIPS", "600"))
FRASE = "hey celeste"
VARIANTES = ("hey celeste", "Hey, Celeste.", "hey... celeste", "¡Hey Celeste!", "Hey celeste.")
SR_SALIDA = 16_000
SEMILLA = 20261025


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
    rng = random.Random(SEMILLA)

    generados = 0
    descartados = 0
    with manifiesto.open("w", encoding="utf-8") as mf:
        for i in range(TOTAL):
            variante = FRASE if i == 0 else rng.choice(VARIANTES)
            exageracion = round(rng.uniform(0.25, 0.85), 2)
            cfg = round(rng.uniform(0.25, 0.7), 2)
            temperatura = round(rng.uniform(0.55, 0.9), 2)

            inicio = time.time()
            wav = modelo.generate(variante, language_id="es", exaggeration=exageracion,
                                  cfg_weight=cfg, temperature=temperatura)
            audio = wav.squeeze().cpu().numpy().astype("float32")

            # Chatterbox a veces entra en repetición: "hey celeste" de 6 s es basura, no frase.
            activo = np.flatnonzero(np.abs(audio) > 1e-4)
            span = (activo[-1] - activo[0] + 1) / modelo.sr if activo.size else 0.0
            if span > 3.5 or span < 0.6:
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
            }, ensure_ascii=False) + "\n")
            mf.flush()
            generados += 1
            if generados % 25 == 0:
                print(f"{generados} válidos ({descartados} descartados) | "
                      f"{time.time() - inicio:.1f}s el último", flush=True)

    print(f"LISTO: {generados} válidos, {descartados} descartados por repetición")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
