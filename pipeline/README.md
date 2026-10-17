# pipeline/

De audio a características, por lotes, sin llenar el disco.

## Por qué por lotes

El corpus de Common Voice es 27.0 son ~52 GB comprimidos y **597,6 h** de audio validado.
El archivo y su extracción completa no caben a la vez en el servidor (96 GB de disco, 71
libres), así que el flujo es: **extraer ~100 h → calcular características → borrar esos
clips**. Cada lote deja su manifiesto y es reanudable: si algo se corta, se vuelve a lanzar
y salta lo ya hecho.

## Flujo

```bash
VENV=~/.hermes/tools/venv-wakeword/.venv311/bin/python   # Python 3.11 + openwakeword 0.6.0

# 1. Qué va en cada lote (a partir del validated.tsv del corpus)
python3 pipeline/planificar_lotes.py \
    --tsv ~/datos-cv/validated.tsv --salida ~/datos-cv/lotes --horas-por-lote 100

# 2. Extraer los clips del lote desde el archivo descargado
tar -xzf ~/datos-cv/common-voice-scripted-speech-27-0-spanis.tar.gz \
    -C ~/datos-cv --wildcards -T ~/datos-cv/lotes/lote_000.txt

# 3. Características del lote (mel + embedding)
$VENV pipeline/extraer_caracteristicas.py \
    --lista ~/datos-cv/lotes/lote_000.txt --audio-dir ~/datos-cv/clips \
    --salida ~/datos-cv/features --lote-id 000

# 4. Verificar el manifiesto del lote y borrar el audio del lote
```

## Mediciones reales

| Medición | Valor |
|---|---|
| Extracción de características (4 vCPU, onnxruntime CPU) | **28,7×** tiempo real en tanda larga |
| Mismo extractor en clips cortos | 41,6× (la cifra que manda es la de tanda larga) |
| Vector por ventana | **96 dimensiones**, ~12,4 ventanas/s |
| Tamaño de las características | ~8 MB por hora de audio (16 bits) |
| 597,6 h | ≈ **21 h de CPU** en serie, ~5 h con cuatro procesos |

## Reglas que este pipeline no rompe

- **El audio no sale del servidor.** Common Voice prohíbe re-hospedar y re-compartir; a
  Kaggle solo irían características derivadas, en dataset privado.
- **Nada se borra sin manifiesto.** Cada `.npy` tiene su fila con la ruta del clip, su
  duración, la forma del vector y el `sha256` del audio decodificado: sin eso no se puede
  auditar de dónde salió cada característica.
- **Un clip corrupto no tumba el lote**: se cuenta como error y el lote sigue.

## Pruebas

```bash
python3 pipeline/test_planificar_lotes.py   # 7 pruebas del planificador (solo stdlib)
```

El extractor se prueba contra MP3 sintéticos generados con ffmpeg: primera corrida procesa,
segunda corrida salta (reanudable). Ver `docs/PASO_A_PASO_FASE2.md`.
