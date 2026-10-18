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

# 0. Descargar el corpus (reanudable, verifica sha256 contra el que publica la API)
MDC_API_KEY=... pipeline/descargar_corpus.sh

# 1. Extraer los .tsv (van al principio del archivo: no hace falta descomprimir 52 GB)
timeout 150 tar -xzf ~/datos-cv/common-voice-scripted-speech-27-0-spanis.tar.gz \
    -C ~/datos-cv/corpus --wildcards 'cv-corpus-27.0-2026-09-11/es/*.tsv'

# 2. Unir duraciones reales (clip_durations.tsv viene en milisegundos)
python3 pipeline/preparar_validado.py \
    --dir ~/datos-cv/corpus/cv-corpus-27.0-2026-09-11/es \
    --salida ~/datos-cv/validado_con_duraciones.tsv

# 3. Planificar lotes por horas (filtrando por acento si interesa)
python3 pipeline/planificar_lotes.py \
    --tsv ~/datos-cv/validado_con_duraciones.tsv --salida ~/datos-cv/lotes \
    --horas-por-lote 250 --acentos "andino-pacífico"

# 4. Extraer + calcular características + borrar clips, por lote y en paralelo
pipeline/procesar_lote.sh 000 3
```

El paso 4 es el que se lanza en segundo plano: extrae los clips del lote en una sola pasada
sobre el archivo, reparte el cálculo de características entre 3 procesos y borra los clips al
terminar. Reanudable de punta a punta.

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
