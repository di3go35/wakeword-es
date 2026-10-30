# ESTADO — documento de traspaso

Última actualización: **2026-10-29**. Si eres una sesión nueva, lee esto primero y luego
`PLAN.md`, `DECISIONES.md` (44 filas) y `docs/resultados.md`.

## Qué es el proyecto, en tres frases

Pipeline abierto (Apache-2.0) para entrenar wake words en **español latinoamericano** con falsos
positivos medidos y procedencia de datos auditable. El activo no es el código: es el dato y la
medición. La frase congelada es **`hey celeste`**.

## Dónde está todo

| Qué | Dónde |
|---|---|
| Repo | `~/wakeword-es` → `github.com/di3go35/wakeword-es` (público) |
| Datos (fuera del repo) | `~/datos-cv/` — 9,5 GB de características, 1,5 GB de audio de test, positivos, tensores |
| Ruido | `~/datos/musan/` (687 MB de ruido + 4,6 GB de música) |
| Venv del pipeline | `~/.hermes/tools/venv-wakeword/.venv311` (Python 3.11: openwakeword 0.6.0, onnxruntime, torch CPU, kokoro, onnx, onnxscript) |
| Venv TTS alterno | `~/.hermes/tools/venv-chatterbox` (chatterbox-tts 0.1.7, torch 2.6 cu124) |
| Venv ASR | `~/.hermes/tools/venv-asr` (faster-whisper) |
| CLI Kaggle | `~/.hermes/tools/venv-kaggle/bin/kaggle` — usuario **diegor35** (no `di3go35`) |
| Kernel de Kaggle | `diegor35/wakeword-chatterbox-positivos` (script kernel, GPU, internet) — 12,69 h de 30 usadas |
| Avisos WhatsApp | `pipeline/avisar.sh` (bridge en `127.0.0.1:3000`) + cron horario |

**Los venvs son de `uv`: no tienen `pip`.** Instalar con
`~/.hermes/bin/uv pip install --python <venv>/bin/python <paquete>`.

## Fase actual, según `PLAN.md`

| Fase | Estado |
|---|---|
| 1. Aprobación | Hecha |
| 2. Auditoría y línea base | **Hecha** (baseline 0,773 → 0,019 FA/h tras corregir el etiquetado) |
| 3. Frase por medición | **Hecha** (`FRASE.md`, hash `f2a3f422…`) |
| 4. Corpus y procedencia | **Hecha** salvo consentimientos de terceros y ruido propio grabado |
| 5. Test congelado | **Parcial**: negativos reales 53,03 h congelados y hasheados; falta test público 100% sintético y la regla de cebado documentada en el banco |
| 6. Modelo y anti-disparo | **Parcial**: cabeza + compuerta RMS + frames consecutivos + enfriamiento + `calibrar`; falta VAD |
| 7. Minado de falsos positivos | **Hecha** (delta medido: 4,469 → 0,094 FA/h) |
| 8. Checkpoint publicable | Pendiente |
| 9. Demo Android | Pendiente (guía escrita en `docs/integracion-android.md`) |
| 10-11. Publicación / candidatos A-B | Pendiente |

## Números vigentes (y de dónde salen)

Modelo: `~/datos-cv/entrenamiento/cabeza_fewshot.onnx` — 196.865 parámetros, ventana de 16 frames.

| Métrica | Valor | Fuente |
|---|---|---|
| Recall limpio | 100% (10/10) | `docs/resultados.md` |
| Recall campo lejano 3 m (simulado) | 100% (10/10) | ídem |
| **Recall ruido (regla operativa: 3 frames sobre 0,9)** | **66% (53/80)** | ídem — **es la condición que no cumple** |
| Recall cero-shot (solo TTS) | **0/10** | `docs/hallazgo-voz-humana.md` |
| Falsos positivos @0,9 con compuerta −45 dBFS | **0,094 FA/h** (5 eventos / 53,03 h) | `docs/resultados.md` |
| Falsos positivos @0,5 | 0,396 FA/h | ídem |
| Silencio digital 30 min | 0 activaciones | ídem |
| "celeste" en frase normal | 0 de 30 disparan | ídem |

**El recall del 66% en ruido no mejora bajando el umbral (68,8% a 0,5) ni con compuerta adaptativa.**
La causa medida: 10 grabaciones reales de una sola persona no cubren la variabilidad de canal.

## Qué se puede rehacer tal cual (comandos)

```bash
cd ~/wakeword-es
make test                                    # 12 pruebas del banco de evaluación
make autotest                                # el criterio falla cuando debe fallar
P=~/.hermes/tools/venv-wakeword/.venv311/bin/python
$P pipeline/entrenar_fewshot.py --reales-npy ~/datos-cv/entrenamiento/pos_reales_fewshot.npy \
   --datos ~/datos-cv/entrenamiento --epochs 15 \
   --neg-dificiles ~/datos-cv/entrenamiento/neg_dificiles_all.npy --peso-dificiles 3 \
   --neg-extra ~/datos-cv/features_negativos_frase
$P pipeline/medir_test.py --modelo ~/datos-cv/entrenamiento/cabeza_fewshot.onnx \
   --features ~/datos-cv/features --tsv ~/datos-cv/validado_con_duraciones.tsv \
   --clips ~/datos-cv/test/clips.txt --umbral 0.9
$P pipeline/calibrar.py --features ~/datos-cv/features --clips <lista> --horas <h> --objetivo 0.5
```

## Convención obligatoria: jobs largos

Los procesos lanzados como *segundo plano de la sesión* **mueren cuando se cierra la app** (el
gateway apaga la sesión a los 2 minutos de inactividad). Todo lo largo va como unidad de systemd de
usuario, que sobrevive porque `Linger=yes`:

```bash
systemd-run --user --unit=<nombre> --collect -p MemoryMax=4G -p CPUQuota=300% -p Nice=10 \
  /bin/bash -c '<comando> > ~/datos-cv/logs/<nombre>.log 2>&1; echo LISTO >> ~/datos-cv/logs/<nombre>.log'
```

Y **no** encadenar con `| tail`: el código de salida pasa a ser el de `tail` y el fallo se pierde.
Redirigir a un log y revisar el log.

## Trampas que costaron tiempo (no repetirlas)

1. **16 frames = 2,04 s de audio, no 1,28 s.** El embedding consume 76 frames de mel (0,76 s) y avanza
   8. Un clip de menos de ~2 s no produce ventana: hay que rellenarlo o no se puede evaluar.
2. **Etiquetar todas las ventanas de un clip positivo** enseña que media frase basta: 0,773 FA/h frente
   a 0,019. Solo es positivo si la frase completa cabe en la ventana.
3. **Kokoro en español pronuncia a la castellana** (`celeste` → `θeleste`). Forzar
   `EspeakG2P(language='es-419')` para seseo.
4. **Kaggle: `torch` y `torchvision` están compilados en pareja.** Instalar algo que mueva `torch` rompe
   `torchvision::nms` y aparece disfrazado de `ModuleNotFoundError: LlamaModel`. Solución:
   `pip uninstall -y torchvision` en el kernel si no se usa visión.
5. **`kaggle kernels output` pagina de 20 en 20** y el CDN devuelve 502 a mitad. Usar
   `--page-size 200` y reintentar.
6. **Un `Path` no tiene `.write()`**: usar `.open()`.
7. **Aumentaciones con el mismo nombre de archivo se pisan** (dos corridas con distinta semilla y mismo
   `--variaciones` generan `diego_01_aug0.wav` en las dos).
8. **Nunca publicar un recall sin declarar la regla de activación.** El mismo conjunto dio 100%
   (mediana de puntuación) y 66% (3 frames sobre 0,9). Decisión 43.
9. **Chatterbox en español descarta 60-72% de sus salidas** (frase repetida, truncada, fragmentos). Hay
   que filtrar por un solo segmento de voz, 0,9-2,0 s y ≤2 rachas finas.
10. **No medir el test como validación.** Las 53 h se han mirado ~10 veces: es un **test de desarrollo**.

## Próximos tres pasos, en orden

1. **Más voz real** (5 personas × 3 sesiones, con consentimiento). Es lo único que puede subir el 66%
   de recall en ruido: ni umbral ni ruido sintético lo mueven.
2. **Fase 5.1**: test público 100% sintético con disjunción por motor, para que un tercero reproduzca
   el número con `make benchmark`.
3. **Fase 9**: demo Android con paridad numérica Python ↔ Android frame por frame
   (`docs/integracion-android.md` tiene los tres tropezones ya documentados).
