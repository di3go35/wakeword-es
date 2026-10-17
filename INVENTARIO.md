# INVENTARIO — piezas, evidencia faltante, descartes y afirmaciones auditadas

## 1. Piezas verificadas que existen hoy (con su papel)

| Pieza | Qué es | Papel en el proyecto | Referencia |
|---|---|---|---|
| openWakeWord (dscripka) | Framework Apache-2.0, backbone congelado + cabeza; modelos preentrenados CC BY-NC-SA 4.0 | Línea base a batir; fuente del candidato C | github.com/dscripka/openWakeWord |
| livekit-wakeword | Fork con cabeza Conv-Attention; documenta el techo multilingüe del backbone | Evidencia de la limitación arquitectónica; fuente del candidato B | github.com/livekit/livekit-wakeword |
| microWakeWord (Kevin Ahrendt) | CNN de streaming (MixConv), objetivo FA/h, TFLite uint8 de 50–300 KB | Linaje del candidato A y del formato para ESP32-S3 | github.com/OHF-Voice/micro-wake-word |
| onnx-wakeword (voicute) | TCN causal ~25K params, <130 KB, 5 capas anti-falso-disparo, español entre 7 idiomas | Referencia del mecanismo de post-proceso; corrige la pretensión de originalidad | github.com/voicute/onnx-wakeword |
| interkelstar/microwakeword-trainer | Reporte de minado de falsos positivos: 91/140 → 1/140 FP, 94% recall | Evidencia del mayor salto medido; justifica el comando `minar-falsos-positivos` | github.com/interkelstar/microwakeword-trainer |
| Benchmark de Picovoice | Comparativa a 1 FA por 10 h, CPU en Raspberry Pi 5 | Método de reporte de métricas (curva, no punto) | picovoice.ai/docs/benchmark/wake-word/ |
| MOCKS (Samsung, Interspeech 2023) | Testset multilingüe de KWS de vocabulario abierto | Comparación externa opcional, sin depender del test propio | HF: voiceintelligenceresearch/MOCKS |
| Common Voice español 26.0 | CC0, MP3, **48,30 GB**; 1.680.810 clips, 2.278,66 h (595,45 h validadas), 26.940 hablantes; desglose por acento con **andino-pacífico: 38.728 clips / 947 hablantes**. Prohibido re-hospedar o re-compartir; usar solo localmente y derivar características | Negativos reales en español, limpios y gratuitos; permite evaluación estratificada por acento | dataset ID `cmqim2spa00synr071fcp7av0` en mozilladatacollective.com |
| MUSAN | CC BY 4.0 general, fuentes mixtas | Ruido/música para aumentación, con auditoría ítem por ítem | openslr.org/17 |
| Chatterbox Multilingual V3 + `es-mx-latam` | MIT, 0,5B, 23+ idiomas, clonación zero-shot, finetune dedicado es-419 | Diversidad de acentos y voces clonadas con consentimiento | HF: ResembleAI/Chatterbox-Multilingual-es-mx-latam |
| Kokoro-82M | Apache-2.0; español: `ef_dora`, `em_alex`, `em_santa` | Volumen de datos sintéticos | HF: hexgrad/Kokoro-82M |
| Paper de Google Research *Streaming Keyword Spotting on Mobile Devices* | Base de microWakeWord | Fundamento del candidato A | Ryabov et al. |

## 2. Evidencia que falta producir (y en qué fase)

| Evidencia | Fase | Cómo se verifica desde fuera |
|---|---|---|
| Curva DET de la línea base (números malos publicados) | 2 | Repo público + comando reproducible |
| Frase congelada con su justificación medida | 3 | `FRASE.md` + hash de commit |
| Registro de procedencia de cada fuente con licencia y hash | 4 | `PROCEDENCIA.md` |
| Consentimientos escritos de 10–15 personas | 4 | Referencias por hash (los PDF no se publican) |
| Tamaño y tiempo reales de extracción de embeddings | 4 | Medición con 1 h extrapolada |
| Test congelado con hashes publicados antes de entrenar | 5 | Hash en el repo antes del primer entrenamiento |
| Curva DET del modelo final vs línea base | 6 | Tabla comparativa en el repo |
| 30 min de silencio digital = 0 activaciones | 6 | Log del test |
| Delta del minado de falsos positivos | 7 | Reporte antes/después |
| Modelo cargable y reproducible por un tercero | 8 | URL + instrucciones |
| Consumo de batería (%/h) y latencia p95 en Android | 9 | Video + tabla con método |
| Paridad numérica Python ↔ Android | 9 | Test frame por frame sobre el mismo WAV |
| Métricas de adopción (descargas, modelos de terceros) | 10 | Datos públicos de Hugging Face / GitHub |

## 3. Descartados, con motivo

| Descartado | Motivo |
|---|---|
| F5-TTS | Pesos CC-BY-NC-4.0 (Emilia); el mantenedor confirma que sigue siendo no comercial tras fine-tuning |
| Fish Speech / OpenAudio S1 | Pesos CC-BY-NC-SA-4.0 y acuerdo escrito obligatorio; el código dejó de ser Apache |
| ChatTTS | CC BY-NC 4.0 en código y modelos |
| Zonos v0.1 | Apache-2.0, pero sin español soportado oficialmente y con calidad reportada como mala |
| Piper como dependencia | Motor GPL-3.0 en el fork activo; licencia por voz; la voz Lessac es Blizzard (solo investigación) |
| ESC-50, UrbanSound8K | CC BY-NC: no pueden entrar en un pipeline con ambición comercial |
| AudioSet / ACAV100M | Origen de la licencia NC de los modelos de openWakeWord |
| Reempaquetar el audio de Common Voice | Sus términos piden no replicarlo en otras plataformas |
| SDKs Rust/C++/Flutter/React Native, Cortex-M, nube/SaaS | Fuera del presupuesto de 120–200 h; declarados fuera de alcance |
| Rediseñar la arquitectura "por fe" antes de medir | Sustituido por tres candidatos medidos contra un test congelado |
| Cifras del documento original sin fuente ("~40% menos de costos", "barreras insuperables", "<1 MB con INT8") | No verificables o ya superadas por la competencia |

## 4. Afirmaciones externas recibidas y su veredicto

| Afirmación | Veredicto | Detalle |
|---|---|---|
| "En silencio, LayerNorm/normalización amplifica el ruido térmico y dispara el modelo" | Parcialmente correcta, no es la causa principal | Mecanismo plausible, sin fuente que lo demuestre. Orden real de probabilidad: (1) VAD y debounce desactivados por defecto, (2) umbral sin calibrar, (3) escala/normalización del frontend mal replicada, (4) LayerNorm sobre ventanas de baja energía. Se resuelve empíricamente con el test de 30 min de silencio |
| "En Android, `MIC`/`DEFAULT` suben la ganancia al máximo en silencio; usar `VOICE_RECOGNITION` o `UNPROCESSED`" | Correcta en la recomendación, exagerada en el mecanismo | AGC normaliza la salida a un nivel prefijado (sube en silencio: así está definido). Que `MIC`/`DEFAULT` lo activen depende del dispositivo/HAL. La recomendación es correcta y está respaldada por la Compatibilidad Android 4.1 §5.3 (con `VOICE_RECOGNITION`, reducción de ruido y AGC "MUST be disabled") |
| "Compuerta RMS + VAD eliminan el 90% de los disparos fantasma" | Mecanismo correcto, cifra inventada | openWakeWord incluye VAD de Silero opcional y su README dice que "puede reducir significativamente" los falsos positivos en ruido no vocal. No publica ningún porcentaje. El piso de energía debe validarse en campo lejano |
| "openWakeWord encadena 3 modelos y el melspectrogram espera int16 cast a float; pasar floats /32768 deja el modelo sordo" | Correcta, con matiz importante | Confirmado por una integración de terceros ("expects raw int16 values cast to float"). Matiz: la API de Python acepta floats y convierte internamente, y el camino TFLite tiene otra distribución/normalización (features sin normalizar de media ~2,2 frente a un clasificador que espera normalizadas). Regla: replicar exactamente el camino con el que se entrenó y verificarlo con test de paridad numérica |
| "Precalcular embeddings de Common Voice una vez y guardarlos como dataset privado con mmap; entrenar en <3 min" | Correcta en el enfoque, optimista en los números | openWakeWord hace exactamente eso (ACAV100M precomputado, 16 GB para 2.000 h ≈ 8 MB/h → 595 h ≈ 4,8 GB, que cabe en la cuota privada de Kaggle). Entrenar la cabeza sobre features precalculadas es de minutos; la **extracción** inicial es el costo real y debe medirse y hacerse por lotes reanudables |
| "Los detectores de streaming sólo emiten cuando su ventana está llena" (hallazgo propio de esta auditoría) | Verificada | PR del probe de OpenVoiceOS: los clips cortos sin silencio líder se pierden; hay que cebar con ~3 s. Aplica al benchmark y a la app |

## 5. Registro de procedencia (plantilla)

Se completa en la fase 4. Una fila por fuente; sin fila, la fuente no entra al pipeline.

| Fuente | Tipo | Licencia | URL | Hash | Uso comercial | Fecha de verificación |
|---|---|---|---|---|---|---|
| (pendiente) | | | | | | |
