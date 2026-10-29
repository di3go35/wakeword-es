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
| Common Voice español **27.0** | CC0-1.0, MP3, **51,94 GB (48,38 GiB)**; 1.683.737 clips, 2.282,72 h (597,6 h validadas), 26.960 hablantes; desglose por acento con **andino-pacífico: 38.848 clips / 950 hablantes**. Prohibido re-hospedar o re-compartir; usar solo localmente y derivar características | Negativos reales en español, limpios y gratuitos; permite evaluación estratificada por acento | dataset ID `cmu5kitx700pemi073feoiqqu`, descargado por API el 2026-10-17 |
| Common Voice es 27.0 — **medido en el subconjunto validado** | **440.794 clips / 619,95 h / 0 clips sin duración / media 5,06 s.** Acentos (clips · horas · hablantes): sin declarar 143.473 · 205,65 h · 13.291; México 142.035 · 194,34 h · 1.083; España Sur 43.700 · 58,12 h · 186; España Norte 40.778 · 57,86 h · 463; **andino-pacífico 17.694 · 26,26 h · 693**; rioplatense 13.778 · 20,5 h · 415; resto menor. `accents` es **texto libre** con filas multi-acento | Es el dato sobre el que se decide: el andino utilizable son 26,26 h, no los 38.848 clips del datasheet (ese número es del corpus completo) | Medición propia con `pipeline/preparar_validado.py`, 2026-10-18 |
| MUSAN | CC BY 4.0 general, fuentes mixtas | Ruido/música para aumentación, con auditoría ítem por ítem | openslr.org/17 |
| Chatterbox Multilingual V3 + `es-mx-latam` | MIT, 0,5B, 23+ idiomas, clonación zero-shot, finetune dedicado es-419 | Diversidad de acentos y voces clonadas con consentimiento | HF: ResembleAI/Chatterbox-Multilingual-es-mx-latam |
| Kokoro-82M | Apache-2.0; español: `ef_dora`, `em_alex`, `em_santa` | Volumen de datos sintéticos | HF: hexgrad/Kokoro-82M |
| Paper de Google Research *Streaming Keyword Spotting on Mobile Devices* | Base de microWakeWord | Fundamento del candidato A | Ryabov et al. |

## 2. Evidencia que falta producir (y en qué fase)

| Evidencia | Fase | Cómo se verifica desde fuera |
|---|---|---|
| Curva DET de la línea base (números malos publicados) | 2 | **Hecha**: `benchmark/resultado_linea_base*.json` (0,773 → 0,019 FA/h tras corregir el etiquetado) |
| Frase congelada con su justificación medida | 3 | **Hecha**: `FRASE.md` + `benchmark/FRASE_HASH` (`f2a3f422…`) |
| Registro de procedencia de cada fuente con licencia y hash | 4 | **Hecha** para Common Voice (`PROCEDENCIA.txt`, SHA256 `b7c1f303…`); MUSAN y TTS documentados en `docs/MODEL_CARD.md` |
| Consentimientos escritos de 10–15 personas | 4 | **Parcial**: solo la voz del autor (10 grabaciones). Falta el resto del piloto |
| Características de los 3 lotes (620 h) | 4 | **Verificado**: lote 000 (170.930 clips, 250,00 h) y lote 001 (179.918 útiles de 179.924, 250,00 h, 6 descartes con motivo documentado) pasan `verificar_features.py` sin faltantes sin explicación. Lote 002 (119,95 h) en ejecución. Factor medido: 21,9× por proceso, 12 h 41 min de CPU en el lote 001 |
| Test congelado con hashes publicados antes de entrenar | 5 | **Hecha pero degradada**: `benchmark/SPLIT_HASH` y `benchmark/test_negativos.json` publicados, con la salvedad de que el test se usó ~10 veces como test de desarrollo |
| Curva DET del modelo final vs línea base | 6 | **Hecha**: `docs/resultados.md` con la curva completa de FA/h y el cuadro de enfoques fallidos |
| 30 min de silencio digital = 0 activaciones | 6 | **Hecha**: 0 activaciones en ceros y en ruido térmico a -70 dBFS (máx 0,0799) |
| Delta del minado de falsos positivos | 7 | **Hecha**: 4,469 → 0,094 FA/h (47×) con recall intacto; segundo minado 0,867 → 0,396 |
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

## 6. Estado de los artefactos (2026-10-26)

| Artefacto | Estado | Evidencia |
|---|---|---|
| Corpus de negativos | 440.788 clips / 619,95 h → características 96-dim | tres lotes verificados, 6 descartes documentados |
| Partición por hablante | Congelada | 14.441 train / 3.247 test, hash `9e237c2d…` |
| Audio de test | 37.106 clips / 53,03 h | `benchmark/test_negativos.json` |
| Frase | `hey celeste` | `FRASE.md`, medición sobre 10,9 M palabras |
| Positivos Kokoro | 1.800 clips, `es-419` (seseo) | manifiesto con hash por clip |
| Positivos Chatterbox | 964 limpios de 1.043 descargados | filtro de frase única y 0,9-2,0 s |
| Modelo final | `cabeza_fewshot.onnx`, 196.865 parámetros | `docs/MODEL_CARD.md` |
| Medición | 53,03 h, 10 condiciones de acento | `docs/resultados.md` |
| Reproducibilidad | `make test` 12/12, `make autotest` falla como debe | CI manual documentado |
