# DECISIONES — wakeword-es

Una fila por decisión cerrada. Lo que no está en esta tabla no está decidido. Lo que
quedó abierto está nombrado como pendiente al final.

| # | Decisión | Qué se decidió | Evidencia que lo respalda | Estado |
|---|---|---|---|---|
| 1 | Propósito | Proyecto de portafolio técnico con ambición de producto: 2–3 usuarios externos reales, no startup desde el día 1 | El nicho completo (openwakeword.com) acumula ~3.947 modelos generados y ~16.033 descargas; a ~4 CHF/modelo el techo es de decenas de miles, no de financiación | Cerrada |
| 2 | Qué se construye | Cadena con **procedencia limpia** + datos y medición en español; el activo es el dato y la licencia, no el código | Los modelos de openWakeWord son CC BY-NC-SA 4.0 por datos de entrenamiento con licencias desconocidas; microWakeWord usa FMA-medium, FSD50K y WHAM; openWakeWord usa ACAV100M/AudioSet/FSD50K/FMA. Ninguna es comercialmente limpia | Cerrada |
| 3 | Usuario cero | Diego en su propio Android; el móvil es laboratorio y demo, no validación externa | El plan Free de Porcupine cubre uso personal no comercial; los modelos comunitarios en español ya existen (openwakeword.com) | Cerrada |
| 4 | Plataforma de la v1 | Android (demo) + Linux/Python (pipeline). ESP32-S3 queda para fase 11 | Presupuesto $0: no hay hardware que comprar; el móvil ya lo tiene | Cerrada |
| 5 | Presupuesto | 6–10 h/semana × 20 semanas, solo, $0, GPU vía Kaggle (30 h/semana) | Kaggle: 30 h/semana de GPU, sesiones 9–12 h, T4×2/P100. GPU no es el cuello de botella: 600 h de GPU vs 120–200 h de tiempo humano | Cerrada |
| 6 | Postura comercial | Todo abierto y gratuito, sin cobrar en la v1; diferenciación por calidad medida y datos limpios en español | Picovoice: Free no comercial, Foundation $6.000/año, Enterprise desde $30.000/año. El hueco de precio ya está ocupado a ~4 CHF/modelo | Cerrada |
| 7 | Licencia | Apache-2.0 + NOTICE en el código, y cada dato y modelo con su licencia en un registro de procedencia versionado. Descartado cualquier esquema NC | Apache-2.0 ya exige atribución; una licencia NC dejaría el proyecto fuera de OSI y mataría la adopción, que es el objetivo declarado | Cerrada |
| 8 | Métrica de aceptación | ≥95% recall a ≤0,5 FA/h en 3 condiciones; 0 activaciones en 30 min de silencio digital; curva DET completa; baseline obligatorio | Los objetivos publicados de openWakeWord son <5% FRR y <0,5 FA/h con umbral ajustado; Picovoice compara a 1 FA/10 h. Un punto único de operación es manipulable | Cerrada |
| 9 | Idioma | Solo español latinoamericano en la v1; multiidioma explícitamente fuera | livekit-wakeword documenta que su precisión multilingüe es menor por el backbone entrenado en inglés | Cerrada |
| 10 | TTS de entrenamiento | Chatterbox Multilingual `es-mx-latam` (MIT, finetune dedicado es-419/es-MX) para diversidad de acentos clonados + Kokoro-82M (Apache-2.0, voces `ef_dora`, `em_alex`, `em_santa`) para volumen. Piper solo como experimento de comparación | Piper: motor GPL-3.0 (el repo MIT quedó archivado en oct-2025), licencia por voz y la voz Lessac es Blizzard (solo investigación) | Cerrada |
| 11 | Descartados de TTS | F5-TTS (pesos CC-BY-NC-4.0; el mantenedor confirma que no es comercial ni fine-tuneado), Fish Speech/OpenAudio (CC-BY-NC-SA-4.0 + acuerdo escrito), ChatTTS (CC BY-NC 4.0), Zonos (Apache-2.0 pero sin español soportado oficialmente) | Modelos de licencia y model cards respectivos | Cerrada |
| 12 | Datos negativos | Common Voice es (CC0: 2.278 h, 595 h validadas, 26.940 hablantes) + MUSAN auditado ítem por ítem + grabaciones propias del entorno | ESC-50 y UrbanSound8K son CC BY-NC; AudioSet/ACAV100M son la razón de la licencia NC de openWakeWord | Cerrada |
| 13 | Artefacto de la v1 | Repo con CLI + notebook reproducible + modelos en Hugging Face, y dataset/benchmark en español como aporte principal. La app Android es demo de fin de fase | Con 120–200 h, el código del pipeline es copiable; el test y las cifras medidas no | Cerrada |
| 14 | Partición de datos | Disjunto por hablante; test congelado con hash publicado antes de entrenar; en el test público además disjunción **por motor de TTS** | El split aleatorio por clip mete al mismo hablante en train y test e infla la métrica | Cerrada |
| 15 | Arquitectura | Tres candidatos máximo: C (backbone congelado + capa anti-disparo + calibración), luego A (CNN de streaming desde cero con objetivo FA/h, linaje microWakeWord/Google Research) y B (cabeza Conv-Attention sobre backbone congelado, linaje livekit). **Se empieza por C** | livekit-wakeword documenta que el embedding congelado (entrenado en inglés) es el techo multilingüe; microWakeWord optimiza FA/h como métrica primaria; interkelstar documenta el salto del minado de FP (91/140 → 1/140) | Cerrada |
| 16 | Formato de salida | Un solo formato por candidato, decidido por el ganador. Se empieza con ONNX (candidato C); TFLite INT8 si gana A | microWakeWord exporta TFLite uint8 de 50–300 KB; el ecosistema Home Assistant/Wyoming usa ONNX | Cerrada |
| 17 | Calibración de umbral | Publicar la curva DET **y** un script `calibrar` que graba 10 min del entorno y devuelve el umbral recomendado | openWakeWord deja el umbral en 0,5 y recomienda "determinar el mejor umbral por pruebas". **Corrección**: la idea no es inédita — `voicute/onnx-wakeword` ya publica 5 capas anti-falso-disparo; el diferenciador es el dato y la medición, no la existencia del mecanismo | Cerrada |
| 18 | Minado de falsos positivos | `minar-falsos-positivos` como comando de primera clase del pipeline desde el día uno | Reporte público de un trainer independiente: de 91/140 a 1/140 FP en casos duros, 94% recall en 50 activaciones reales reservadas | Cerrada |
| 19 | Auditoría de arquitectura | Máximo 2 días de trabajo; salida = `ARQUITECTURA.md` con tabla mecanismo → síntoma | El 80% de la evidencia ya está recogida y citada | Cerrada |
| 20 | Publicación del corpus | Pipeline + manifiestos + audio propio con consentimiento; el audio de Common Voice **no** se reempaqueta: cada usuario lo descarga por su script. Embeddings derivados solo en dataset privado | Términos de Common Voice piden no replicar el dataset en otras plataformas (CC0 en lo legal, petición de custodia en lo práctico) | Cerrada |
| 21 | Test público vs privado | Dos niveles: público sintético reproducible (disjunción por motor y hablante) + privado real (solo agregados y hashes) | Reproducibilidad: sin audio publicable, un benchmark no es verificable por terceros | Cerrada |
| 22 | Nombre | Repo `wakeword-es` (descriptivo y buscable); el nombre de marca se decide después | Convención de descubrimiento en GitHub | Cerrada |
| 23 | Fecha límite | Blanda, con checkpoint publicable obligatorio en la semana 14 (≈2027-01-23) | Restricción dominante: 120–200 h totales de una sola persona | Cerrada |
| 24 | Cómputo | Kaggle para el entrenamiento pesado; el VPS orquesta. **Revisada por la fila 28**: la extracción de características sí corre en el VPS | VPS: 4 vCPU, 7,8 GB RAM, 73 GB libres, **sin GPU**, Python 3.14 (el pipeline estándar exige 3.10–3.12) | Cerrada |
| 25 | Captura en Android | `AudioSource.VOICE_RECOGNITION` o `UNPROCESSED` (AGC y supresión desactivados por especificación), más test de paridad numérica Python ↔ Android | Compatibilidad Android 4.1 §5.3: con VOICE_RECOGNITION, reducción de ruido y AGC "MUST be disabled"; requisitos de respuesta plana ±3 dB de 100 Hz a 4 kHz | Cerrada |

| 26 | Aprobación del plan | **Aprobado por Diego el 2026-10-17** («dale»). Es la fase 1 del plan y desbloquea la ejecución | Mensaje explícito de aprobación tras la Ronda 4 | Cerrada |
| 27 | Definición de "activación" y del criterio | Una activación es un frame ≥ umbral que no cae dentro del debounce (cooldown por defecto 1,0 s). El criterio es **código ejecutable**: `--criterio` devuelve 1 si no se cumple, y el banco aborta con código 2 si el hash del manifiesto del test dejó de coincidir | `benchmark/evaluar.py`, `benchmark/README.md`; verificado con `make test` (12/12) y `make autotest` (perfil bueno cumple, perfil malo falla) | Cerrada |
| 28 | Dónde se extraen las características | En el **propio servidor**, no en Kaggle: medido **28,7× tiempo real** (120 s de audio en 4,18 s) → 595 h ≈ 20,7 h de CPU en serie y ~5 h con 4 procesos. A Kaggle solo viaja el dataset **privado** de características derivadas, y Kaggle queda para el entrenamiento con GPU del candidato A | Medición del 2026-10-17 en este host: uv + Python 3.11 + `openwakeword==0.6.0` + onnxruntime 1.31.0 (CPU); features de forma (1491, 96) para 120 s | Cerrada |
| 29 | Evaluación estratificada por acento | **Aceptada por Diego el 2026-10-17.** El acento andino-pacífico (Perú, Colombia, Ecuador, oeste de Bolivia y Venezuela andina) entra como **condición propia** del benchmark, no disuelto en el promedio: 38.728 clips de 947 hablantes en Common Voice es 26.0 | Desglose de acentos de la ficha del dataset + el criterio ya exige el mínimo por grupo, no el agregado | Cerrada |

## Pendientes (nombrados como pendientes, no decididos)

1. **La frase exacta.** Se elegirá por medición en la fase 3; hoy no hay frase decidida.
2. **Licencia final de los modelos entrenados** (CC-BY vs Apache-2.0 vs otra). Depende de
   si se publican las voces clonadas con consentimiento.
3. **Si se publican grabaciones propias bajo CC0.** Requiere consentimiento específico
   para publicación, distinto del consentimiento para entrenar.
4. **XTTS-v2 (Coqui)** quedó fuera de la evaluación por su licencia CPML; no se verificó
   a fondo en esta sesión. Si alguien lo propone, auditar antes de descartar.
5. **Volumen final del corpus de negativos** (595 h vs un subconjunto). Se decidirá con el
   tiempo real de extracción medido en la fase 4 y la cuota privada de Kaggle.
6. **Fase 11**: si se persigue ESP32-S3 y/o los candidatos A y B, y con qué presupuesto.
