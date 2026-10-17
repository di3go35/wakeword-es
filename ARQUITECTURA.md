# ARQUITECTURA — mecanismo → síntoma → comprobación

Auditoría de las implementaciones de referencia, hecha para explicar los dos síntomas
observados en las pruebas previas (activaciones sin ruido; fallos de reconocimiento) y
para elegir candidato con evidencia, no por preferencia.

## Cadena de inferencia de openWakeWord / livekit-wakeword (candidato C)

Tres modelos encadenados en streaming, a 16 kHz:

1. `melspectrogram.onnx` (~1,1 MB): audio PCM → tramas de espectrograma mel de 32 bandas.
2. `embedding_model.onnx` (~1,3 MB): Google `speech_embedding/1` congelado; mel → vector de
   96 dimensiones por ventana (ventanas de 76 tramas con paso de 8, según la documentación
   de AudioFeatures).
3. `tu_modelo.onnx` (~200 KB): cabeza clasificadora sobre 16 ventanas de embeddings.

Predicción cada 80 ms; el valor devuelto por fragmento es el **máximo** de las predicciones
internas, y por defecto **no hay debounce ni VAD**.

## Tabla de mecanismos

| Mecanismo | Síntoma que explica | Fuente | Cómo se comprueba |
|---|---|---|---|
| Backbone de embeddings congelado, entrenado predominantemente en inglés | Recall bajo y variable en español; frases que "a veces sí, a veces no" | README de livekit-wakeword (los propios mantenedores): la precisión multilingüe es menor porque "el embedding congelado de Google fue entrenado predominantemente con datos en inglés" | Comparar la misma cabeza sobre A (entrenada desde cero) y C: si A supera a C de forma consistente, el techo es el backbone |
| VAD y debounce **desactivados por defecto** (`vad_threshold=0`) | Activaciones sin voz y sin ruido | `openwakeword/model.py` (`vad_threshold` default 0 desactiva VAD); Silero VAD es opcional | Test de 30 min de silencio digital: 0 activaciones exigidas |
| Umbral fijo 0,5 sin calibrar por entorno | Disparos fantasma y falsos rechazos en el mismo modelo | README de openWakeWord: "determinar el mejor umbral por pruebas" | Curva DET por entorno + script `calibrar` |
| Predicción por fragmento con máximo interno, sin lógica temporal | Múltiples activaciones de la misma frase; disparos por transitorios | Discusión #100 de openWakeWord (respuesta del mantenedor: no implementa debounce) y PR #116 (debounce opcional) | Contar activaciones duplicadas por frase en el test |
| Ventana deslizante que necesita llenarse antes de emitir (~2,5–3 s) | Falsos rechazos en clips cortos y en los primeros segundos tras arrancar | PR del probe de OpenVoiceOS: los detectores de streaming no emiten hasta llenar su ventana; los clips sin silencio líder se pierden | Benchmark y app con ~3 s de silencio líder; probar el mismo clip con y sin cebado |
| Escala y normalización del frontend mel/embedding mal replicadas al salir de Python | Modelo "sordo" (scores ~0,000) o constantemente activo en Android/C++ | Integración de terceros que comenta el código: "convert to float WITHOUT normalizing — openWakeWord expects raw int16 values cast to float"; reporte independiente del backend TFLite con features sin normalizar (media ~2,2) frente a un clasificador que espera normalizadas | Test de paridad numérica Python ↔ Android sobre el mismo WAV, frame por frame |
| AGC y supresión de ruido del sistema operativo en Android | Ganancia bombeando en silencio: basura a la entrada del modelo | Compatibilidad Android 4.1 §5.3: con `VOICE_RECOGNITION`, reducción de ruido y AGC "MUST be disabled"; requisitos: respuesta plana ±3 dB de 100 Hz a 4 kHz | Grabar el mismo silencio con `MIC`/`DEFAULT` vs `VOICE_RECOGNITION`/`UNPROCESSED` y comparar energía y activaciones |
| LayerNorm sobre ventanas de baja energía (hipótesis, no verificada) | Ruido térmico normalizado hasta parecer señal | No hay fuente que lo demuestre; es plausible por construcción de LayerNorm | Mismo test de silencio digital; se acepta o se descarta según el resultado |
| Negativos de entrenamiento en inglés (AudioSet/ACAV100M/FMA/FSD50K/WHAM) | TV y conversación en español tratadas como dominio no visto | Dataset card de microWakeWord (FMA-medium, FSD50K, WHAM); licencia CC BY-NC-SA de los modelos de openWakeWord por datos con licencias desconocidas | Entrenar con negativos en español (Common Voice) y medir el delta de FA/h |
| Objetivo de entrenamiento por exactitud por ventana en vez de FA/h | Falsos positivos en ruido ambiente que la métrica de entrenamiento no penaliza | README de microWakeWord: optimizar primero "false accepts on ambient background noise", con pesos por conjunto de features | Usar FA/h medida como métrica de selección de pesos en el candidato A |
| Positivos generados con pocas voces TTS | El modelo aprende el timbre del TTS, no la fonética de la frase | README de livekit-wakeword: la TTS produce habla sintética "menos diversa" que un pool amplio de hablantes | Split disjunto por hablante y por motor |

## Consecuencias de diseño que ya están en el plan

- La compuerta de energía (RMS) y el VAD entran como red de seguridad, pero con una
  advertencia: un piso demasiado agresivo rompe la detección a 3 m, que es justo una de
  las condiciones del criterio de éxito. Se calibra contra campo lejano, no contra
  escritorio.
- El minado de falsos positivos es el mecanismo con mayor salto documentado por terceros
  (91/140 → 1/140) y por eso es comando de primera clase.
- Todo lo que se reimplemente fuera de Python exige paridad numérica verificada.

## Correcciones a afirmaciones previas

- La "capa anti-disparo" como idea original: **no lo es**. `voicute/onnx-wakeword` publica
  5 capas (frames consecutivos, relación pico/fondo, cooldown, detección de ráfaga, salto
  de energía) sobre un TCN causal de ~25K parámetros en 7 idiomas, español incluido.
- "Nadie hace español para wake words": **falso**. Existen `jlopez994/rufus-playbooks` y
  `Jota-project/jota-wake-trainer`, en español y para Home Assistant, construidos sobre el
  mismo backbone congelado — y por eso arrastran el mismo techo.
- "INT8 bajo 1 MB" como diferenciador: **ya superado** por la competencia (200 KB ONNX en
  openWakeWord, 50 KB en microWakeWord, <130 KB en onnx-wakeword).
