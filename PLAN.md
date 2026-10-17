# PLAN — wakeword-es

Proyecto: pipeline abierto, con procedencia limpia, para entrenar wake words en español
latinoamericano con falsos positivos medidos y reproducibles.

- Fecha de escritura: 2026-10-17 (sábado).
- Presupuesto real: 6–10 h/semana × 20 semanas, una persona, $0 de presupuesto,
  GPU vía Kaggle (30 h/semana, T4×2/P100, sesiones de 9–12 h). Semana 20 ≈ 2027-03-06.
- Checkpoint publicable: semana 14 ≈ 2027-01-23.
- Fecha límite: blanda (no hay entrega universitaria asociada), por lo que el
  checkpoint de la semana 14 es el compromiso duro real.
- Entorno de cómputo del pipeline: Kaggle (Python 3.10–3.12).
  El VPS actual (4 vCPU, 7,8 GB RAM, 73 GB libres, Python 3.14, sin GPU) sirve para
  orquestar y documentar, **no** para entrenar.

## Regla de oro

Ninguna afirmación de calidad entra al repo sin un número medido y reproducible por
terceros. Toda fase termina con una verificación que se puede hacer **desde fuera de la
máquina** de Diego (URL, hash, comando reproducible).

## Criterio de éxito (definición de "mejor")

Sobre la batería congelada de la fase 5, en tres condiciones (limpio, TV/música a ~10 dB
SNR, campo lejano a 3 m con reverb real):

- ≥95% de recall a ≤0,5 falsos positivos por hora.
- 0 activaciones en 30 min de silencio digital.
- Curva DET completa publicada (recall a 0,2 / 0,5 / 1,0 FA/h), no un punto único.
- Baseline obligatorio corrido con el mismo test: openWakeWord preentrenado, el mejor
  modelo comunitario en español disponible y el intento previo con livekit-wakeword.

---

## Fase 1 — Aprobación de este plan (bloqueante)

Nada se ejecuta hasta que Diego diga «dale».

**Verificación externa:** aprobación registrada en `DECISIONES.md` con fecha.

---

## Fase 2 — Auditoría y línea base honesta (semanas 1–2)

1. Terminar `ARQUITECTURA.md` (tabla mecanismo → síntoma; base ya redactada).
2. Crear el repo público `wakeword-es` con licencia Apache-2.0 + NOTICE, y un README que
   declare el problema y el criterio de éxito **sin promesas**.
3. Reproducir el pipeline de referencia (openWakeWord / livekit-wakeword) con una frase
   provisional y correr la batería completa.
4. Publicar los resultados, aunque sean malos.

**Verificación externa:** `benchmark/RESULTADOS.md` con la curva DET de la línea base y
un comando (`make benchmark`) que un tercero pueda correr en su máquina obteniendo el
mismo número (± tolerancia declarada). Si la línea base sale bien, el test está mal
construido y se arregla antes de seguir.

---

## Fase 3 — Frase elegida por medición (semana 3)

1. Medir la frecuencia del n-grama candidato en las transcripciones de Common Voice es
   (descartar frases cuyo núcleo sea una palabra frecuente; "oye" sola queda descartada
   por diseño).
2. Medir distancia fonética a las palabras más frecuentes del español.
3. Elegir la frase y congelarla (hash de commit).

**Verificación externa:** `FRASE.md` con el script, la tabla de candidatas con sus
puntajes y el hash del commit donde se congeló.

---

## Fase 4 — Corpus y procedencia (semanas 3–6)

1. Alta en Mozilla Data Collective y descarga del corpus español (CC0). **No se
   reempaqueta ni se publica el audio de Common Voice.**
2. Precomputar embeddings por lotes **reanudables en el propio servidor**: medido a
   **28,7× tiempo real** (openwakeword 0.6.0 + onnxruntime CPU en 4 vCPU) → 595 h ≈
   **20,7 h de CPU en serie, ~5 h con 4 procesos**. Vector verificado: **96 dimensiones**,
   ~12,4 ventanas/s. Tamaño estimado ~4,8 GB a 16 bits. A Kaggle solo viajan las
   **características derivadas** (dataset privado); el audio no sale del servidor.
3. MUSAN: auditar ítem por ítem y descartar lo que no sea claro.
4. Grabar 30–60 min de ruido propio (TV, calle, cafetería, ventilador).
5. Conseguir consentimiento escrito de 10–15 personas para clonar su voz.
6. Generar positivos con Chatterbox `es-mx-latam` (MIT) y Kokoro (Apache-2.0).

**Verificación externa:** `PROCEDENCIA.md` con una fila por fuente (licencia, URL, hash,
uso comercial sí/no) + consentimientos referenciados por hash + el dataset privado con su
tamaño real medido.

---

## Fase 5 — Test congelado y banco de evaluación (semanas 5–8)

1. Test público reproducible: 100% sintético, con disjunción **por motor de TTS y por
   hablante** (las voces del test nunca se usan en entrenamiento).
2. Test privado real: grabaciones propias y de personas que consientan; se publican solo
   agregados y hashes.
3. Regla de cebado: ~3 s de silencio líder antes de cada clip, porque los detectores de
   streaming no emiten hasta llenar su ventana deslizante (causa conocida de falsos
   rechazos en clips cortos). El benchmark **y** la app deben replicarlo.
4. Publicar hashes y manifiestos antes de entrenar.

**Verificación externa:** un tercero clona el repo, corre `make benchmark` con la línea
base de la fase 2 y reproduce el número.

---

## Fase 6 — Modelo, capa anti-disparo y calibración (semanas 6–11)

1. Candidato C: backbone congelado + cabeza entrenada con negativos en español +
   compuerta de energía (RMS) + VAD + `patience`/`debounce`.
2. `calibrar`: graba 10 min del entorno del usuario y devuelve el umbral recomendado para
   ese dispositivo y ese ruido (formato 1: características + script).
3. Negativos en español (embeddings de Common Voice) + negativos adversariales de TTS.

**Verificación externa:** curva DET contra la línea base; 30 min de silencio digital = 0
activaciones; el piso de energía validado en campo lejano a 3 m (no puede romper la
detección lejana).

---

## Fase 7 — Bucle de minado de falsos positivos (semanas 9–13)

1. `minar-falsos-positivos`: recolecta los clips que dispararon, re-extrae negativos con
   ventanas densas (1 ventana cada 10 ms) y reentrena.
2. Medir el delta antes/después sobre el test congelado.

**Verificación externa:** reporte del delta con los comandos exactos.
Referencia pública del efecto esperado: un trainer independiente pasó de 91/140 a 1/140
falsos positivos en casos duros manteniendo 94% de recall.

---

## Fase 8 — Checkpoint publicable (semana 14)

Repo público + modelo ONNX + benchmark + `ARQUITECTURA.md` + `PROCEDENCIA.md` + modelo en
Hugging Face + tabla comparativa contra la línea base.

**Verificación externa:** URLs públicas reales y un tercero que carga el modelo y
reproduce el número declarado.

---

## Fase 9 — Demo Android (semanas 15–18)

1. App de prueba que corre el modelo en el teléfono de Diego (no es producto).
2. Captura con `AudioSource.VOICE_RECOGNITION` o `UNPROCESSED` (AGC y supresión de ruido
   desactivados por especificación de Android), cebado de ventana, calibración en el
   dispositivo.
3. **Test de paridad numérica Python ↔ Android** sobre el mismo WAV, frame por frame:
   sin eso, cualquier diferencia de escala o normalización del frontend deja el modelo
   sordo o locuaz.
4. Medir batería (%/h) y latencia p95.

**Verificación externa:** video corto + tabla de consumo y latencia medidos en su
teléfono con el método descrito.

---

## Fase 10 — Publicación y decisión de continuidad (semanas 18–20)

1. Difusión en la comunidad hispanohablante de Home Assistant / Rhasspy / OpenVoiceOS.
2. Recoger los primeros modelos entrenados por terceros.

**Verificación externa:** métricas de adopción (descargas, issues, modelos hechos por
otros) y decisión explícita y escrita sobre la fase 11.

---

## Fase 11 (opcional, fuera de la v1)

ESP32-S3 con TFLite INT8, y los candidatos A (CNN de streaming desde cero con objetivo
FA/h) y B (cabeza Conv-Attention sobre backbone congelado) si el candidato C no alcanza
el criterio de éxito.

## Fuera de alcance declarado

Rust, C++, Flutter, React Native, Cortex-M, nube/SaaS de entrenamiento, multiidioma,
certificación de hardware y cualquier promesa de "menos de 1 MB" o de ahorro de costos
sin fuente.
