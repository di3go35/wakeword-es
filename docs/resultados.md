# Resultados medidos

Todos los números de este documento salen de `pipeline/` y se pueden reproducir. Fecha: 2026-10-26.

## El conjunto de evaluación

| | |
|---|---|
| Habla real de negativos | **37.106 clips / 53,03 h**, 3.247 hablantes, 10 condiciones de acento |
| Partición | Disjunta por hablante, congelada con hash (`benchmark/SPLIT_HASH`) |
| Positivos TTS de entrenamiento | Kokoro 3 voces (`es-419`) + Chatterbox, originales y aumentados |
| Positivos de voz real | 10 grabaciones del usuario (micrófono de laptop) |
| Acento andino-pacífico | 5,00 h de audio real, 138 hablantes |

**Aviso metodológico:** este test se ha mirado ~10 veces durante el desarrollo, así que es un **test
de desarrollo**, no un test final. La cifra publicable exige grabaciones nuevas que nunca se hayan
usado para decidir nada.

## Resultado final (recetario few-shot)

Umbral **0,9** con 3 frames consecutivos y 1 s de enfriamiento.

| Condición | Resultado | Cómo se midió |
|---|---|---|
| Recall, limpio | **100%** (10/10) | En muestra; el usuario grabó esos audios |
| Recall, ruido no visto (puntuación) | 100% de los clips con mediana ≥0,99 | Mediana de la puntuación por clip, validación dejando uno fuera |
| **Recall, ruido no visto (regla operativa)** | **66%** (53/80 variantes) | 3 frames sobre 0,9 y enfriamiento — el número que ve el usuario |
| Recall, campo lejano (3 m) | **100%** (10/10) | Simulado: RIR de 0,5 s, paso bajo 5 kHz, -18 dBFS |
| Falsos positivos | **0,113 FA/h** | 6 eventos en 53,03 h de habla real |
| Especificidad: "celeste" en frase normal | **0 de 30 disparan** | 30 frases TTS con "celeste"/"hey" (máx 0,2198) |

### Corrección metodológica (2026-10-26, tarde)

La fila de "100% con ruido" se midió como **mediana de la puntuación por clip**, no como activación
sostenida. Con la regla que se usa en el producto (3 frames consecutivos sobre 0,9), el recall en
ruido es **66%**. No es un fallo de la compuerta: sin compuerta da lo mismo (66,2%). Y no depende del
SNR: falla 45% a SNR <0 dB, 38% entre 0 y 10 dB y 26% por encima de 10 dB — es decir, hay fallos a
+19 dB. La causa probable es que la voz real con solo 10 muestras no cubre la variabilidad de canal
(EQ + reverberación + ruido) que sí cubre el TTS.

### Compuerta de energía (fase 6.1)

| Piso | Recall limpio | Recall lejano 3 m | Recall ruido |
|---|---|---|---|
| sin compuerta | 100% | 100% | 66% |
| −45 dBFS fijo | 100% | **100%** | 66% |
| −60 dBFS fijo | 100% | 100% | 66% |
| adaptativo (suelo local + 8 dB) | 100% | 100% | 66% |

La compuerta **no cuesta recall en ninguna condición** y quita los disparos en silencio digital (12 → 11
eventos en los 10 clips limpios; el evento eliminado estaba en un frame a −120 dBFS, silencio exacto).
El riesgo que el plan advertía —que el piso rompiera el campo lejano— **no se materializa a −45**: a
−35 sí (recall lejano 100% → 40%). Recomendación: piso **−45 dBFS**.
| Silencio digital (30 min) | **0 activaciones** | Probabilidad máxima 0,0799 |
| Ruido térmico -70 dBFS (30 min) | **0 activaciones** | Probabilidad máxima 0,0070 |

Curva de falsos positivos del modelo desplegado:

| Umbral | FA/h |
|---|---|
| 0,50 | 0,396 |
| 0,70 | 0,302 |
| 0,80 | 0,113 |
| **0,90** | **0,113** |
| 0,95 | 0,075 |

Criterio del proyecto: ≥95% recall y ≤0,5 FA/h. **Cumple a partir de umbral 0,8.**

## Los enfoques que no funcionaron, y por qué

| Enfoque | Recall voz real | FA/h | Conclusión |
|---|---|---|---|
| Solo TTS (Kokoro) | 0/10 | 0,019 | Aprende el motor, no la frase |
| TTS multi-motor (Kokoro + Chatterbox) | 0/10 | 0,019 | Generaliza entre sintetizadores, no a personas |
| + aumentación de canal | 0/10 | 0,019 | Sube la probabilidad 5× (0,0123 → 0,0634), no cruza |
| Few-shot sin negativos difíciles | 87,5% | 4,469 | Acierta pero dispara 4,5 veces por hora |
| Few-shot + minado | 87,5% | 0,094 | El minado baja los falsos positivos 47× |
| Few-shot + ruido | 100% | 0,867 | Mejor recall, peor precisión |
| Few-shot + ruido + reminado | 100% | 0,151 | Cerca, pero disparaba con "celeste" en una frase |
| **+ negativos de frase ("hey, ¿viste a Celeste?")** | **100%** | **0,113** | Punto de operación final: 0/30 frases disparan |

## Comparación con TTS nunca visto (control de motor)

Con el modelo multi-motor, sobre clips de Chatterbox que no estaban en el entrenamiento:

| Umbral | Recall |
|---|---|
| 0,50 | 99,5% (191/192) |
| 0,70 | 98,4% (189/192) |

Es decir: el modelo **sí** generaliza entre sintetizadores. Lo que no puede, sin grabaciones reales,
es cruzar a voz humana.

## Control negativo: ¿detecta la frase o "esto es TTS"?

Misma voz, otras frases, con el modelo entrenado solo con TTS:

| Frase | Prob. media | Clips sobre 0,5 |
|---|---|---|
| hola mundo | 0,0000 | 0/27 |
| buenos días | 0,0001 | 0/27 |
| mañana temprano | 0,0000 | 0/27 |
| la casa azul | 0,0699 | 1/27 |

El modelo aprende la frase dentro de su dominio. El control que **no** sirve para detectar el fallo de
dominio es este: da 0,0000 y aun así el modelo fallaba con voz humana. El único control que lo
detecta es cambiar de motor o de persona.

## Tamaño y coste

| | |
|---|---|
| Cabeza clasificadora | 196.865 parámetros, ~790 KB en float32 |
| Ventana | 16 frames de embedding = **2,04 s de audio** (no 1,28 s) |
| Extracción de características | ~30× tiempo real en CPU de 4 vCPU |
| Entrenamiento | 2-5 s por época sobre 200.000 ventanas |
| Ruido de fondo | MUSAN (CC BY 4.0) |
