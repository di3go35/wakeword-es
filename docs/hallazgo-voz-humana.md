# La voz humana real es el muro

**Fecha:** 2026-10-26 · **Decisiones:** 37

## Qué se midió

Diez grabaciones reales de Diego diciendo "hey celeste" —micrófono de laptop, nivel bajo (pico entre
-16 y -28 dBFS, RMS entre -38 y -50)— contra el mejor modelo del proyecto hasta ese momento:

- Cabeza de 196.865 parámetros sobre el embedding congelado de Google (96 dimensiones).
- Entrenada con 11.479 ventanas positivas de **dos motores TTS** (Kokoro y Chatterbox) y 151.396
  ventanas negativas de habla real (Common Voice es 27.0).

## Resultado

| Condición | Probabilidad máxima | Recall @0,5 |
|---|---|---|
| Nivel original (muy bajo) | 0,0019 | **0,000** |
| Normalizado a -6 dBFS | 0,0123 | **0,000** |
| Estirado ×1,4 (por si el fallo era velocidad) | 0,0006 | **0,000** |
| Estirado ×1,8 | 0,0005 | **0,000** |

Para comparar, el mismo modelo sobre TTS nunca visto:

| Fuente | Recall @0,5 |
|---|---|
| Chatterbox, 192 clips no vistos | 0,995 |
| Kokoro `em_santa`, 600 clips (voz no vista) | 0,992 |

## Verificaciones que descartan explicaciones alternativas

1. **¿El audio dice la frase?** Whisper `medium` sobre las versiones normalizadas transcribe
   "¡Hey, Celeste!" en 5 de 10 y variantes cercanas ("Celesta", "Celestin") en el resto. El
   `small` sobre el audio sin normalizar alucinaba ("¡Hasta la próxima!"), que es el
   comportamiento esperado de Whisper con nivel muy bajo. La frase está.
2. **¿Es el nivel?** Normalizar a -6 dBFS sube la probabilidad de 0,0019 a 0,0123 — mejora 6×
   pero sigue a dos órdenes de magnitud del umbral.
3. **¿Es la velocidad?** Las grabaciones tienen 0,4-1,0 s de voz, frente a 1,2-1,6 s de los
   positivos TTS. Estirar el tiempo ×1,4 y ×1,8 **no mejora nada**.
4. **¿Faltan negativos difíciles?** No: minando 40.000 clips de hablantes de entrenamiento
   (~1,1 M ventanas), solo **24 ventanas** superan 0,10. El modelo no es que confunda habla real
   con la frase; es que la rechaza casi siempre.

## Interpretación

El modelo aprendió **cómo suena "hey celeste" en un sintetizador**, no cómo suena "hey celeste".
La generalización que sí tiene es entre motores TTS, que comparten el mismo tipo de señal (limpia,
sin ruido de sala, con un timbre sintético característico). Una voz humana con micrófono de laptop
cae fuera de esa distribución y el embedding congelado —entrenado por Google predominantemente en
inglés— la separa sin piedad.

Esto convierte el "0,019 FA/h" en una métrica engañosa por sí sola: un modelo que nunca se dispara
tiene falsos positivos perfectos y utilidad nula. El par de métricas es lo que informa.

## Las dos salidas posibles

1. **Aumentación de canal** (en curso): simular sobre los positivos TTS lo que diferencia una
   grabación real — nivel, respuesta de micrófono, reverberación y ruido. Si funciona, el modelo
   sigue siendo zero-shot: sirve para un usuario nuevo sin grabaciones.
2. **Few-shot**: que el usuario grabe unas pocas muestras y se ajuste el modelo. Cambia el diseño
   del producto (no funciona "de fábrica") pero es lo que hacen varias herramientas del sector.

Si ninguna de las dos levanta el recall, el techo está en el **embedding congelado** y hay que
pasar al candidato A/B: entrenar la red sobre espectrogramas de mel en vez de sobre características
de Google.
