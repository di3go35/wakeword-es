# Model card — `cabeza_fewshot.onnx`

## Qué es

Clasificador de 196.865 parámetros que decide si una ventana de características contiene la frase de
activación **"hey celeste"**. No procesa audio: recibe las 96 dimensiones que produce la cadena
`melspectrogram.onnx` → `embedding_model.onnx` de openWakeWord (Apache-2.0).

- Entrada: `input` de forma `(batch, 16, 96)` — 16 frames de embedding.
- Salida: `output` de forma `(batch, 1)` — logit (aplicar sigmoide).
- Tamaño: ~790 KB en float32 (exportado con datos externos: `cabeza.onnx.data`).

## Entrenamiento

| Componente | Cantidad |
|---|---|
| Positivos TTS | 47.891 ventanas de Kokoro (`es-419`, 3 voces) y Chatterbox, originales y con aumentación de canal |
| Positivos de voz real | 2.632 ventanas de 10 grabaciones del usuario, aumentadas ×10 (canal, nivel, reverberación, ruido MUSAN) |
| Negativos reales | 151.396 ventanas de Common Voice es 27.0, hablantes disjuntos del test |
| Negativos difíciles | 3.831 ventanas minadas dos veces sobre hablantes de entrenamiento |
| Negativos de frase | 302 ventanas de 30 frases con "celeste" o "hey" que no deben activar |

Etiquetado: una ventana es positiva **solo si contiene la frase completa** (2,04 s de audio). Las
ventanas que cortan la frase entran como negativas.

## Métricas (test de desarrollo, 53,03 h de habla real de 3.247 hablantes)

| Condición | Resultado |
|---|---|
| Recall limpio | 100% (10/10, en muestra) |
| Recall con ruido no visto | 100% (10/10, dejando un audio fuera) |
| Recall campo lejano 3 m (simulado) | 100% (10/10) |
| Falsos positivos @0,9 | **0,113 FA/h** (6 eventos en 53 h) |
| Falsos positivos @0,8 | 0,113 FA/h |
| Especificidad ("celeste" en frase normal) | 0 de 30 disparan |
| Silencio digital 30 min | 0 activaciones |
| Ruido térmico -70 dBFS 30 min | 0 activaciones |

## Uso previsto

Wake word **personal**: el usuario graba 10-15 muestras de su propia voz y se entrena su modelo. No es
un modelo "de fábrica": entrenado solo con TTS, el recall con voz humana real es **0**.

Umbral recomendado: **0,9**, con 3 frames consecutivos sobre el umbral y 1 s de enfriamiento. Compuerta
de energía antes de la red: si el RMS del frame está por debajo de -45 dBFS, forzar 0.

## Fuera de alcance

- Otras frases: el modelo está entrenado para una sola. Cambiar la frase exige reentrenar.
- Otros idiomas y acentos no representados en los positivos.
- Detección continua en un dispositivo con AGC activo: el AGC cambia la señal en el tiempo y el modelo
  no está entrenado para eso. Usar `VOICE_RECOGNITION` o `UNPROCESSED`.
- Uso forense o de identificación de hablantes: no identifica personas.

## Limitaciones declaradas

1. Las 10 grabaciones de voz real son de **una sola persona**, una sola sesión y un micrófono de
   laptop. El 100% de recall refleja esa condición, no la población.
2. El test de 53 h se ha mirado durante el desarrollo: es un **test de desarrollo**. La cifra
   publicable necesita grabaciones que nunca se hayan usado para decidir nada.
3. El campo lejano es **simulado** (RIR de 0,5 s, paso bajo 5 kHz, -18 dBFS), no grabado a 3 m.
4. La cabeza depende del embedding congelado de Google, entrenado predominantemente en inglés. Si el
   techo aparece ahí, el siguiente paso es entrenar una CNN sobre mel (candidatos A/B en
   `DECISIONES.md`).

## Procedencia

Código Apache-2.0. Positivos TTS: Kokoro-82M (Apache-2.0) y Chatterbox (MIT). Negativos: Common Voice
27.0 (CC0-1.0, no redistribuido). Ruido: MUSAN (CC BY 4.0). Voz del usuario: grabaciones propias con
consentimiento implícito de uso para el proyecto.
