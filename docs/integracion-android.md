# Integración en Android (y en cualquier cliente)

Todo lo que sigue sale de mediciones hechas en este repositorio, no de la documentación de terceros.
Los tres tropiezos que se describen aquí costaron una tarde cada uno.

## La cadena real son tres modelos

El `.onnx` que se entrena aquí **no recibe audio**. Recibe características. En el dispositivo hay que
encadenar:

```
PCM 16 kHz mono
   └─ melspectrogram.onnx      → espectrograma
        └─ embedding_model.onnx → vectores de 96 dimensiones cada 80 ms
             └─ cabeza.onnx     → una puntuación por ventana de 16 frames (1,28 s)
```

Las tres piezas vienen del paquete `openwakeword` (Apache-2.0 en código) y se pueden empaquetar como
assets. La cabeza es la que produce este repositorio.

## Tropezón 1: la escala del mel

El grafo `melspectrogram.onnx` **espera enteros de 16 bits convertidos a float** (rango -32768 a
32767), no flotantes normalizados entre -1 y 1. Si se le pasa `float32` dividido entre 32768, el modelo
queda sordo y la cabeza nunca activa: el síntoma es "no me reconoce" con un pipeline que parece
correcto.

```kotlin
// Kotlin: del AudioRecord (PCM16) al mel
val muestras = ShortArray(n)
val comoFloat = FloatArray(n) { muestras[it].toFloat() }   // escala 32768, NO normalizada
```

## Tropezón 2: el AGC de Android mata la señal

Con `MediaRecorder.AudioSource.MIC` o `DEFAULT`, Android aplica control automático de ganancia: sube
la ganancia al máximo en silencio y comprime la voz. El modelo, entrenado con niveles estables, ve una
señal que no reconoce. Usar:

```kotlin
MediaRecorder.AudioSource.VOICE_RECOGNITION   // sin AGC, sin supresión de ruido agresiva
// o UNPROCESSED si el dispositivo lo soporta
```

Medido en este proyecto: las grabaciones de prueba tenían pico entre -16 y -28 dBFS. Un móvil cerca de
la boca da -3 a -12. Una diferencia de 20 dB no rompe el modelo si el entrenamiento incluye
aumentación de nivel (que aquí está), pero el AGC sí lo rompe porque cambia la señal **en el tiempo**,
no solo el nivel.

## Tropezón 3: la ventana de 16 frames no cubre 1,28 s de audio

Cada frame de embedding consume 76 frames de mel (0,76 s) y avanza 8 (0,08 s). Por eso:

- Una ventana de 16 frames cubre **2,04 s de audio**, no 1,28 s.
- Un clip de menos de ~2 s produce menos de 16 frames y **no se puede evaluar**.
- Al etiquetar positivos hay que exigir que la frase completa quepa en la ventana. Si se etiquetan
  todas las ventanas del clip, el modelo aprende que media frase basta y dispara con "celeste" solo:
  medido, 0,773 FA/h frente a 0,019.

## Umbral y capa anti-disparo

Lo que se midió con el modelo final (few-shot, ruido, negativos minados) sobre 53 h de habla real:

| Umbral | Falsos positivos | Recall en muestra |
|---|---|---|
| 0,50 | 0,396 FA/h | 100% |
| 0,90 | 0,151 FA/h | 100% |

Recomendación: **umbral 0,9 con al menos 3 frames consecutivos sobre el umbral y 1 s de enfriamiento**.
Los frames consecutivos son lo que evita el "disparo por un pico" y el enfriamiento evita ráfagas de
eventos por una sola palabra.

Además, antes de evaluar la red conviene una compuerta de energía: si el RMS del frame está por debajo
de -45 dBFS, forzar puntuación 0. Cuesta nada y elimina los disparos en silencio digital.

## Qué NO prometer

- Este modelo **requiere grabaciones del usuario**. Un modelo entrenado solo con TTS no reconoce a una
  persona real: medido, recall 0 de 10 con grabaciones reales, incluso simulando micrófonos.
- El test de 53 h es un test de desarrollo: se ha mirado durante la construcción. La cifra publicable
  necesita grabaciones nuevas.
