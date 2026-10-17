# Consentimiento para uso de voz en entrenamiento de modelos

> Plantilla. No es asesoría legal: si el proyecto llega a venderse o a usarse en un
> producto, un abogado debe revisar el texto y el alcance antes de firmar nada.

## Qué se pide y para qué

Los modelos de activación por voz se entrenan con ejemplos de habla. Este proyecto
necesita **muestras de voz de personas reales** (no solo sintéticas) para que la wake word
funcione con acentos reales, en ruido real y con micrófonos reales.

Se pide autorización para:

1. Grabar y usar la voz de la persona firmante para entrenar modelos de reconocimiento de
   palabras de activación.
2. Generar variaciones sintéticas de esa voz (clonación) con modelos de texto a voz, para
   aumentar la diversidad de los datos de entrenamiento.
3. Publicar o no publicar el audio, según la opción marcada abajo.

No se pide: uso de la voz para suplantar a la persona, contenido publicitario,
contenido político, ni nada distinto de datos de entrenamiento técnico.

---

## Ficha

- **Nombre de la persona:** ______________________________________
- **Correo o teléfono de contacto:** ______________________________
- **Fecha:** ____________________
- **Proyecto:** wakeword-es (github.com/di3go35/wakeword-es)
- **Responsable:** Diego — ______________________________________

## Opciones (marca las que autorices)

- [ ] **A. Solo entrenamiento interno.** Mi voz se usa para entrenar modelos, pero el
      audio no se publica ni se redistribuye.
- [ ] **B. Entrenamiento + publicación del audio** bajo licencia
      [ ] CC0 (dominio público)  [ ] CC BY 4.0 (atribución)
      Entiendo que, si marco CC0, cualquiera podrá usar esa grabación incluso con fines
      comerciales.
- [ ] **C. Entrenamiento + clonación de la voz** para generar muestras sintéticas
      adicionales. Elimine esta opción si no la autoriza.

## Límites que puedes poner

- Número máximo de grabaciones: ____________________
- Prohibido el uso en: ______________________________
- Fecha de caducidad de esta autorización (opcional): ____________________

## Revocación

Puedo revocar esta autorización escribiendo a la persona responsable. La revocación aplica
a usos futuros: los modelos ya entrenados no pueden "desaprender", pero el audio se
retirará de cualquier conjunto publicable y no se usará en nuevos entrenamientos.

## Notas sobre datos personales

- Las grabaciones se guardan en un directorio privado, no en el repositorio público.
- En el repositorio se registra únicamente un **hash** que identifica el conjunto, nunca
  el audio ni el nombre.
- Este documento se conserva firmado por ambas partes.

---

**Firma de la persona que autoriza:** ______________________________

**Firma de la persona responsable:** ______________________________
