# wakeword-es

Pipeline abierto para entrenar **wake words en español latinoamericano** con falsos
positivos **medidos y reproducibles**, y con procedencia de datos auditable.

Nace de un problema concreto y vivo: los modelos de wake word actuales están entrenados
sobre un backbone de embeddings cuyo idioma dominante es el inglés, y sus datos negativos
provienen de corpus con licencias restrictivas. En la práctica eso se traduce en dos
síntomas que cualquiera que haya probado estos modelos reconoce: **se activan sin que nadie
hable** y **no reconocen a la persona que sí dijo la frase**.

## Qué hace este repositorio (y qué no)

Hace:

- Un **banco de evaluación** que convierte el criterio de éxito en código ejecutable
  (`benchmark/`), con test congelado por hash, curva DET completa y prueba de silencio.
- Un pipeline de datos con **registro de procedencia**: cada fuente con su licencia, su
  hash y si su uso comercial es viable o no.
- Un comando de **minado de falsos positivos**: recolectar los disparos reales y reentrenar
  con ellos.

No hace (todavía): prometer precisión, publicar cifras sin poder reproducirlas, ni
compararse con nadie sin correr el mismo test.

## Criterio de éxito

Sobre la batería congelada, en tres condiciones (limpio, TV/música a ~10 dB SNR, campo
lejano a 3 m):

- **≥95% de recall a ≤0,5 falsos positivos por hora**, exigido en cada condición.
- **0 activaciones en 30 minutos de silencio digital.**
- Curva DET completa publicada; baseline honesto corrido con el mismo test.

## Estado

| Fase | Qué es | Estado |
|---|---|---|
| 1 | Aprobación del plan | **Hecha** (2026-10-17) |
| 2 | Auditoría de arquitectura y línea base | En curso — falta correr la línea base |
| 3 | Frase elegida por medición | Pendiente |
| 4 | Corpus y registro de procedencia | Pendiente |
| 5 | Test congelado | Pendiente |
| 6–8 | Modelo, calibración, checkpoint publicable | Pendiente |
| 9–10 | Demo Android, publicación | Pendiente |

Documentación de trabajo: [`PLAN.md`](PLAN.md) · [`DECISIONES.md`](DECISIONES.md) ·
[`ARQUITECTURA.md`](ARQUITECTURA.md) · [`INVENTARIO.md`](INVENTARIO.md).

## Uso

```bash
make test       # pruebas del banco de evaluación (fixtures sintéticos)
make autotest   # demuestra que el criterio puede cumplirse y puede fallar
```

## Licencia

Código bajo Apache-2.0 (ver `LICENSE` y `NOTICE`). Los datos y modelos que se produzcan
aquí llevan su propia licencia, declarada en el registro de procedencia.
Este repositorio **no redistribuye** audio de Common Voice ni de otras fuentes: cada
usuario descarga sus fuentes y este proyecto aporta los scripts y los manifiestos.

## Por qué no hay un "mejor motor" en esta portada

Porque la comparación la dirá el test, no el entusiasmo. Tres candidatos están definidos en
`DECISIONES.md` (fila 15) y se elegirán por medición contra el mismo test congelado. Si el
candidato que empieza primero pierde, se publica igual.
