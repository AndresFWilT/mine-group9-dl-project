# ⚡ Optimizaciones Aplicadas para Acelerar Entrenamiento

## Problema Identificado
El entrenamiento estaba tomando ~25 segundos por batch en CPU, resultando en:
- **17 minutos por época** (41 batches)
- **~28 horas para 100 épocas** (inaceptable)

## Optimizaciones Implementadas

### 1. Modelo Más Pequeño
- **Antes**: EfficientNet-B2 (9.1M parámetros)
- **Ahora**: EfficientNet-B0 (5.3M parámetros)
- **Ganancia**: ~40% más rápido

### 2. Tamaño de Imagen Reducido
- **Antes**: 224×224 píxeles
- **Ahora**: 160×160 píxeles
- **Ganancia**: ~50% menos operaciones por imagen

### 3. Batch Size Aumentado
- **Antes**: 32
- **Ahora**: 64
- **Ganancia**: Mejor utilización de CPU, menos overhead

### 4. Épocas Reducidas
- **Antes**: 100 épocas
- **Ahora**: 30 épocas
- **Ganancia**: 70% menos tiempo total

### 5. Early Stopping Más Agresivo
- **Antes**: patience=15
- **Ahora**: patience=8
- **Ganancia**: Se detiene antes si no mejora

### 6. Augmentations Simplificados
- **Antes**: 7 transformaciones complejas
- **Ahora**: 2 transformaciones simples (flip + brightness/contrast)
- **Ganancia**: ~30% más rápido en preprocesamiento

### 7. Hidden Dimensions Reducidas
- **Antes**: 512 → 256
- **Ahora**: 256 → 128
- **Ganancia**: Menos parámetros en el head

## Tiempo Estimado Ahora

Con estas optimizaciones:
- **Por batch**: ~8-10 segundos (vs 25 antes)
- **Por época**: ~5-6 minutos (vs 17 antes)
- **Total (30 épocas)**: ~2.5-3 horas (vs 28 horas antes)

**Reducción total: ~90% más rápido**

## Trade-offs

### Ventajas:
✅ Entrenamiento factible en CPU
✅ Resultados en tiempo razonable
✅ Modelo aún efectivo (B0 es muy bueno)

### Desventajas:
⚠️ Accuracy puede ser ligeramente menor (pero aún competitiva)
⚠️ Modelo más pequeño (pero suficiente para 13 clases)

## Recomendaciones

1. **Para proyecto final**: Esta configuración es adecuada
2. **Si tienes GPU**: Puedes volver a B2 y 224×224
3. **Si necesitas más precisión**: Entrena más épocas o usa B1

## Configuración Actual

Ver `configs/config.yaml` para valores exactos.

