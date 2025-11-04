# Text Detection and Classification System

Sistema avanzado de detección de texto en imágenes usando OCR (Reconocimiento Óptico de Caracteres) con EasyOCR. Este script analiza imágenes de tiles/capturas de pantalla y clasifica automáticamente aquellas que contienen texto legible.

## 🚀 Características Principales

- **Detección OCR avanzada**: Utiliza EasyOCR para detección de texto de alta precisión
- **Soporte multiidioma**: Compatible con español, inglés y otros idiomas
- **Procesamiento por lotes**: Procesa miles de imágenes automáticamente
- **Filtrado inteligente**: Umbral de confianza configurable para mejorar la precisión
- **Reportes detallados**: Genera informes completos en JSON y TXT
- **Preprocesamiento automático**: Optimiza imágenes para mejor reconocimiento OCR
- **Seguimiento de progreso**: Indicadores visuales del progreso de procesamiento
- **Parámetros simplificados**: Interface limpia sin duplicaciones

## 📋 Requisitos

### Dependencias Python
```bash
pip install easyocr opencv-python pillow numpy
```

### Librerías requeridas
- `easyocr` - Motor OCR principal
- `opencv-python` - Procesamiento de imágenes
- `pillow` - Manipulación de imágenes
- `numpy` - Operaciones matemáticas con arrays

## 🛠️ Instalación

1. **Activar entorno virtual** (recomendado):
```bash
# Windows
venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Verificar instalación**:
```bash
python text_detector.py --help
```

## 📖 Uso

### Comando básico
```bash
python text_detector.py --input "ruta/a/imagenes" --output "carpeta/salida"
```

### Ejemplos de uso

#### Procesamiento básico (configuración por defecto)
```bash
python text_detector.py
```

#### Configuración personalizada - Español con alta confianza
```bash
python text_detector.py --input "..\images-2-tiles\processed_images" \
                       --output "imagenes-con-texto" \
                       --confidence 0.7 \
                       --languages es
```

#### Procesamiento multiidioma
```bash
python text_detector.py --input "source_images" \
                       --output "text_images" \
                       --confidence 0.5 \
                       --languages es en fr
```

#### Modo de prueba
```bash
# Procesar solo 5 imágenes para pruebas
python text_detector.py --test --confidence 0.5
```

#### Procesamiento limitado
```bash
# Procesar solo las primeras 100 imágenes
python text_detector.py --limit 100 --confidence 0.6
```

## ⚙️ Parámetros de Configuración

| Parámetro | Descripción | Valor por defecto | Ejemplo |
|-----------|-------------|------------------|---------|
| `--input` | Directorio de imágenes de entrada | `../images-2-tiles/processed_images` | `--input "mi_carpeta"` |
| `--output` | Directorio de salida para imágenes con texto | `tiles-with-text` | `--output "resultados"` |
| `--output-dir` | Alias para `--output` | - | `--output-dir "salida"` |
| `--confidence` | Umbral de confianza OCR (0.0-1.0) | `0.5` | `--confidence 0.7` |
| `--languages` | Idiomas para OCR | `['en', 'es']` | `--languages es en fr` |
| `--limit` | Número máximo de imágenes a procesar | `None` | `--limit 50` |
| `--test` | Modo de prueba (procesa solo 5 imágenes) | `False` | `--test` |

## 📊 Tipos de Archivos Soportados

- **Imágenes**: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`
- **Procesamiento**: Automático de archivos en mayúsculas y minúsculas
- **Exclusiones**: Se ignoran archivos `*_metadata.json`

## 📈 Salidas y Reportes

### Estructura de salida
```
text-detector-classificator/
├── imagenes-con-texto/           # Imágenes con texto detectado
│   ├── imagen1.png
│   ├── imagen2.jpg
│   └── images_with_text.txt      # Lista de imágenes procesadas
├── reports/                      # Reportes detallados
│   └── text_detection_report_YYYYMMDD_HHMMSS.json
└── logs/                         # Logs de procesamiento
    └── text_detection_YYYYMMDD_HHMMSS.log
```

### Contenido del reporte JSON
```json
{
  "execution_info": {
    "timestamp": "2025-11-02T11:32:48",
    "source_directory": "ruta/entrada",
    "output_directory": "ruta/salida",
    "confidence_threshold": 0.5,
    "languages": ["es"]
  },
  "statistics": {
    "total_images": 2726,
    "images_with_text": 1376,
    "images_without_text": 1350,
    "errors": 0,
    "processing_time": 1313.74
  },
  "performance": {
    "avg_time_per_image": 0.48,
    "images_per_second": 2.07
  },
  "images_with_text": [
    {
      "filename": "ejemplo.png",
      "text_detected": [
        {
          "text": "Texto detectado",
          "confidence": 0.85,
          "bbox": [[x1, y1], [x2, y2], ...]
        }
      ],
      "combined_text": "Todo el texto detectado"
    }
  ]
}
```

## 🔧 Configuración Avanzada

### Ajuste de confianza
- **0.3-0.4**: Detección más permisiva (más falsos positivos)
- **0.5-0.6**: Balance entre precisión y cobertura (recomendado)
- **0.7-0.9**: Alta precisión (puede perder texto de baja calidad)

### Optimización de idiomas
- **Español**: `--languages es`
- **Inglés**: `--languages en`
- **Multiidioma**: `--languages es en fr de`
- Ver [códigos de idioma EasyOCR](https://github.com/JaidedAI/EasyOCR#supported-languages) para más opciones

### Preprocesamiento automático
El script incluye optimizaciones automáticas:
- Redimensionado de imágenes pequeñas
- Conversión de color para mejor OCR
- Filtrado de ruido en detecciones

## 🐛 Solución de Problemas

### Error: "No module named 'easyocr'"
```bash
pip install easyocr
```

### Error: "CUDA not available"
EasyOCR funcionará en CPU automáticamente. Para usar GPU:
```bash
pip install torch torchvision
```

### Rendimiento lento
- Reduce el número de idiomas: `--languages es`
- Usa el parámetro `--limit` para pruebas
- Considera usar GPU si está disponible

### Memoria insuficiente
- Procesa en lotes más pequeños usando `--limit`
- Cierra otras aplicaciones que consuman memoria
- Usa imágenes de menor resolución si es posible

## 📋 Casos de Uso

### 1. Clasificación de capturas de pantalla
```bash
python text_detector.py --input "screenshots" --output "text_screenshots" --confidence 0.6
```

### 2. Análisis de tiles de aplicaciones
```bash
python text_detector.py --input "app_tiles" --output "tiles_with_text" --languages en es
```

### 3. Procesamiento de documentos escaneados
```bash
python text_detector.py --input "scanned_docs" --output "text_docs" --confidence 0.8
```

### 4. Detección en múltiples idiomas
```bash
python text_detector.py --input "multilang_images" --output "detected_text" --languages es en fr de
```

## 📊 Métricas de Rendimiento

En un procesamiento típico de 2,726 imágenes:
- **Tiempo total**: ~22 minutos
- **Velocidad**: ~2 imágenes/segundo
- **Tasa de detección**: 50.5%
- **Precisión**: 100% (sin errores)

## 🔄 Historial de Cambios

### Versión 2.0 (Noviembre 2025)
- ✅ **NUEVO**: Eliminación de parámetros duplicados (`--input-dir`, `--source`)
- ✅ **MEJORADO**: Optimización del parámetro `--input` único
- ✅ **MEJORADO**: Interface más limpia sin redundancias
- ✅ **VALIDADO**: Prueba exhaustiva con 2,726 imágenes
- ✅ **ACTUALIZADO**: Documentación completa y ejemplos

### Características eliminadas (simplificación)
- ❌ `--input-dir` (usar `--input`)
- ❌ `--source` (usar `--input`)

### Versión 1.x
- Implementación inicial con EasyOCR
- Soporte multiidioma básico
- Generación de reportes JSON
- Múltiples parámetros de entrada (redundantes)

## 💡 Mejores Prácticas

### Para obtener mejores resultados:
1. **Usa rutas absolutas** para evitar confusiones
2. **Comienza con confianza 0.5** y ajusta según resultados
3. **Especifica solo idiomas necesarios** para mejor rendimiento
4. **Usa `--test` primero** para validar configuración
5. **Revisa los logs** para identificar problemas

### Flujo de trabajo recomendado:
```bash
# 1. Prueba rápida
python text_detector.py --test --confidence 0.5

# 2. Procesamiento limitado
python text_detector.py --limit 50 --confidence 0.6

# 3. Procesamiento completo
python text_detector.py --confidence 0.6 --languages es
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) por el excelente motor OCR
- OpenCV por las herramientas de procesamiento de imágenes
- La comunidad de Python por las librerías utilizadas

## 📞 Soporte

Para reportar bugs o solicitar features:
- Crea un issue en el repositorio
- Incluye información del sistema y logs de error
- Proporciona ejemplos de imágenes problemáticas (si es posible)

---

**Desarrollado con ❤️ para el Sistema de Reconstrucción de Tiles de Capturas de Pantalla de Windows**
