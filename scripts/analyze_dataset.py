"""
Script para análisis rápido del dataset
"""
import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_dataset(source_dir):
    """Analizar distribución del dataset"""
    print("="*60)
    print("ANÁLISIS DEL DATASET - PICIFORMES")
    print("="*60)
    
    # Obtener clases
    classes = sorted([d for d in os.listdir(source_dir) 
                     if os.path.isdir(os.path.join(source_dir, d))])
    
    # Contar imágenes por clase
    class_counts = {}
    total_images = 0
    
    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(images)
        class_counts[class_name] = count
        total_images += count
    
    print(f"\nTotal de clases: {len(classes)}")
    print(f"Total de imágenes: {total_images}")
    print(f"\nDistribución por clase:")
    print("-"*60)
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_images) * 100
        print(f"{class_name:30s}: {count:4d} ({percentage:5.2f}%)")
    
    # Estadísticas
    counts = list(class_counts.values())
    print(f"\nEstadísticas:")
    print(f"  Mínimo: {min(counts)}")
    print(f"  Máximo: {max(counts)}")
    print(f"  Promedio: {sum(counts)/len(counts):.1f}")
    print(f"  Desviación estándar: {(sum((x - sum(counts)/len(counts))**2 for x in counts) / len(counts))**0.5:.1f}")
    
    # Visualización
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_names = [c[0].replace('_', ' ')[:20] for c in sorted_classes]
    counts = [c[1] for c in sorted_classes]
    
    plt.barh(range(len(class_names)), counts)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel('Número de Imágenes')
    plt.title('Distribución de Imágenes por Clase')
    plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    plt.hist(counts, bins=10, edgecolor='black')
    plt.xlabel('Número de Imágenes')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de Distribución')
    
    plt.tight_layout()
    
    # Guardar
    output_path = "results/dataset_analysis.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGráfico guardado en: {output_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("Análisis completado!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
    else:
        source_dir = "/Users/jnsilvag/Downloads/Data_Esp_Pic"
    
    analyze_dataset(source_dir)

