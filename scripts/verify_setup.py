"""
Script de verificación: Validar que todo está listo para entrenamiento
"""
import os
import sys
from pathlib import Path
import yaml


def check_file_exists(filepath, description):
    """Verificar que un archivo existe"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists


def check_directory_exists(dirpath, description):
    """Verificar que un directorio existe"""
    exists = os.path.isdir(dirpath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dirpath}")
    return exists


def check_python_packages():
    """Verificar que los paquetes necesarios están instalados"""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'PIL', 'sklearn',
        'albumentations', 'yaml', 'pandas', 'matplotlib', 'seaborn'
    ]
    
    print("\n📦 Verificando paquetes Python:")
    print("-" * 60)
    
    all_installed = True
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'yaml':
                __import__('yaml')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NO INSTALADO")
            all_installed = False
    
    return all_installed


def verify_setup():
    """Verificar configuración completa"""
    print("=" * 60)
    print("VERIFICACIÓN DE CONFIGURACIÓN - BirdID-Piciformes")
    print("=" * 60)
    
    all_ok = True
    
    # 1. Verificar estructura de directorios
    print("\n📁 Verificando estructura de directorios:")
    print("-" * 60)
    
    dirs_to_check = [
        ('configs', 'Directorio de configuración'),
        ('src/data', 'Código de datos'),
        ('src/models', 'Código de modelos'),
        ('src/evaluation', 'Código de evaluación'),
        ('scripts', 'Scripts de entrenamiento'),
        ('data/splits', 'Splits del dataset'),
    ]
    
    for dirpath, desc in dirs_to_check:
        if not check_directory_exists(dirpath, desc):
            all_ok = False
    
    # 2. Verificar archivos clave
    print("\n📄 Verificando archivos clave:")
    print("-" * 60)
    
    files_to_check = [
        ('configs/config.yaml', 'Archivo de configuración'),
        ('src/data/preprocessing.py', 'Script de preprocesamiento'),
        ('src/data/dataset.py', 'Dataset class'),
        ('src/models/models.py', 'Modelos'),
        ('scripts/train_classification.py', 'Script de entrenamiento'),
        ('data/splits/train.txt', 'Split de entrenamiento'),
        ('data/splits/val.txt', 'Split de validación'),
        ('data/splits/test.txt', 'Split de test'),
        ('data/splits/class_mapping.txt', 'Mapeo de clases'),
    ]
    
    for filepath, desc in files_to_check:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    # 3. Verificar dataset
    print("\n📊 Verificando dataset:")
    print("-" * 60)
    
    config_path = 'configs/config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_path = config.get('data', {}).get('source_dir', '')
        if dataset_path:
            if check_directory_exists(dataset_path, 'Dataset original'):
                # Contar clases
                classes = [d for d in os.listdir(dataset_path) 
                          if os.path.isdir(os.path.join(dataset_path, d))]
                print(f"   📌 Clases encontradas: {len(classes)}")
            else:
                all_ok = False
        else:
            print("❌ Ruta del dataset no configurada en config.yaml")
            all_ok = False
    else:
        print("❌ No se puede verificar dataset (config.yaml no existe)")
        all_ok = False
    
    # 4. Verificar paquetes Python
    packages_ok = check_python_packages()
    if not packages_ok:
        all_ok = False
    
    # 5. Verificar GPU (opcional)
    print("\n🖥️  Verificando hardware:")
    print("-" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA no disponible (se usará CPU - más lento)")
    except:
        print("❌ PyTorch no instalado")
        all_ok = False
    
    # Resumen final
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ¡TODO LISTO! Puedes proceder con el entrenamiento.")
        print("\nPróximo paso:")
        print("   python3 scripts/train_classification.py")
    else:
        print("❌ Hay problemas que resolver antes de entrenar.")
        print("\nRevisa los errores arriba y:")
        print("   1. Instala paquetes faltantes: pip install -r requirements.txt")
        print("   2. Ejecuta preprocesamiento: python3 src/data/preprocessing.py")
        print("   3. Verifica rutas en configs/config.yaml")
    print("=" * 60)
    
    return all_ok


if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)

