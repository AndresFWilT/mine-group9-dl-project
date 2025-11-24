"""
Script de entrenamiento para clasificación de Piciformes - Versión Google Colab
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import shutil
import subprocess

try:
    script_dir = Path(__file__).parent.parent
    if script_dir.exists():
        sys.path.append(str(script_dir))
except NameError:
    pass

possible_paths = [
    '/content',
    '/content/mine-group9-dl-project',
    '/content/drive/MyDrive/mine-group9-dl-project',
    '/content/drive/MyDrive/Deep Learning/mine-group9-dl-project',
    os.getcwd(),
]

for path in possible_paths:
    if path and os.path.exists(path):
        if path not in sys.path:
            sys.path.insert(0, path)

def find_and_import_model():
    """Buscar y importar create_model_from_config desde diferentes ubicaciones"""
    possible_locations = [
        '/content/mine-group9-dl-project/src/models/models.py',
        '/content/drive/MyDrive/mine-group9-dl-project/src/models/models.py',
        '/content/drive/MyDrive/Deep Learning/mine-group9-dl-project/src/models/models.py',
        'src/models/models.py',
        '../src/models/models.py',
        '../../src/models/models.py',
    ]
    
    for model_path in possible_locations:
        if os.path.exists(model_path):
            model_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(model_path))))
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            try:
                from src.models.models import create_model_from_config
                return create_model_from_config
            except ImportError:
                continue
    
    try:
        from src.models.models import create_model_from_config
        return create_model_from_config
    except ImportError:
        pass
    
    return None

create_model_from_config = find_and_import_model()

if create_model_from_config is None:
    import ssl
    import warnings
    
    class EfficientNetClassifier(nn.Module):
        """EfficientNet para clasificación de Piciformes"""
        def __init__(self, model_name='efficientnet_b2', num_classes=13, pretrained=True,
                     dropout1=0.5, dropout2=0.3, hidden_dim1=512, hidden_dim2=256):
            super().__init__()
            weights = 'DEFAULT' if pretrained else None
            
            if pretrained:
                old_ssl = ssl._create_default_https_context
                try:
                    ssl._create_default_https_context = ssl._create_unverified_context
                    
                    if model_name == 'efficientnet_b0':
                        self.backbone = torchvision.models.efficientnet_b0(weights=weights)
                        in_features = self.backbone.classifier[1].in_features
                    elif model_name == 'efficientnet_b2':
                        self.backbone = torchvision.models.efficientnet_b2(weights=weights)
                        in_features = self.backbone.classifier[1].in_features
                    elif model_name == 'efficientnet_b3':
                        self.backbone = torchvision.models.efficientnet_b3(weights=weights)
                        in_features = self.backbone.classifier[1].in_features
                    else:
                        raise ValueError(f"Modelo {model_name} no soportado")
                    
                    ssl._create_default_https_context = old_ssl
                except Exception as e:
                    ssl._create_default_https_context = old_ssl
                    weights = None
                    
                    if model_name == 'efficientnet_b0':
                        self.backbone = torchvision.models.efficientnet_b0(weights=None)
                        in_features = self.backbone.classifier[1].in_features
                    elif model_name == 'efficientnet_b2':
                        self.backbone = torchvision.models.efficientnet_b2(weights=None)
                        in_features = self.backbone.classifier[1].in_features
                    elif model_name == 'efficientnet_b3':
                        self.backbone = torchvision.models.efficientnet_b3(weights=None)
                        in_features = self.backbone.classifier[1].in_features
                    else:
                        raise ValueError(f"Modelo {model_name} no soportado")
            else:
                if model_name == 'efficientnet_b0':
                    self.backbone = torchvision.models.efficientnet_b0(weights=None)
                    in_features = self.backbone.classifier[1].in_features
                elif model_name == 'efficientnet_b2':
                    self.backbone = torchvision.models.efficientnet_b2(weights=None)
                    in_features = self.backbone.classifier[1].in_features
                elif model_name == 'efficientnet_b3':
                    self.backbone = torchvision.models.efficientnet_b3(weights=None)
                    in_features = self.backbone.classifier[1].in_features
                else:
                    raise ValueError(f"Modelo {model_name} no soportado")
            
            self.backbone.classifier = nn.Identity()
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features, hidden_dim1),
                nn.BatchNorm1d(hidden_dim1),
                nn.ReLU(),
                nn.Dropout(dropout1),
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.BatchNorm1d(hidden_dim2),
                nn.ReLU(),
                nn.Dropout(dropout2),
                nn.Linear(hidden_dim2, num_classes)
            )
        
        def forward(self, x):
            x = self.backbone.features(x)
            x = self.classifier(x)
            return x
    
    class ResNetClassifier(nn.Module):
        """ResNet50 para clasificación de Piciformes"""
        def __init__(self, num_classes=13, pretrained=True,
                     dropout1=0.5, dropout2=0.3, hidden_dim1=512, hidden_dim2=256):
            super().__init__()
            
            if pretrained:
                old_ssl = ssl._create_default_https_context
                try:
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self.backbone = torchvision.models.resnet50(weights='DEFAULT')
                    ssl._create_default_https_context = old_ssl
                except Exception as e:
                    ssl._create_default_https_context = old_ssl
                    self.backbone = torchvision.models.resnet50(weights=None)
            else:
                self.backbone = torchvision.models.resnet50(weights=None)
            
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
            self.classifier = nn.Sequential(
                nn.Linear(in_features, hidden_dim1),
                nn.BatchNorm1d(hidden_dim1),
                nn.ReLU(),
                nn.Dropout(dropout1),
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.BatchNorm1d(hidden_dim2),
                nn.ReLU(),
                nn.Dropout(dropout2),
                nn.Linear(hidden_dim2, num_classes)
            )
        
        def forward(self, x):
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    def create_model_from_config(config):
        """Crear modelo según configuración"""
        import torchvision
        model_config = config['model']
        architecture = model_config['architecture']
        num_classes = config['data']['num_classes']
        
        if architecture.startswith('efficientnet'):
            return EfficientNetClassifier(
                model_name=architecture,
                num_classes=num_classes,
                pretrained=model_config['pretrained'],
                dropout1=model_config['dropout_rate_1'],
                dropout2=model_config['dropout_rate_2'],
                hidden_dim1=model_config['hidden_dim_1'],
                hidden_dim2=model_config['hidden_dim_2']
            )
        elif architecture == 'resnet50':
            return ResNetClassifier(
                num_classes=num_classes,
                pretrained=model_config['pretrained'],
                dropout1=model_config['dropout_rate_1'],
                dropout2=model_config['dropout_rate_2'],
                hidden_dim1=model_config['hidden_dim_1'],
                hidden_dim2=model_config['hidden_dim_2']
            )
        else:
            raise ValueError(f"Arquitectura {architecture} no soportada")
    
    create_model_from_config = create_model_from_config


class SubsetDataset(torch.utils.data.Dataset):
    """Aplica transformaciones a subconjuntos del dataset para train/val/test"""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
    
    def __getitem__(self, idx):
        path, target = self.dataset.samples[self.indices[idx]]
        image = self.dataset.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, target
    
    def __len__(self):
        return len(self.indices)


def load_config(config_path=None):
    """Carga configuración YAML o usa valores por defecto optimizados"""
    if config_path is None:
        possible_paths = [
            '/content/mine-group9-dl-project/configs/config_high_accuracy.yaml',
            '/content/mine-group9-dl-project/configs/config.yaml',
            '/content/configs/config_high_accuracy.yaml',
            '/content/configs/config.yaml',
            './configs/config_high_accuracy.yaml',
            './configs/config.yaml',
            'configs/config_high_accuracy.yaml',
            'configs/config.yaml'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config():
    """Configuración optimizada para máxima accuracy: EfficientNet-B2, 256x256, augmentations completos"""
    return {
        'data': {
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'image_size': 256,  # Aumentado para más detalles
            'num_classes': 13,
            'seed': 42
        },
        'model': {
            'architecture': 'efficientnet_b2',
            'pretrained': True,
            'dropout_rate_1': 0.4,  # Reducido ligeramente
            'dropout_rate_2': 0.2,
            'hidden_dim_1': 512,
            'hidden_dim_2': 256
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,  # Más épocas para convergencia
            'early_stopping_patience': 20,  # Más paciencia
            'learning_rate': 0.0003,  # LR inicial ligeramente mayor
            'weight_decay': 0.0001,
            'optimizer': 'adamw',
            'label_smoothing': 0.05,  # Reducido
            'use_class_weights': True,
            'lr_schedule': {
                'type': 'cosine',
                'min_lr': 1e-7
            }
        },
        'paths': {
            'models_dir': '/content/models',
            'results_dir': '/content/results',
            'logs_dir': '/content/logs'
        }
    }


def setup_data_from_archive(archive_path=None, extract_to='/content/Data_Esp_Pic'):
    """Extrae imágenes desde .7z o .zip ubicado en Google Drive"""
    if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:
        return extract_to
    
    if archive_path is None:
        possible_archives = [
            '/content/drive/MyDrive/Deep Learning/Data_Esp_Pic.7z',
            '/content/drive/MyDrive/Deep Learning/Data_Esp_Pic.zip',
            '/content/drive/MyDrive/Data_Esp_Pic.7z',
            '/content/drive/MyDrive/Data_Esp_Pic.zip',
            '/content/Data_Esp_Pic.7z',
            '/content/Data_Esp_Pic.zip'
        ]
        for arch in possible_archives:
            if os.path.exists(arch):
                archive_path = arch
                break
    
    if archive_path is None or not os.path.exists(archive_path):
        return None
    
    print(f"Extrayendo {archive_path}...")
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        if archive_path.endswith('.7z'):
            try:
                subprocess.run(['7z'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                subprocess.run(['apt-get', 'update'], check=True, capture_output=True)
                subprocess.run(['apt-get', 'install', '-y', 'p7zip-full'], check=True, capture_output=True)
            
            result = subprocess.run(
                ['7z', 'x', archive_path, f'-o{extract_to}', '-y'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise Exception(f"Error al extraer 7z: {result.stderr}")
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        
        contents = os.listdir(extract_to)
        if len(contents) == 1 and os.path.isdir(os.path.join(extract_to, contents[0])):
            inner_dir = os.path.join(extract_to, contents[0])
            for item in os.listdir(inner_dir):
                shutil.move(os.path.join(inner_dir, item), os.path.join(extract_to, item))
            os.rmdir(inner_dir)
        
        return extract_to
    except Exception as e:
        print(f"Error al extraer: {e}")
        return None


def get_data_loaders(data_dir, batch_size=32, image_size=224, train_split=0.7, val_split=0.15, test_split=0.15, seed=42):
    """
    Crear data loaders para entrenamiento, validación y test
    
    Estructura esperada: data_dir/Clase1/, data_dir/Clase2/, ...
    Divide automáticamente en 70% train, 15% val, 15% test
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Directorio no encontrado: {data_dir}")
    
    # Augmentations completos para máxima accuracy
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),  # Resize más grande primero
        transforms.RandomCrop(image_size),  # Crop aleatorio
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),  # Aves pueden estar invertidas
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))  # Regularización adicional
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(root=data_dir)
    idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
    num_classes = len(full_dataset.classes)
    
    print(f"Dataset: {len(full_dataset)} imagenes, {num_classes} clases")
    
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_test_indices = random_split(
        range(total_size),
        [train_size, val_size + test_size],
        generator=generator
    )
    
    val_indices, test_indices = random_split(
        val_test_indices,
        [val_size, test_size],
        generator=generator
    )
    
    train_dataset = SubsetDataset(full_dataset, train_indices, train_transform)
    val_dataset = SubsetDataset(full_dataset, val_indices, val_transform)
    test_dataset = SubsetDataset(full_dataset, test_indices, val_transform)
    
    num_workers = 2 if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    print(f"Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, idx_to_class, train_dataset


def calculate_class_weights(train_dataset, num_classes):
    """Calcula pesos inversamente proporcionales a frecuencia de clases para balanceo"""
    class_counts = torch.zeros(num_classes)
    
    for idx in tqdm(range(len(train_dataset)), desc="Contando clases"):
        _, label = train_dataset[idx]
        class_counts[label] += 1
    
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)
    weights = weights / weights.sum() * num_classes
    
    return weights


def freeze_backbone(model):
    """Congela backbone, solo entrena head"""
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    print("Backbone congelado, solo entrenando head")


def unfreeze_backbone(model):
    """Descongela todo el modelo"""
    for param in model.parameters():
        param.requires_grad = True
    print("Todo el modelo descongelado para fine-tuning")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Ejecuta una época: forward pass, cálculo de loss, backward pass, actualización de pesos"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Evalúa modelo en modo evaluación (sin actualizar pesos)"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Genera y guarda matriz de confusión como imagen PNG"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusión guardada en: {save_path}")


def save_model_as_keras(model, save_path, config, class_names):
    """Recrea arquitectura equivalente en Keras y guarda en formato .keras"""
    try:
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            subprocess.run(['pip', 'install', '-q', 'tensorflow'], check=True)
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        
        model_config = config['model']
        architecture = model_config['architecture']
        num_classes = len(class_names)
        image_size = config['data']['image_size']
        
        inputs = keras.Input(shape=(image_size, image_size, 3))
        
        if architecture.startswith('efficientnet'):
            if architecture == 'efficientnet_b0':
                base_model = keras.applications.EfficientNetB0(
                    include_top=False,
                    weights='imagenet' if model_config['pretrained'] else None,
                    input_shape=(image_size, image_size, 3)
                )
            elif architecture == 'efficientnet_b2':
                base_model = keras.applications.EfficientNetB2(
                    include_top=False,
                    weights='imagenet' if model_config['pretrained'] else None,
                    input_shape=(image_size, image_size, 3)
                )
            elif architecture == 'efficientnet_b3':
                base_model = keras.applications.EfficientNetB3(
                    include_top=False,
                    weights='imagenet' if model_config['pretrained'] else None,
                    input_shape=(image_size, image_size, 3)
                )
            else:
                raise ValueError(f"Arquitectura {architecture} no soportada")
            
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
        elif architecture == 'resnet50':
            base_model = keras.applications.ResNet50(
                include_top=False,
                weights='imagenet' if model_config['pretrained'] else None,
                input_shape=(image_size, image_size, 3)
            )
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
        else:
            raise ValueError(f"Arquitectura {architecture} no soportada")
        
        x = layers.Dense(model_config['hidden_dim_1'])(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(model_config['dropout_rate_1'])(x)
        
        x = layers.Dense(model_config['hidden_dim_2'])(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(model_config['dropout_rate_2'])(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        keras_model = keras.Model(inputs, outputs)
        keras_model.save(save_path)
        
        import json
        metadata = {
            'class_names': class_names,
            'num_classes': len(class_names),
            'image_size': config['data']['image_size'],
            'architecture': config['model']['architecture']
        }
        metadata_path = save_path.replace('.keras', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        raise Exception(f"Error al crear modelo Keras: {e}")


def main():
    """Pipeline de entrenamiento mejorado: fine-tuning en dos etapas para máxima accuracy"""
    config = load_config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    
    data_dir = None
    possible_dirs = [
        '/content/Data_Esp_Pic',
        '/content/mine-group9-dl-project/src/data/Data_Esp_Pic',
        '/content/drive/MyDrive/Deep Learning/Data_Esp_Pic',
        '/content/drive/MyDrive/Data_Esp_Pic',
        '/content/src/data/Data_Esp_Pic',
        './Data_Esp_Pic',
        './src/data/Data_Esp_Pic'
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break
    
    if data_dir is None:
        extracted_dir = setup_data_from_archive()
        if extracted_dir:
            data_dir = extracted_dir
    
    if data_dir is None:
        raise ValueError(
            "No se encontro el directorio de datos.\n"
            "Monta Google Drive y ejecuta:\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "Luego ejecuta el script nuevamente."
        )
    
    print(f"Directorio de datos: {data_dir}")
    
    print("\nCargando datasets...")
    train_loader, val_loader, test_loader, idx_to_class, train_dataset = get_data_loaders(
        data_dir=data_dir,
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        seed=config['data']['seed']
    )
    
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Clases: {class_names}")
    
    config['data']['num_classes'] = len(class_names)
    
    print(f"\nCreando modelo: {config['model']['architecture']}")
    model = create_model_from_config(config)
    model = model.to(device)
    
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(train_dataset, config['data']['num_classes'])
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config['training']['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing'])
    
    # ETAPA 1: Entrenar solo Head (backbone congelado)
    print("\n" + "="*60)
    print("ETAPA 1: ENTRENANDO SOLO HEAD (Backbone congelado)")
    print("="*60)
    
    freeze_backbone(model)
    
    # Optimizer solo para head con LR más alto
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, 
                           lr=config['training']['learning_rate'] * 2,
                           weight_decay=config['training']['weight_decay'])
    
    lr_config = config['training']['lr_schedule']
    epochs_stage1 = min(20, config['training']['epochs'] // 4)  # 20 épocas o 25% del total
    if lr_config['type'] == 'cosine':
        min_lr = float(lr_config.get('min_lr', 1e-6))
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_stage1, eta_min=min_lr)
    else:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_acc_stage1 = 0.0
    patience_counter = 0
    
    for epoch in range(epochs_stage1):
        print(f"\nEpoca {epoch+1}/{epochs_stage1} (Etapa 1: Head)")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc_stage1:
            best_val_acc_stage1 = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            print("Early stopping en Etapa 1")
            break
    
    # ETAPA 2: Fine-tuning completo
    print("\n" + "="*60)
    print("ETAPA 2: FINE-TUNING COMPLETO (Todo el modelo)")
    print("="*60)
    
    unfreeze_backbone(model)
    
    # Optimizer para todo el modelo con LR más bajo
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    
    epochs_stage2 = config['training']['epochs'] - epochs_stage1
    if lr_config['type'] == 'cosine':
        min_lr = float(lr_config.get('min_lr', 1e-7))
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_stage2, eta_min=min_lr)
    elif lr_config['type'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_val_acc = best_val_acc_stage1
    best_epoch = 0
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(epochs_stage2):
        print(f"\nEpoca {epoch+1}/{epochs_stage2} (Etapa 2: Fine-tuning)")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if lr_config['type'] == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            model_path = os.path.join(config['paths']['models_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch + epochs_stage1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, model_path)
            print(f"Mejor modelo guardado (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping en epoca {epoch+1}")
            break
    
    print("\n" + "="*60)
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Mejor epoca: {best_epoch+1+epochs_stage1}, Mejor Val Acc: {best_val_acc:.2f}%")
    print("="*60)
    
    print("\nEvaluando en conjunto de test...")
    checkpoint = torch.load(os.path.join(config['paths']['models_dir'], 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    results_dir = config['paths']['results_dir']
    
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    save_confusion_matrix(test_labels, test_preds, class_names, cm_path)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss durante entrenamiento')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.xlabel('Época')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy durante entrenamiento')
    plt.tight_layout()
    training_curves_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(training_curves_path, dpi=300)
    plt.close()
    
    print(f"\nResultados guardados en: {results_dir}")
    
    model_path = os.path.join(config['paths']['models_dir'], 'best_model.pt')
    drive_models_dir = '/content/drive/MyDrive/Deep Learning/models'
    drive_results_dir = '/content/drive/MyDrive/Deep Learning/results'
    
    if os.path.exists('/content/drive'):
        print("\nGuardando en Google Drive...")
        try:
            os.makedirs(drive_models_dir, exist_ok=True)
            os.makedirs(drive_results_dir, exist_ok=True)
            
            drive_model_path = os.path.join(drive_models_dir, 'best_model.pt')
            shutil.copy2(model_path, drive_model_path)
            print(f"Modelo PyTorch guardado: {drive_model_path}")
            
            try:
                drive_keras_path = os.path.join(drive_models_dir, 'best_model.keras')
                save_model_as_keras(model, drive_keras_path, config, class_names)
                print(f"Modelo Keras guardado: {drive_keras_path}")
            except Exception as e:
                print(f"Advertencia: No se pudo guardar Keras: {e}")
            
            shutil.copy2(cm_path, os.path.join(drive_results_dir, 'confusion_matrix.png'))
            shutil.copy2(training_curves_path, os.path.join(drive_results_dir, 'training_curves.png'))
            print(f"Resultados guardados en: {drive_results_dir}")
            
        except Exception as e:
            print(f"Error al guardar en Drive: {e}")


if __name__ == "__main__":
    main()
