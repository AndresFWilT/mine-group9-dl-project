"""
Modelos de clasificación para Piciformes
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
import ssl
import warnings


class EfficientNetClassifier(nn.Module):
    """EfficientNet para clasificación de Piciformes"""
    
    def __init__(self, model_name='efficientnet_b2', num_classes=13, pretrained=True,
                 dropout1=0.5, dropout2=0.3, hidden_dim1=512, hidden_dim2=256):
        super().__init__()
        
        # Intentar cargar modelo pre-entrenado con manejo de errores SSL
        weights = 'DEFAULT' if pretrained else None
        
        if pretrained:
            # Intentar descargar con contexto SSL no verificado (temporal para macOS)
            old_ssl = ssl._create_default_https_context
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
                
                if model_name == 'efficientnet_b0':
                    self.backbone = tv_models.efficientnet_b0(weights=weights)
                    in_features = self.backbone.classifier[1].in_features
                elif model_name == 'efficientnet_b2':
                    self.backbone = tv_models.efficientnet_b2(weights=weights)
                    in_features = self.backbone.classifier[1].in_features
                elif model_name == 'efficientnet_b3':
                    self.backbone = tv_models.efficientnet_b3(weights=weights)
                    in_features = self.backbone.classifier[1].in_features
                else:
                    raise ValueError(f"Modelo {model_name} no soportado")
                
                # Restaurar contexto SSL
                ssl._create_default_https_context = old_ssl
                print(f"✅ Modelo pre-entrenado {model_name} cargado correctamente")
                
            except Exception as e:
                # Restaurar contexto SSL
                ssl._create_default_https_context = old_ssl
                print(f"⚠️  Error descargando modelo pre-entrenado: {e}")
                print("⚠️  Usando modelo sin pre-entrenar (weights=None)")
                weights = None
                
                if model_name == 'efficientnet_b0':
                    self.backbone = tv_models.efficientnet_b0(weights=None)
                    in_features = self.backbone.classifier[1].in_features
                elif model_name == 'efficientnet_b2':
                    self.backbone = tv_models.efficientnet_b2(weights=None)
                    in_features = self.backbone.classifier[1].in_features
                elif model_name == 'efficientnet_b3':
                    self.backbone = tv_models.efficientnet_b3(weights=None)
                    in_features = self.backbone.classifier[1].in_features
                else:
                    raise ValueError(f"Modelo {model_name} no soportado")
        else:
            # Sin pre-entrenar
            if model_name == 'efficientnet_b0':
                self.backbone = tv_models.efficientnet_b0(weights=None)
                in_features = self.backbone.classifier[1].in_features
            elif model_name == 'efficientnet_b2':
                self.backbone = tv_models.efficientnet_b2(weights=None)
                in_features = self.backbone.classifier[1].in_features
            elif model_name == 'efficientnet_b3':
                self.backbone = tv_models.efficientnet_b3(weights=None)
                in_features = self.backbone.classifier[1].in_features
            else:
                raise ValueError(f"Modelo {model_name} no soportado")
        
        # Reemplazar clasificador
        self.backbone.classifier = nn.Identity()
        
        # Head personalizado
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
        
        # Backbone pre-entrenado con manejo de errores SSL
        if pretrained:
            old_ssl = ssl._create_default_https_context
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
                self.backbone = tv_models.resnet50(weights='DEFAULT')
                ssl._create_default_https_context = old_ssl
                print("✅ Modelo pre-entrenado ResNet50 cargado correctamente")
            except Exception as e:
                ssl._create_default_https_context = old_ssl
                print(f"⚠️  Error descargando modelo pre-entrenado: {e}")
                print("⚠️  Usando modelo sin pre-entrenar (weights=None)")
                self.backbone = tv_models.resnet50(weights=None)
        else:
            self.backbone = tv_models.resnet50(weights=None)
        
        in_features = self.backbone.fc.in_features
        
        # Reemplazar fc
        self.backbone.fc = nn.Identity()
        
        # Head personalizado
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