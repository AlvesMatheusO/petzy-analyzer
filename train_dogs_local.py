# -*- coding: utf-8 -*-
"""🐕 TREINO DE CÃES COM 5 CLASSES - ESCALA CSU (0-4) - VERSÃO LOCAL"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Evita warning do matplotlib em modo não interativo
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import pickle
from glob import glob

# ============================================
# CONFIGURAÇÕES (para execução local)
# ============================================
# Diretório base do projeto (onde este script está)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Subpastas dentro do projeto
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SPECIES = "dogs"
NUM_CLASSES = 5   # CSU: 0,1,2,3,4

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 15

# Diretórios de treino e validação
train_dir = os.path.join(DATASET_DIR, SPECIES, "train")
val_dir = os.path.join(DATASET_DIR, SPECIES, "validation")

# Diretórios de saída
OUTPUT_MODELS_DIR = os.path.join(BASE_DIR, "models_pain")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Criar pastas se não existirem
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================
# VERIFICAR DIRETÓRIOS
# ============================================
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError(
        f"Pastas não encontradas.\n"
        f"Esperado: {train_dir}\n"
        f"e {val_dir}\n"
        f"Certifique-se de que o dataset está em dataset/dogs/train/0/,1/,2/,3/,4/ ..."
    )

def contar_imagens(base_path):
    contagem = {}
    for c in range(NUM_CLASSES):
        path = os.path.join(base_path, str(c))
        if os.path.exists(path):
            n = len(glob(os.path.join(path, "*.*")))
            contagem[c] = n
    return contagem

train_counts = contar_imagens(train_dir)
val_counts = contar_imagens(val_dir)

# ============================================
# ANÁLISE DO DESBALANCEAMENTO
# ============================================
print("\n" + "="*60)
print("🐕 CÃES - ESCALA CSU (0-4)")
print("="*60)
print("\n📊 TREINO:")
total_train = sum(train_counts.values())
for c in range(NUM_CLASSES):
    qtd = train_counts.get(c, 0)
    pct = (qtd / total_train) * 100 if total_train else 0
    bar = "█" * int(pct/2) + "░" * (50 - int(pct/2))
    print(f"  [{bar}] Classe {c}: {qtd:4d} ({pct:5.1f}%)")

print("\n📊 VALIDAÇÃO:")
total_val = sum(val_counts.values())
for c in range(NUM_CLASSES):
    qtd = val_counts.get(c, 0)
    pct = (qtd / total_val) * 100 if total_val else 0
    bar = "█" * int(pct/2) + "░" * (50 - int(pct/2))
    print(f"  [{bar}] Classe {c}: {qtd:4d} ({pct:5.1f}%)")

# ============================================
# GERADORES COM DATA AUGMENTATION
# ============================================
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='reflect'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

# ============================================
# PESOS DAS CLASSES (reforço para a minoria)
# ============================================
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)

enhanced_weights = class_weights.copy()
min_class_count = min(train_counts.values())
for i, count in train_counts.items():
    if count < min_class_count * 1.5:  # classes com poucos exemplos
        enhanced_weights[i] *= 2.5
    elif count < min_class_count * 3:
        enhanced_weights[i] *= 1.5

# Normalizar para que a soma dos pesos seja NUM_CLASSES
enhanced_weights = enhanced_weights / enhanced_weights.sum() * NUM_CLASSES
class_weight_dict = dict(enumerate(enhanced_weights))

print("\n⚖️  PESOS DAS CLASSES (ajustados para desbalanceamento):")
for i in range(NUM_CLASSES):
    print(f"  Classe {i}: {class_weight_dict[i]:.4f}")

# ============================================
# CONSTRUÇÃO DO MODELO (5 classes)
# ============================================
print("\n🏗️  CONSTRUINDO MODELO EfficientNetV2S")
base = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(base.input, out)
model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ============================================
# CALLBACKS
# ============================================
callbacks = [
    ModelCheckpoint(os.path.join(CHECKPOINT_DIR, f"{SPECIES}_best.keras"),
                   monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
]

# ============================================
# FASE 1: BASE CONGELADA
# ============================================
print("\n" + "="*60)
print(f"⏱️  FASE 1: BASE CONGELADA ({INITIAL_EPOCHS} épocas)")
print("="*60)
history1 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# FASE 2: FINE-TUNING (descongelar últimas 60 camadas)
# ============================================
base.trainable = True
for layer in base.layers[:-60]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

print("\n" + "="*60)
print(f"⏱️  FASE 2: FINE-TUNING ({FINE_TUNE_EPOCHS} épocas)")
print("="*60)
history2 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# GRÁFICOS
# ============================================
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(range(1, len(acc)+1), acc, label='Treino', marker='o', markersize=3)
axes[0].plot(range(1, len(acc)+1), val_acc, label='Validação', marker='s', markersize=3)
axes[0].axvline(x=INITIAL_EPOCHS+0.5, color='red', linestyle='--', alpha=0.5)
axes[0].set_title('🐕 CÃES (CSU 5 classes) - Acurácia', fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Acurácia')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(loss)+1), loss, label='Treino', marker='o', markersize=3)
axes[1].plot(range(1, len(loss)+1), val_loss, label='Validação', marker='s', markersize=3)
axes[1].axvline(x=INITIAL_EPOCHS+0.5, color='red', linestyle='--', alpha=0.5)
axes[1].set_title('🐕 CÃES (CSU 5 classes) - Loss', fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_MODELS_DIR, "dogs_csu_training.png"), dpi=150)
plt.close()  # Fecha a figura para liberar memória

# ============================================
# SALVAR MODELO E METADADOS
# ============================================
model_path = os.path.join(OUTPUT_MODELS_DIR, f"petzy_pain_{SPECIES}.keras")
model.save(model_path)

info = {
    'input_size': IMG_SIZE[0],
    'preprocessing': 'efficientnet_v2_preprocess',
    'species': SPECIES,
    'num_classes': NUM_CLASSES,
    'class_names': ['0', '1', '2', '3', '4'],
    'class_weights': {str(k): float(v) for k, v in class_weight_dict.items()}
}
with open(os.path.join(OUTPUT_MODELS_DIR, f"model_info_{SPECIES}.pkl"), 'wb') as f:
    pickle.dump(info, f)

print("\n" + "="*60)
print("✅ TREINAMENTO DE CÃES (5 classes) CONCLUÍDO!")
print("="*60)
print(f"Melhor acurácia de validação: {max(val_acc):.4f}")
print(f"Modelo salvo em: {model_path}")