
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Vortrainiertes ViT-B/16-Modell laden
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)

# Vortrainierte Parameter des Backbones einfrieren
for param in model.parameters():
    param.requires_grad = False

# Klassifikationskopf für die 16 Dokumentklassen ersetzen
in_features = model.heads.head.in_features
model.heads.head = nn.Linear(in_features, 16)

# Modell für den Test in den Evaluierungsmodus setzen
model.eval()

# Gesamtzahl und Anzahl der trainierbaren Parameter bestimmen
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)

# Vorwärtsdurchlauf mit einem künstlichen Bild der Größe 224 × 224 testen
dummy_input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)
