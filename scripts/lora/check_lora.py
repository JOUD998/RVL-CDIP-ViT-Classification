
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from lora_attention import LoRAMultiheadAttention

# Verfügbares Rechengerät auswählen (Apple MPS oder CPU)
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

# Vortrainiertes ViT-B/16-Modell laden
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

# Ursprüngliche Modellparameter einfrieren
for param in model.parameters():
    param.requires_grad = False

# Self-Attention in allen 12 Encoder-Blöcken durch LoRA erweitern
for layer in model.encoder.layers:
    layer.self_attention = LoRAMultiheadAttention(
        layer.self_attention,
        rank=4,
        alpha=8
    )

# Klassifikationskopf für die 16 Dokumentklassen ersetzen
model.heads.head = nn.Linear(
    model.heads.head.in_features, 16
)

model = model.to(device)
model.eval()

# Anzahl der trainierbaren Parameter überprüfen
trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", trainable_params)

# Künstliche Testdaten für einen vollständigen Trainingsschritt erzeugen
images = torch.randn(1, 3, 224, 224, device=device)
labels = torch.tensor([3], device=device)

criterion = nn.CrossEntropyLoss()

# Vorwärtsdurchlauf und Verlustberechnung testen
outputs = model(images)
loss = criterion(outputs, labels)

print("Output shape:", outputs.shape)
print("Loss:", loss.item())

# Rückwärtsdurchlauf zur Überprüfung der Gradientenberechnung
loss.backward()

print(
    "LoRA Q gradient:",
    model.encoder.layers[0].self_attention.lora_q_B.grad is not None
)

print(
    "LoRA V gradient:",
    model.encoder.layers[0].self_attention.lora_v_B.grad is not None
)

print(
    "Classification head gradient:",
    model.heads.head.weight.grad is not None
)

print("Forward and backward completed!")


# Ursprüngliche Attention-Ausgabe mit der LoRA-Variante vergleichen
# Bei einer initialen LoRA-Aktualisierung von null sollten beide übereinstimmen
attention = model.encoder.layers[0].self_attention

attention.eval()

x = torch.randn(
    1, 197, 768,
    device=device
)

with torch.no_grad():
    original_output, _ = attention.attention(
        x, x, x,
        need_weights=False
    )

    lora_output, _ = attention(
        x, x, x,
        need_weights=False
    )

    # Größte absolute Abweichung zwischen beiden Ausgaben bestimmen
    max_difference = (
        original_output - lora_output
    ).abs().max().item()

print("Max attention difference:", max_difference)
