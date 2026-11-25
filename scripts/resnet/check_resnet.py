
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from scripts.data_pipeline.prepare_cnn_dataset import train_loader

# Vortrainiertes ResNet18-Modell laden
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Vortrainierte Parameter einfrieren
for param in model.parameters():
    param.requires_grad = False

# Klassifikationsschicht für die 16 Dokumentklassen ersetzen
model.fc = nn.Linear(model.fc.in_features, 16)

# Modell für den Test in den Evaluierungsmodus setzen
model.eval()

# Einen Batch mit echten Dokumentbildern laden
images, labels = next(iter(train_loader))

# Vorwärtsdurchlauf ohne Gradientenberechnung durchführen
with torch.no_grad():
    outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)

predictions = outputs.argmax(dim=1)

print("True labels:     ", labels)
print("Predicted labels:", predictions)

# Kreuzentropie als Verlustfunktion verwenden
criterion = nn.CrossEntropyLoss()

# Adam-Optimierer ausschließlich für die neue Klassifikationsschicht
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

# Verlust für den Test-Batch berechnen
loss = criterion(outputs, labels)

print("Loss:", loss.item())

# Überprüfen, welche Modellparameter trainierbar sind
print("Backbone trainable:", model.layer1[0].conv1.weight.requires_grad)
print("FC trainable:", model.fc.weight.requires_grad)
