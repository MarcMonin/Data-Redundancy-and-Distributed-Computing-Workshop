import numpy as np
from collections import defaultdict

class ConsensusModel:
    def __init__(self, model_names, classes):
        self.models = {name: [] for name in model_names}  # Stocke les prédictions de classes
        self.weights = {name: 1.0 for name in model_names}  # Poids initiaux (1.0)
        self.alpha = 0.1  # Facteur d'ajustement du poids
        self.classes = classes  # Liste des classes possibles

    def update_predictions(self, predictions):
        """Ajoute une nouvelle série de prédictions."""
        for model_name, pred in predictions.items():
            self.models[model_name].append(pred)
    
    def compute_consensus(self):
        """Calcule la classe majoritaire pondérée comme consensus."""
        class_votes = {cls: 0 for cls in self.classes}
        for model_name in self.models:
            for pred in self.models[model_name]:
                class_votes[pred] += self.weights[model_name]
        consensus = max(class_votes, key=class_votes.get)
        return consensus

    def update_weights(self):
        """Ajuste les poids des modèles en fonction de leur précision relative."""
        consensus = self.compute_consensus()
        for model_name in self.models:
            correct_predictions = sum(1 for pred in self.models[model_name] if pred == consensus)
            total_predictions = len(self.models[model_name])
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            self.weights[model_name] = max(0, self.weights[model_name] + self.alpha * (accuracy - 0.5))
        
    def get_weights(self):
        return self.weights

# Exemple d'utilisation
model_names = ["model_A", "model_B", "model_C"]
classes = ["class_0", "class_1", "class_2"]
consensus_model = ConsensusModel(model_names, classes)

# Simuler des prédictions de classification
predictions_batch = {
    "model_A": ["class_0", "class_1", "class_2"],
    "model_B": ["class_0", "class_0", "class_1"],
    "model_C": ["class_1", "class_1", "class_2"]
}

consensus_model.update_predictions(predictions_batch)
print("Consensus initial:", consensus_model.compute_consensus())
consensus_model.update_weights()
print("Nouveaux poids après ajustement:", consensus_model.get_weights())
