import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
import numpy as np
import os

class WorkoutClassifier:
    def __init__(self):
        self.model_name = "prithivMLmods/Gym-Workout-Classifier-SigLIP2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        # Mapping model labels to our internal analyzer names
        self.label_mapping = {
            "squat": "Squat",
            "deadlift": "Deadlift",
            "push up": "Push-Up",
            "bench press": "Bench Press",
            "plank": "Plank",
            "tricep dips": "Dips",
            "barbell biceps curl": "Biceps Curl",
            "chest fly machine": "Chest Fly",
            "lat pulldown": "Lat Pulldown"
        }
        
        self.fallback_labels = {
            0: "barbell biceps curl", 1: "bench press", 2: "chest fly machine", 3: "deadlift",
            4: "decline bench press", 5: "hammer curl", 6: "hip thrust", 7: "incline bench press",
            8: "lat pulldown", 9: "lateral raises", 10: "leg extension", 11: "leg raises",
            12: "plank", 13: "pull up", 14: "push up", 15: "romanian deadlift",
            16: "russian twist", 17: "shoulder press", 18: "squat", 19: "t bar row",
            20: "tricep dips", 21: "tricep pushdown"
        }

    def load_model(self):
        """Lazy load the model to save memory during startup."""
        if self.model is None:
            print(f"Loading Workout Classifier: {self.model_name}...")
            self.model = SiglipForImageClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.labels = self._get_labels()

    def _get_labels(self):
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) > 0:
            out = {}
            for k, v in id2label.items():
                try: out[int(k)] = v
                except: pass
            if len(out) > 0: return out
        return self.fallback_labels

    def classify(self, image_np, topk=3):
        """
        Classifies a single frame.
        Input: numpy array (RGB)
        Output: List of (label, probability)
        """
        self.load_model()
        
        image = Image.fromarray(image_np).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

        vals, idxs = torch.topk(probs, k=topk)
        
        results = []
        for v, i in zip(vals.tolist(), idxs.tolist()):
            raw_label = self.labels.get(i, f"class_{i}")
            # Map to internal names if possible
            mapped_label = self.label_mapping.get(raw_label.lower(), raw_label.title())
            results.append((mapped_label, float(v)))
            
        return results

# Singleton instance for the app
_classifier = None
def get_workout_classifier():
    global _classifier
    if _classifier is None:
        _classifier = WorkoutClassifier()
    return _classifier
