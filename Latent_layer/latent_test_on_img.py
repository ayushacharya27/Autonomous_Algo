import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import json
import google.generativeai as genai



# Autoencoder class

class ConvAutoencoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ConvAutoencoder, self).__init__(**kwargs)

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(128, 128, 1)),
            layers.Conv2D(16, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(128, (3,3), strides=2, padding='same', activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(8, 8, 128)),
            layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(16, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(1, (3,3), padding='same', activation='sigmoid'),
        ])

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)



# load model + encoder
model = load_model(
    "/home/ayush/Autonomous_Algo/Training_codes/best_trained.keras",
    custom_objects={"ConvAutoencoder": ConvAutoencoder}
)

encoder = model.encoder



# Saved Signatures
'''signatures = {
    "flood": np.load("flood_signature.npy"),
    # add more later
}'''

signatures = {
    "dirty_road": np.load("/home/ayush/Autonomous_Algo/Latent_layer/Signatures/dirtyroad_signature.npy"),
    "flood": np.load("/home/ayush/Autonomous_Algo/Latent_layer/Signatures/flood_signature.npy"),
    "clean_road": np.load("/home/ayush/Autonomous_Algo/Latent_layer/Signatures/cleanroad_signature.npy"),
    "pedestrian_road": np.load("/home/ayush/Autonomous_Algo/Latent_layer/Signatures/pedestrianroad_signature.npy"),
    "pothole_road": np.load("/home/ayush/Autonomous_Algo/Latent_layer/Signatures/pothole_signature.npy"),

    
    # add more later
}



# Cosine Similiarity
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)



# Extract latent from one image
def get_latent(img_path):
    img = cv2.imread(img_path, 0)          # grayscale
    img = cv2.resize(img, (256,256))       # must match model size
    img = img.astype("float32") / 255.0
    img = img[..., np.newaxis]             # (128,128,1)
    img = np.expand_dims(img, axis=0)      # (1,128,128,1)

    latent = encoder(img)                  # (1,8,8,128)
    latent = tf.reshape(latent, (latent.shape[0], -1))
    return latent.numpy()[0]



# image_path

img_path = "/home/ayush/Autonomous_Algo/flooded_road_test3.jpg"

latent = get_latent(img_path)

best_class = None
best_score = -1

for cname, sig in signatures.items():
    score = cosine_similarity(latent, sig)
    print(f"{cname} -> {score:.4f}")

    if score > best_score:
        best_score = score
        best_class = cname

print("\nPredicted:", best_class)
print("Score:", best_score)



# LLM Handling
scene_report = {
    "vehicle_id": "AV_01",
    "scene_analysis": {
        "predicted_event": best_class,
        "confidence": float(best_score),
        "anomaly_score": float(1 - best_score)
    },
    "vehicle_state": {
        "current_speed_kmph": 45, # Hard Coded
        "Steering_Curvature": +90
    }
}

# System Prompt
def build_prompt(scene_report):
    return f"""
You are an autonomous vehicle control planner.

Your task:
Based on the scene data, Return at least 2 coordinated control actions whenever risk is moderate or high.
Prioritize layered safety strategies.


STRICT RULES:
- Return ONLY valid JSON.
- Do NOT include explanations.
- Do NOT include markdown.
- Do NOT include comments.
- Do NOT include any text outside JSON.
- Only use actions from the allowed list.
- If no action is required, return an empty list.

Allowed actions:
- reduce_speed
- emergency_brake
- increase_following_distance
- adjust_steering
- maintain_speed

Required JSON format:

{{
  "recommended_actions": [
    {{
      "action": "<one_of_allowed_actions>",
      "parameters": {{
        "target_speed_kmph": <number_optional>,
        "steering_adjustment_deg": <number_optional>,
        "intensity": "<low|medium|high_optional>"
      }}
    }}
  ]
}}

Scene Data:
{json.dumps(scene_report, indent=2)}
"""


GEMINI_API_KEY="Add Your Key"
genai.configure(api_key=GEMINI_API_KEY)


def get_model():
    return genai.GenerativeModel("gemini-2.5-flash")


prompt = build_prompt(scene_report)
model = get_model()
response = model.generate_content(prompt)

def parse_llm_response(response_text):
    try:
        cleaned = response_text.strip()

        # Remove markdown fences safely
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "")
            cleaned = cleaned.replace("```", "")
            cleaned = cleaned.strip()

        data = json.loads(cleaned)

        # Required top-level field
        if "recommended_actions" not in data:
            raise ValueError("Missing recommended_actions")

        # Allowed actions
        allowed_actions = {
            "reduce_speed",
            "emergency_brake",
            "increase_following_distance",
            "adjust_steering",
            "activate_traction_control",
            "maintain_speed"
        }

        # Validate each action
        validated_actions = []
        for action_obj in data["recommended_actions"]:

            if "action" not in action_obj:
                continue

            if action_obj["action"] not in allowed_actions:
                continue

            # Ensure parameters exists
            if "parameters" not in action_obj:
                action_obj["parameters"] = {}

            validated_actions.append(action_obj)

        return {
            "recommended_actions": validated_actions
        }

    except Exception as e:
        print("LLM parsing failed:", e)

        # Safe fallback behavior
        return {
            "recommended_actions": [
                {
                    "action": "reduce_speed",
                    "parameters": {
                        "target_speed_kmph": 15,
                        "intensity": "high"
                    }
                }
            ]
        }


analysis_json = parse_llm_response(response.text)
print(analysis_json)

#print(response.text)





