from flask import Flask, request, jsonify, send_file
import whisper
import os
import numpy as np
import soundfile as sf
import resampy
import requests
import tensorflow as tf
import tensorflow_hub as hub
from transformers import pipeline

# Load NLP model globally
nlp_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    device=0
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

loaded_models = {}
yamnet_model = None
yamnet_class_names = None

YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

THREAT_CLASS_WEIGHTS = {
    "Gunshot, gunfire": 40,
    "Explosion": 40,
    "Screaming": 30,
    "Crying, sobbing": 15,
    "Breaking": 15,
    "Glass": 10,
    "Shatter": 15,
    "Boom": 25,
    "Thump, thud": 10,
    "Artillery fire": 40,
    "Machine gun": 40,
    "Burst, pop": 15,
    "Fire alarm": 20,
    "Alarm": 15,
    "Emergency vehicle": 10,
    "Siren": 15,
    "Fighting": 25,
    "Crowd": 5,
}

def get_whisper_model(model_name: str):
    if model_name not in loaded_models:
        loaded_models[model_name] = whisper.load_model(model_name, device="cuda")
    return loaded_models[model_name]

def get_yamnet():
    global yamnet_model, yamnet_class_names
    if yamnet_model is None:
        yamnet_model = hub.load(YAMNET_MODEL_URL)

        class_map_path = yamnet_model.class_map_path().numpy().decode()
        with tf.io.gfile.GFile(class_map_path) as f:
            lines = f.read().splitlines()

        yamnet_class_names = [line.split(",")[2] for line in lines[1:]]

    return yamnet_model, yamnet_class_names

import librosa
import numpy as np

def load_wav_16k_mono(path: str) -> np.ndarray:
    wav, sr = librosa.load(path, sr=16000, mono=True)

    if len(wav) == 0:
        raise ValueError("Audio is empty")

    # Normalize audio
    wav = wav / (np.max(np.abs(wav)) + 1e-6)

    return wav.astype(np.float32)

@app.route("/")
def home():
    return send_file("index.html")

def get_text_threat_score(text):
    try:
        result = nlp_model(text)[0]
        label = result["label"]
        confidence = result["score"]

        if "toxic" in label.lower() or "threat" in label.lower():
            return int(confidence * 100)
        else:
            return int(confidence * 30)
    except:
        return 0

def contextual_fusion(text_score, yamnet_score, matched_events):
    boost = 0
    labels = [e["label"] for e in matched_events]

    if "Screaming" in labels and "Crowd" in labels:
        boost += 15
    if "Gunshot, gunfire" in labels and "Screaming" in labels:
        boost += 25
    if "Explosion" in labels:
        boost += 30
    if text_score > 70:
        boost += 20

    if yamnet_score > 70:
        final = 0.8 * yamnet_score + 0.2 * text_score
    elif text_score > 70:
        final = 0.8 * text_score + 0.2 * yamnet_score
    else:
        final = 0.5 * text_score + 0.5 * yamnet_score

    return min(int(final + boost), 100)

import requests

def build_context(text, matched_events, keyword_score, yamnet_score):
    events = sorted(matched_events, key=lambda x: x["score"], reverse=True)[:5]

    event_lines = []
    for i, e in enumerate(events):
        event_lines.append(f"{i+1}. {e['label']} (confidence {e['score']:.2f})")

    event_text = "\n".join(event_lines)

    return f"""
AUDIO TIMELINE:
{event_text}

Speech transcript:
"{text}"

Scores:
- NLP threat score: {keyword_score}/100
- Audio threat score: {yamnet_score}/100
"""


def generate_audio_explanation_llm(text, matched_events, yamnet_score, keyword_score):
    context = build_context(text, matched_events, keyword_score, yamnet_score)
    confidence_hint = ""

    if yamnet_score > 70:
     confidence_hint = "The audio strongly indicates a high-intensity situation."
    elif yamnet_score > 40:
     confidence_hint = "The audio suggests moderate activity."
    else:
     confidence_hint = "The audio signals are weak or unclear."
    prompt = f"""
You are analyzing an audio clip for research and interpretation purposes.

{context}

Your task:
- Describe what sounds are present
- Explain what might be happening in a neutral and observational way
- Do NOT give instructions or advice
- Do NOT assume certainty
- Focus only on describing the scene

Output format:

1. Situation Summary:
(2–4 sentences describing the audio)

2. Key Observations:
- bullet points of detected elements

3. Overall Assessment:
Brief neutral conclusion about intensity (low / moderate / high activity)
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "temperature": 0.8,
                "stream": False
            },
            timeout=40
        )

        return response.json().get("response", "No explanation generated.")

    except Exception as e:
        return "Explanation unavailable (LLM error)"

def generate_audio_explanation(text, matched_events, yamnet_score):
    labels = [e["label"] for e in matched_events]
    text_lower = text.lower()

    # 🎯 Uncertainty-aware explanation (IMPORTANT)
    if "Gunshot, gunfire" in labels and "Screaming" in labels:
        return "Audio contains sounds consistent with gunfire and people in distress."

    if "Gunshot, gunfire" in labels or "Machine gun" in labels:
        return "Possible gunfire detected in the audio."

    if "Explosion" in labels:
        return "Audio contains sounds similar to an explosion."

    if "Screaming" in labels:
        return "Audio suggests people may be in distress."

    # 🗣️ Text context (soft interpretation)
    if any(word in text_lower for word in ["kill", "shoot", "attack"]):
        return "Speech contains references to violence, but context is unclear."

    return "No clear threatening audio patterns detected."

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    model_name = request.form.get("model", "base")
    audio_file = request.files["audio"]
    temp_path = audio_file.filename
    audio_file.save(temp_path)

    try:
        whisper_model = get_whisper_model(model_name)
        result = whisper_model.transcribe(temp_path)
        text = result.get("text", "").strip()

        if not text or len(text.strip()) < 5:
         text = "No meaningful speech detected in the audio"
        language = result.get("language", "unknown")

        keyword_score = get_text_threat_score(text)

        waveform = load_wav_16k_mono(temp_path)
        yamnet, class_names = get_yamnet()

        chunk_size = 16000 * 5

        event_counts = {}
        event_scores = {}
        all_scores = []

        for i in range(0, len(waveform), chunk_size):
            chunk = waveform[i:i + chunk_size]

            scores_tf, _, _ = yamnet(chunk)
            scores_np = scores_tf.numpy()

            all_scores.append(scores_np)

            for frame in scores_np:
                for idx, prob in enumerate(frame):
                    label = class_names[idx]

                    IGNORE_LABELS = ["Glass", "Emergency vehicle"]

                    if label in THREAT_CLASS_WEIGHTS and prob > 0.15 and label not in IGNORE_LABELS:
                        event_counts[label] = event_counts.get(label, 0) + 1
                        event_scores[label] = max(event_scores.get(label, 0), prob)

        all_scores_np = np.vstack(all_scores)
        mean_scores = all_scores_np.mean(axis=0)

        top_indices = mean_scores.argsort()[-10:][::-1]
        top_events = [
            {"label": class_names[i], "score": float(round(float(mean_scores[i]), 4))}
            for i in top_indices
        ]

        yamnet_score = 0
        matched_threats = []

        for label in event_counts:
            weight = THREAT_CLASS_WEIGHTS[label]
            frequency = event_counts[label]
            confidence = event_scores[label]

            contribution = weight * (confidence ** 2) * (1 + frequency / 5)
            yamnet_score += contribution

            matched_threats.append({
                "label": label,
                "score": float(round(confidence, 4)),
                "frequency": int(frequency),
                "contribution": float(round(contribution, 2)),
            })

        CRITICAL_EVENTS = ["Gunshot, gunfire", "Explosion", "Machine gun"]

        for label in event_counts:
            if label in CRITICAL_EVENTS:
                yamnet_score += 30

        yamnet_score = min(yamnet_score, 100)

        combined_score = contextual_fusion(
            keyword_score,
            yamnet_score,
            matched_threats
        )

        def score_to_status(s):
            if s == 0: return "No Threat"
            if s < 40: return "Low Threat"
            if s < 70: return "Moderate Threat"
            return "High Threat"

        response = {
    "text": text,
    "language": language,
    "threat_level": combined_score,
    "threat_status": score_to_status(combined_score),
    "keyword_threat_score": keyword_score,
    "yamnet_threat_score": float(round(yamnet_score, 2)),
    "matched_threat_events": matched_threats,
    "top_audio_events": top_events,

    
    "audio_explanation": generate_audio_explanation_llm(
    text,
    matched_threats,
    yamnet_score,
    keyword_score
),
}
    except Exception as e:
        response = {"error": str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)