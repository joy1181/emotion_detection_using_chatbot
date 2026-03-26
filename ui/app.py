from flask import Flask, Response, jsonify, render_template, request
import atexit
import json
import os
import threading
import time
from datetime import datetime

import cv2
import dlib
import numpy as np
from dotenv import load_dotenv
from imutils import face_utils
from openai import OpenAI
from PIL import Image
from scipy.spatial import distance
from transformers import AutoImageProcessor, AutoModelForImageClassification


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
negative_emotions = {"Angry", "Fear", "Sad", "Disgust"}

face_detector = None
predictor = None
emotion_model = None
emotion_model_type = None
hf_emotion_processor = None
hf_emotion_id_to_label = {}
camera = None
active_camera_backend = None
is_streaming = False
current_emotions = {"timestamp": None, "faces_count": 0, "faces": []}
face_pipeline_ready = False
face_pipeline_last_error = "Face pipeline has not been validated yet."

conversation_history = []
emotion_history = []
pending_messages = []
negative_state = {"emotion": None, "started_at": None, "last_triggered_at": None}

state_lock = threading.Lock()
monitor_stop_event = threading.Event()
monitor_thread = None

RUNTIME_DIR = os.path.join(BASE_DIR, "runtime_data")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
LOG_FILE = os.path.join(RUNTIME_DIR, "activity_log.jsonl")
STATE_FILE = os.path.join(RUNTIME_DIR, "session_state.json")

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY_ENV_KEYS = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY")
HF_EMOTION_MODEL = os.getenv("HF_EMOTION_MODEL", "dima806/facial_emotions_image_detection")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_BACKEND = os.getenv("CAMERA_BACKEND", "auto").strip().lower()
MAX_CONTEXT_TOKENS = 10000
MAX_RESPONSE_TOKENS = 512
NEGATIVE_TRIGGER_SECONDS = 5
PROACTIVE_COOLDOWN_SECONDS = 120
MAX_EMOTION_HISTORY = 200


def set_face_pipeline_status(ready, error=None):
    global face_pipeline_ready, face_pipeline_last_error
    face_pipeline_ready = ready
    face_pipeline_last_error = error


def reset_current_emotions(timestamp=None):
    global current_emotions
    with state_lock:
        current_emotions = {
            "timestamp": timestamp or now_iso(),
            "faces_count": 0,
            "faces": [],
        }


def validate_face_pipeline():
    if face_detector is None or predictor is None:
        set_face_pipeline_status(False, "Face detector or landmark predictor is unavailable.")
        return False
    if emotion_model is None:
        set_face_pipeline_status(False, "Emotion classification model is unavailable.")
        return False

    gray_probe = np.zeros((32, 32), dtype=np.uint8)
    rgb_probe = np.zeros((32, 32, 3), dtype=np.uint8)
    probe_rect = dlib.rectangle(0, 0, 31, 31)

    try:
        face_detector(gray_probe, 0)
        face_detector(rgb_probe, 0)
        predictor(gray_probe, probe_rect)
    except Exception as exc:
        set_face_pipeline_status(False, f"Face pipeline self-check failed: {exc}")
        return False

    set_face_pipeline_status(True, None)
    return True


def ensure_runtime_dirs():
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    os.makedirs(CAPTURES_DIR, exist_ok=True)


def release_camera():
    global camera, active_camera_backend
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
    camera = None
    active_camera_backend = None


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def append_jsonl(path, payload):
    ensure_runtime_dirs()
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_session_state():
    ensure_runtime_dirs()
    with state_lock:
        payload = {
            "conversation_history": conversation_history,
            "emotion_history": emotion_history,
            "current_emotions": current_emotions,
            "updated_at": now_iso(),
        }
    with open(STATE_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def log_event(event_type, payload):
    append_jsonl(
        LOG_FILE,
        {
            "timestamp": now_iso(),
            "event_type": event_type,
            "payload": payload,
        },
    )


def get_deepseek_api_key():
    for env_key in DEEPSEEK_API_KEY_ENV_KEYS:
        value = os.getenv(env_key)
        if value:
            return value
    return None


def deepseek_available():
    return bool(get_deepseek_api_key())


def get_deepseek_client():
    api_key = get_deepseek_api_key()
    if not api_key:
        raise RuntimeError("DeepSeek API key is missing. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def estimate_tokens(text):
    if not text:
        return 0
    return max(1, len(text) // 4)


def get_primary_emotion_snapshot():
    with state_lock:
        faces = current_emotions.get("faces", [])
        timestamp = current_emotions.get("timestamp")
        if not faces:
            return {
                "label": "Neutral",
                "confidence": 0.0,
                "timestamp": timestamp,
                "faces_count": 0,
            }
        primary = faces[0]
        return {
            "label": primary.get("predicted_emotion", "Neutral"),
            "confidence": float(primary.get("confidence", 0.0)),
            "timestamp": timestamp,
            "faces_count": current_emotions.get("faces_count", 0),
        }


def trim_conversation_history(messages):
    budget = MAX_CONTEXT_TOKENS - MAX_RESPONSE_TOKENS
    total = 0
    trimmed = []
    for message in reversed(messages):
        content = message.get("content", "")
        message_tokens = estimate_tokens(content) + 12
        if total + message_tokens > budget:
            break
        trimmed.append({"role": message["role"], "content": content})
        total += message_tokens
    trimmed.reverse()
    return trimmed


def enqueue_pending_message(message):
    with state_lock:
        pending_messages.append(message)


def append_conversation_message(role, content, metadata=None):
    message = {
        "role": role,
        "content": content,
        "timestamp": now_iso(),
    }
    if metadata:
        message["metadata"] = metadata
    with state_lock:
        conversation_history.append(message)
    log_event("conversation", message)
    save_session_state()
    return message


def track_emotion_transition(snapshot):
    label = snapshot["label"]
    confidence = float(snapshot["confidence"])
    timestamp = snapshot["timestamp"] or now_iso()

    should_append = False
    with state_lock:
        last = emotion_history[-1] if emotion_history else None
        if last is None:
            should_append = True
        elif last["label"] != label:
            should_append = True
        elif abs(float(last["confidence"]) - confidence) >= 0.15:
            should_append = True

        if should_append:
            emotion_history.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "timestamp": timestamp,
                }
            )
            if len(emotion_history) > MAX_EMOTION_HISTORY:
                del emotion_history[:-MAX_EMOTION_HISTORY]

    if should_append:
        log_event("emotion_transition", snapshot)
        save_session_state()


def build_system_prompt():
    snapshot = get_primary_emotion_snapshot()
    with state_lock:
        recent_emotions = emotion_history[-20:]

    system_state = {
        "current_emotion": {
            "label": snapshot["label"],
            "confidence": round(snapshot["confidence"], 4),
            "timestamp": snapshot["timestamp"],
            "faces_count": snapshot["faces_count"],
        },
        "emotion_history": recent_emotions,
        "time": now_iso(),
    }

    return (
        "You are an empathetic AI assistant in an emotion-aware web application. "
        "You should respond naturally, supportively, and conversationally. "
        "Use the current facial emotion context when it is useful, but do not sound robotic or repeatedly mention system internals. "
        "If the user appears distressed, respond with calm, supportive language and encourage small practical next steps. "
        "Keep replies concise unless the user asks for more detail. "
        f"System state: {json.dumps(system_state, ensure_ascii=False)}"
    )


def request_deepseek_response(trigger, latest_user_message=None):
    client = get_deepseek_client()
    with state_lock:
        trimmed_history = trim_conversation_history(conversation_history)

    messages = [{"role": "system", "content": build_system_prompt()}]
    messages.extend(trimmed_history)

    if trigger == "proactive" and not latest_user_message:
        messages.append(
            {
                "role": "user",
                "content": (
                    "The user has been showing a sustained negative facial emotion for at least 5 seconds. "
                    "Start the conversation gently, acknowledge that they may not want to talk, and invite them to share if they want."
                ),
            }
        )

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        max_tokens=MAX_RESPONSE_TOKENS,
        temperature=0.7,
    )
    content = response.choices[0].message.content if response.choices else ""
    return (content or "").strip()


def generate_chat_response(trigger, latest_user_message=None):
    try:
        reply = request_deepseek_response(trigger, latest_user_message)
        if not reply:
            raise RuntimeError("DeepSeek returned an empty response.")
        return reply, None
    except Exception as exc:
        log_event("deepseek_error", {"trigger": trigger, "error": str(exc)})
        return None, "对话服务暂时不可用，请稍后再试。"


def eye_aspect_ratio(eye):
    point_a = distance.euclidean(eye[1], eye[5])
    point_b = distance.euclidean(eye[2], eye[4])
    point_c = distance.euclidean(eye[0], eye[3])
    return (point_a + point_b) / (2.0 * point_c)


def normalize_emotion_label(label):
    mapping = {
        "angry": "Angry",
        "disgust": "Disgust",
        "fear": "Fear",
        "happy": "Happy",
        "neutral": "Neutral",
        "sad": "Sad",
        "surprise": "Surprise",
    }
    return mapping.get((label or "").strip().lower())


def get_camera_backend_candidates():
    backend_map = {
        "auto": [(None, "default"), (cv2.CAP_DSHOW, "dshow"), (cv2.CAP_MSMF, "msmf")],
        "default": [(None, "default")],
        "dshow": [(cv2.CAP_DSHOW, "dshow")],
        "msmf": [(cv2.CAP_MSMF, "msmf")],
    }
    return backend_map.get(CAMERA_BACKEND, [(None, "default")])


def open_camera_device():
    global active_camera_backend
    for backend, backend_name in get_camera_backend_candidates():
        capture = cv2.VideoCapture(CAMERA_INDEX) if backend is None else cv2.VideoCapture(CAMERA_INDEX, backend)
        if capture is not None and capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(5):
                success, _ = capture.read()
                if success:
                    active_camera_backend = backend_name
                    return capture
                time.sleep(0.1)
        if capture is not None:
            capture.release()
    active_camera_backend = None
    return None


def setup_face_detection():
    global face_detector, predictor
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = os.path.join(BASE_DIR, "Models", "shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(predictor_path):
        print(f"ERROR: {predictor_path} not found")
        return False
    predictor = dlib.shape_predictor(predictor_path)
    return True


def load_hf_emotion_model():
    global emotion_model, emotion_model_type, hf_emotion_processor, hf_emotion_id_to_label
    try:
        hf_emotion_processor = AutoImageProcessor.from_pretrained(HF_EMOTION_MODEL)
        emotion_model = AutoModelForImageClassification.from_pretrained(HF_EMOTION_MODEL)
        emotion_model_type = "huggingface"
        hf_emotion_id_to_label = {
            int(index): label for index, label in emotion_model.config.id2label.items()
        }
        print(f"Loaded Hugging Face emotion model: {HF_EMOTION_MODEL}")
        return True
    except Exception as exc:
        print(f"Error loading Hugging Face emotion model {HF_EMOTION_MODEL}: {exc}")
        emotion_model = None
        emotion_model_type = None
        hf_emotion_processor = None
        hf_emotion_id_to_label = {}
        return False


def load_emotion_model():
    return load_hf_emotion_model()


def predict_emotion_probabilities(face_region):
    if emotion_model is None:
        raise RuntimeError("Emotion model is not initialized.")

    if emotion_model_type == "huggingface":
        rgb_face = cv2.cvtColor(face_region, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(rgb_face)
        inputs = hf_emotion_processor(images=image, return_tensors="pt")
        outputs = emotion_model(**inputs)
        logits = outputs.logits[0].detach().cpu().numpy()
        logits = logits - np.max(logits)
        scores = np.exp(logits)
        raw_probabilities = scores / np.sum(scores)

        mapped_probabilities = {emotion: 0.0 for emotion in emotion_labels}
        for index, probability in enumerate(raw_probabilities):
            normalized_label = normalize_emotion_label(hf_emotion_id_to_label.get(index, ""))
            if normalized_label:
                mapped_probabilities[normalized_label] = float(probability)

        total = sum(mapped_probabilities.values())
        if total > 0:
            mapped_probabilities = {
                emotion: probability / total for emotion, probability in mapped_probabilities.items()
            }
        return mapped_probabilities

    raise RuntimeError(f"Unsupported emotion model type: {emotion_model_type}")


"""
将原始BGR图像转换为灰度图和RGB图,并确保它们是连续的uint8数组,
# 以兼容dlib和Hugging Face模型的输入要求。
# 检测人脸并提取面部区域进行情绪预测,同时计算EAR以评估眨眼程度。
# 将每个检测到的人脸的情绪数据存储在全局状态中，并在视频帧上绘制边界框和情绪标签。
真实摄像头帧的数据：
原始 frame 形状是 (480, 640, 3)
dtype 是 uint8
gray 是 uint8 且连续
rgb_frame 是 uint8 且连续
"""
def process_frame(frame):
    global current_emotions

    if not face_pipeline_ready:
        reset_current_emotions()
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #转换为灰度图
    if gray.dtype != np.uint8:  #确保是uint8类型
        gray = cv2.convertScaleAbs(gray) 
    gray = np.ascontiguousarray(gray, dtype=np.uint8) #优化内存访问
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if rgb_frame.dtype != np.uint8:
        rgb_frame = cv2.convertScaleAbs(rgb_frame)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    try:
        faces = face_detector(rgb_frame, 1)  #出错点
    except Exception as exc:
        message = f"Face detection failed for current frame: {exc}"
        print(message)
        set_face_pipeline_status(False, message)
        reset_current_emotions()
        return frame

    emotions_data = []

    for index, face in enumerate(faces):
        try:
            landmarks = predictor(rgb_frame, face)
            landmarks = face_utils.shape_to_np(landmarks)
            x, y, width, height = face_utils.rect_to_bb(face)
            face_region = gray[y:y + height, x:x + width]
            if face_region.size == 0:
                continue

            all_emotions = predict_emotion_probabilities(face_region)
            predicted_emotion = max(all_emotions, key=all_emotions.get)
            confidence = float(all_emotions[predicted_emotion])

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            face_data = {
                "face_id": index,
                "bbox": [int(x), int(y), int(width), int(height)],
                "predicted_emotion": predicted_emotion,
                "confidence": confidence,
                "ear": float(ear),
                "all_emotions": all_emotions,
                "landmarks": landmarks.tolist(),
            }
            emotions_data.append(face_data)

            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{predicted_emotion} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        except Exception as exc:
            print(f"Error processing face {index}: {exc}")
            set_face_pipeline_status(False, f"Face processing failed for face {index}: {exc}")

    snapshot = {
        "timestamp": now_iso(),
        "faces_count": len(emotions_data),
        "faces": emotions_data,
    }
    with state_lock:
        current_emotions = snapshot

    track_emotion_transition(get_primary_emotion_snapshot())
    return frame


def generate_frames():
    global camera, is_streaming 
    while is_streaming:
        if camera is None:
            break
        success, frame = camera.read()
        if not success:
            is_streaming = False
            release_camera()
            break
        if emotion_model is not None and face_detector is not None and predictor is not None:
            frame = process_frame(frame)
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(0.033)


def monitor_negative_emotions():
    while not monitor_stop_event.is_set():
        time.sleep(1)
        snapshot = get_primary_emotion_snapshot()
        current_time = time.time()

        with state_lock:
            active_stream = is_streaming

        if not active_stream or snapshot["label"] not in negative_emotions:
            with state_lock:
                negative_state["emotion"] = None
                negative_state["started_at"] = None
            continue

        with state_lock:
            if negative_state["emotion"] != snapshot["label"]:
                negative_state["emotion"] = snapshot["label"]
                negative_state["started_at"] = current_time
            started_at = negative_state["started_at"]
            last_triggered_at = negative_state["last_triggered_at"]

        if started_at is None or current_time - started_at < NEGATIVE_TRIGGER_SECONDS:
            continue
        if last_triggered_at and current_time - last_triggered_at < PROACTIVE_COOLDOWN_SECONDS:
            continue

        reply, error_message = generate_chat_response("proactive")
        content = reply or error_message
        message = append_conversation_message(
            "assistant",
            content,
            {
                "trigger": "proactive",
                "emotion": snapshot,
            },
        )
        enqueue_pending_message(message)
        with state_lock:
            negative_state["last_triggered_at"] = current_time


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start_camera")
def start_camera():
    global camera, is_streaming

    if camera is not None and not camera.isOpened():
        release_camera()

    if camera is None:
        camera = open_camera_device()
        if camera is None:
            is_streaming = False
            return jsonify(
                {
                    "error": "Could not open camera",
                    "camera_index": CAMERA_INDEX,
                    "camera_backend": CAMERA_BACKEND,
                }
            ), 500
    is_streaming = True
    payload = {
        "status": "Camera started",
        "camera_index": CAMERA_INDEX,
        "camera_backend": active_camera_backend,
        "face_pipeline_ready": face_pipeline_ready,
    }
    if not face_pipeline_ready and face_pipeline_last_error:
        payload["warning"] = face_pipeline_last_error
    return jsonify(payload)


@app.route("/stop_camera")
def stop_camera():
    global is_streaming
    is_streaming = False
    release_camera()
    return jsonify({"status": "Camera stopped"})


@app.route("/get_emotions")
def get_emotions():
    primary = get_primary_emotion_snapshot()
    with state_lock:
        payload = {
            **current_emotions,
            "primary_emotion": primary,
            "emotion_history": emotion_history[-20:],
            "face_pipeline_ready": face_pipeline_ready,
            "face_pipeline_last_error": face_pipeline_last_error,
        }
    return jsonify(payload)


@app.route("/system_status")
def system_status():
    status = {
        "face_detector": face_detector is not None,
        "predictor": predictor is not None,
        "emotion_model": emotion_model is not None,
        "emotion_model_type": emotion_model_type,
        "face_pipeline_ready": face_pipeline_ready,
        "face_pipeline_last_error": face_pipeline_last_error,
        "camera": camera is not None and camera.isOpened() if camera else False,
        "camera_index": CAMERA_INDEX,
        "camera_backend": active_camera_backend,
        "streaming": is_streaming,
        "deepseek_ready": deepseek_available(),
        "chatbot_ready": deepseek_available(),
        "emotion_history_count": len(emotion_history),
        "conversation_turns": len(conversation_history),
    }
    return jsonify(status)


@app.route("/capture_frame")
def capture_frame():
    global camera, is_streaming
    if camera is None or not camera.isOpened():
        is_streaming = False
        release_camera()
        return jsonify({"error": "Camera not available"}), 500

    success, frame = camera.read()
    if not success:
        is_streaming = False
        release_camera()
        return jsonify({"error": "Could not capture frame"}), 500
    if emotion_model is not None and face_detector is not None and predictor is not None:
        frame = process_frame(frame)

    ensure_runtime_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = os.path.join(CAPTURES_DIR, f"emotion_capture_{timestamp}.jpg")
    data_filename = os.path.join(CAPTURES_DIR, f"emotion_capture_{timestamp}.json")
    cv2.imwrite(image_filename, frame)
    with open(data_filename, "w", encoding="utf-8") as handle:
        json.dump(current_emotions, handle, ensure_ascii=False, indent=2)

    payload = {
        "status": "Frame captured",
        "image_file": image_filename,
        "data_file": data_filename,
        "emotions": current_emotions,
        "face_pipeline_ready": face_pipeline_ready,
    }
    if not face_pipeline_ready and face_pipeline_last_error:
        payload["warning"] = face_pipeline_last_error
    return jsonify(payload)


@app.route("/ask_chatbot", methods=["POST"])
def ask_chatbot():
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"answer": "请输入你想说的话。"}), 400

    user_message = append_conversation_message("user", question, {"trigger": "user"})
    reply, error_message = generate_chat_response("user", question)
    assistant_content = reply or error_message
    assistant_message = append_conversation_message(
        "assistant",
        assistant_content,
        {"trigger": "user_response", "reply_to": user_message["timestamp"]},
    )
    enqueue_pending_message(assistant_message)
    return jsonify(
        {
            "answer": assistant_content,
            "error": error_message is not None,
            "message": assistant_message,
        }
    )


@app.route("/chat_updates")
def chat_updates():
    with state_lock:
        updates = list(pending_messages)
        pending_messages.clear()
    return jsonify({"messages": updates})


@app.route("/conversation_history")
def conversation_history_endpoint():
    with state_lock:
        payload = list(conversation_history)
    return jsonify({"messages": payload})


@app.route("/emotion_history")
def emotion_history_endpoint():
    with state_lock:
        payload = list(emotion_history)
    return jsonify({"emotion_history": payload})


@app.route("/reset_conversation", methods=["POST"])
def reset_conversation():
    with state_lock:
        conversation_history.clear()
        pending_messages.clear()
    log_event("conversation_reset", {})
    save_session_state()
    return jsonify({"status": "Conversation history reset"})


def initialize_system():
    print("Initializing system...")
    ensure_runtime_dirs()

    if not setup_face_detection():
        print("Failed to setup face detection.")
        set_face_pipeline_status(False, "Face detector setup failed.")
    if not load_emotion_model():
        print("Failed to load emotion model.")
        set_face_pipeline_status(False, "Emotion model load failed.")
    if not validate_face_pipeline() and face_pipeline_last_error:
        print(face_pipeline_last_error)
    if not deepseek_available():
        print("DeepSeek API key not found. Chatbot will return friendly error messages until configured.")

    global monitor_thread
    if monitor_thread is None:
        monitor_thread = threading.Thread(target=monitor_negative_emotions, daemon=True)
        monitor_thread.start()

    save_session_state()
    print("System initialization complete!")
    return True


def cleanup():
    global is_streaming
    monitor_stop_event.set()
    is_streaming = False
    release_camera()


atexit.register(cleanup)


if __name__ == "__main__":
    if initialize_system():
        app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
    else:
        print("Critical components failed to initialize. Exiting.")
