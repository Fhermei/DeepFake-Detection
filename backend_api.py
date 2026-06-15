import os
import gc
import cv2
import numpy as np
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'deepfake_detector_final.h5')
model = None

def get_model():
    global model
    if model is None:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded!")
    return model

print("Loading model...")
get_model()
print("Ready!")


def extract_frames(video_path, num_frames=30, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames == 0:
        cap.release()
        return None

    indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            frames.append((frame / 255.0).astype(np.float16))
        else:
            frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.float16))

    cap.release()

    while len(frames) < num_frames:
        frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.float16))

    return np.array(frames[:num_frames], dtype=np.float16)


@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file received'}), 400

    video_file = request.files['video']
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'upload.mp4')
    video_file.save(temp_path)

    try:
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = round(total_frames / fps, 2) if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        frames = extract_frames(temp_path)
        if frames is None:
            return jsonify({'error': 'Could not read video frames'}), 400

        frames_input = np.expand_dims(frames.astype(np.float32), axis=0)
        del frames
        gc.collect()

        m = get_model()
        prediction = m.predict(frames_input, verbose=0, batch_size=1)
        probability = float(prediction[0][0])
        del frames_input
        gc.collect()

        if probability < 0.5:
            result = "DEEPFAKE"
            confidence = round((1 - probability) * 100, 2)
        else:
            result = "REAL"
            confidence = round(probability * 100, 2)

        return jsonify({
            'result': result,
            'confidence': confidence,
            'probability': round(probability, 6),
            'deepfake_probability': round((1 - probability) * 100, 2),
            'real_probability': round(probability * 100, 2),
            'video_info': {
                'duration': duration,
                'total_frames': total_frames,
                'fps': round(fps, 2),
                'resolution': f"{width}x{height}"
            }
        })

    except Exception as e:
        gc.collect()
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'model_loaded': model is not None})


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Deepfake Detection API is running!'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)