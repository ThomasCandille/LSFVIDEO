import cv2
import numpy as np
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================
# CONFIGURATION
# ==============================

INPUT_DIR = "bonjour"
OUTPUT_DIR = f"training_data/{INPUT_DIR}"
NB_VARIATIONS = 10
SEED = None
VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mov", ".mkv", ".m4v")
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUTPUT_RESIZE_FACTOR = 0.5
OUTPUT_FPS_FACTOR = 1.0

# Augmentation ranges tuned for realistic camera-domain shift.
BRIGHTNESS_RANGE = (-0.12, 0.12)
CONTRAST_RANGE = (0.85, 1.20)
SATURATION_RANGE = (0.85, 1.20)
HUE_RANGE = (-0.03, 0.03)
GAMMA_RANGE = (0.90, 1.10)
NOISE_STD_RANGE = (0.0, 0.02)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)


# ==============================
# PARAMÈTRES ALÉATOIRES (PAR VIDÉO)
# ==============================

def generate_params():
    blur_probability = 0.25
    downscale_probability = 0.30
    jpeg_probability = 0.20

    downscale_factor = 1.0
    if random.random() < downscale_probability:
        downscale_factor = random.uniform(0.70, 0.90)

    return {
        "brightness": random.uniform(*BRIGHTNESS_RANGE),
        "contrast": random.uniform(*CONTRAST_RANGE),
        "saturation": random.uniform(*SATURATION_RANGE),
        "hue": random.uniform(*HUE_RANGE),
        "gamma": random.uniform(*GAMMA_RANGE),
        "noise_std": random.uniform(*NOISE_STD_RANGE),
        "blur": random.random() < blur_probability,
        "downscale_factor": downscale_factor,
        "jpeg_quality": random.randint(65, 95) if random.random() < jpeg_probability else 0,
    }


# ==============================
# AUGMENTATION FRAME
# ==============================

def augment_frame(frame, params, height, width):
    # Work in float32 [0, 1] for stable math and fast NumPy/OpenCV ops.
    frame_f = frame.astype(np.float32) / 255.0

    # Brightness + contrast
    frame_f = frame_f * params["contrast"] + params["brightness"]
    frame_f = np.clip(frame_f, 0.0, 1.0)

    # Saturation + hue in HSV space (OpenCV hue range: [0, 179]).
    hsv = cv2.cvtColor((frame_f * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= params["saturation"]
    hsv[..., 0] = (hsv[..., 0] + params["hue"] * 180.0) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1], 0.0, 255.0)
    frame_f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Gamma shift for mild exposure curve variations.
    frame_f = np.power(frame_f, params["gamma"])

    # Additive Gaussian sensor noise.
    noise_std = params["noise_std"]
    if noise_std > 0.0:
        noise = np.random.normal(0.0, noise_std, frame_f.shape).astype(np.float32)
        frame_f = frame_f + noise

    frame_f = np.clip(frame_f, 0.0, 1.0)
    frame_u8 = (frame_f * 255.0).astype(np.uint8)

    if params["blur"]:
        frame_u8 = cv2.GaussianBlur(frame_u8, (3, 3), 0)

    downscale_factor = params["downscale_factor"]
    if downscale_factor < 1.0:
        new_w = max(1, int(width * downscale_factor))
        new_h = max(1, int(height * downscale_factor))
        frame_u8 = cv2.resize(frame_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frame_u8 = cv2.resize(frame_u8, (width, height), interpolation=cv2.INTER_LINEAR)

    jpeg_quality = params["jpeg_quality"]
    if jpeg_quality > 0:
        ok, enc = cv2.imencode(".jpg", frame_u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if ok:
            decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if decoded is not None:
                frame_u8 = decoded

    return frame_u8


# ==============================
# TRAITEMENT VIDÉO
# ==============================

def process_video(input_path, output_path, params, report_every_frames=0):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("Impossible d'ouvrir la vidéo")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    output_fps = max(1.0, float(fps) * OUTPUT_FPS_FACTOR)
    output_width = max(2, int(width * OUTPUT_RESIZE_FACTOR))
    output_height = max(2, int(height * OUTPUT_RESIZE_FACTOR))

    # Some encoders are more stable with even dimensions.
    if output_width % 2 != 0:
        output_width -= 1
    if output_height % 2 != 0:
        output_height -= 1

    # Initialize MP4 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))

    if not out.isOpened():
        raise Exception("Impossible de créer la vidéo MP4")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if output_width != width or output_height != height:
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

        augmented = augment_frame(frame, params, output_height, output_width)
        out.write(augmented)

        frame_count += 1
        if report_every_frames > 0 and frame_count % report_every_frames == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()

    if frame_count == 0:
        raise Exception("Aucune frame n'a ete lue depuis la video d'entree")

    return frame_count


def process_task(task):
    input_video = task["input_video"]
    output_path = task["output_path"]
    params = task["params"]

    start = time.perf_counter()
    frame_count = process_video(input_video, output_path, params, report_every_frames=0)
    duration = time.perf_counter() - start

    return {
        "input_video": input_video,
        "output_path": output_path,
        "frames": frame_count,
        "duration_s": duration,
    }


def list_input_videos(input_dir):
    if not os.path.isdir(input_dir):
        raise Exception(f"Le dossier d'entree est introuvable: {input_dir}")

    videos = []
    for filename in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, filename)
        if not os.path.isfile(full_path):
            continue
        if os.path.splitext(filename)[1].lower() in VIDEO_EXTENSIONS:
            videos.append(full_path)

    if not videos:
        raise Exception(f"Aucune video trouvee dans {input_dir} (extensions: {VIDEO_EXTENSIONS})")

    return videos


# ==============================
# GÉNÉRATION DATASET
# ==============================

def generate_dataset():
    input_videos = list_input_videos(INPUT_DIR)
    print(f"{len(input_videos)} video(s) detectee(s) dans {INPUT_DIR}")

    tasks = []
    for input_video in input_videos:
        input_stem = os.path.splitext(os.path.basename(input_video))[0]
        for i in range(NB_VARIATIONS):
            output_filename = f"{input_stem}_aug_{i}.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            tasks.append(
                {
                    "input_video": input_video,
                    "output_path": output_path,
                    "params": generate_params(),
                }
            )

    print(f"{len(tasks)} tache(s) d'augmentation a executer.")
    print(f"Execution parallele avec {MAX_WORKERS} worker(s).")

    completed = 0
    start_all = time.perf_counter()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            print(
                f"[{completed}/{len(tasks)}] {os.path.basename(result['output_path'])} "
                f"- {result['frames']} frames en {result['duration_s']:.2f}s"
            )

    total_duration = time.perf_counter() - start_all
    print(f"\n✅ Dataset généré avec succès pour toutes les vidéos en {total_duration:.2f}s.")


# ==============================
# EXECUTION
# ==============================

if __name__ == "__main__":
    generate_dataset()