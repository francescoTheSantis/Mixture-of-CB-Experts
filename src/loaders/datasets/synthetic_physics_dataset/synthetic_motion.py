import os
import cv2
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from env import DATA_PATH, OPENAI_API_KEY
except:
    import sys
    from pathlib import Path
    # Add the project root to path (3 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from env import DATA_PATH

# -----------------------------
# CONFIG
# -----------------------------
FPS = 30
FRAME_SIZE = (256, 256)
BALL_RADIUS = 6

# These will be overridden by function parameters
DEFAULT_OUTPUT_DIR = None  # Set dynamically
DEFAULT_ACCELERATION_VALUES = [0.5]

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def normalize_position(x, y):
    """Normalize pixel coordinates to [0, 10] range"""
    x_norm = (x / FRAME_SIZE[0]) * 10.0
    y_norm = (y / FRAME_SIZE[1]) * 10.0
    return x_norm, y_norm

def draw_ball(frame, center, radius=BALL_RADIUS, color=(0, 0, 255)):
    cv2.circle(frame, (int(center[0]), int(center[1])), radius, color, -1)
    return frame

def generate_linear_uniform_motion(num_frames):
    """
    Straight-line constant velocity motion
    Stops generation when ball goes out of bounds
    """
    x0 = np.random.uniform(BALL_RADIUS + 5, FRAME_SIZE[0] - BALL_RADIUS - 5)
    y0 = np.random.uniform(BALL_RADIUS + 5, FRAME_SIZE[1] - BALL_RADIUS - 5)

    angle = np.random.uniform(0, 2*np.pi)
    speed = np.random.uniform(1.0, 5.0)  # pixels per frame

    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)

    positions = []
    for i in range(num_frames):
        x = x0 + vx*i
        y = y0 + vy*i
        # Check if position is within bounds (with margin for ball radius)
        if x < BALL_RADIUS or x > FRAME_SIZE[0]-BALL_RADIUS or \
           y < BALL_RADIUS or y > FRAME_SIZE[1]-BALL_RADIUS:
            # Stop generation if ball goes out of bounds
            break
        positions.append((x, y))
    
    # Ensure we have at least 2 frames
    if len(positions) < 2:
        positions = [(x0, y0), (x0, y0)]
    
    return np.array(positions), (vx, vy)

def generate_linear_acc_motion(num_frames, acceleration_values):
    """
    Straight-line uniformly accelerated motion
    Stops generation when ball goes out of bounds
    """
    x0 = np.random.uniform(BALL_RADIUS + 5, FRAME_SIZE[0] - BALL_RADIUS - 5)
    y0 = np.random.uniform(BALL_RADIUS + 5, FRAME_SIZE[1] - BALL_RADIUS - 5)

    angle = np.random.uniform(0, 2*np.pi)
    
    a = np.random.choice(acceleration_values)  # Pick from user-defined acceleration values
    v0 = np.random.uniform(0.0, 2.0)  # initial speed

    ax = a * np.cos(angle)
    ay = a * np.sin(angle)
    v0x = v0 * np.cos(angle)
    v0y = v0 * np.sin(angle)

    positions = []
    for i in range(num_frames):
        x = x0 + v0x*i + 0.5*ax*i*i
        y = y0 + v0y*i + 0.5*ay*i*i
        # Check if position is within bounds (with margin for ball radius)
        if x < BALL_RADIUS or x > FRAME_SIZE[0]-BALL_RADIUS or \
           y < BALL_RADIUS or y > FRAME_SIZE[1]-BALL_RADIUS:
            # Stop generation if ball goes out of bounds
            break
        positions.append((x, y))
    
    # Ensure we have at least 2 frames
    if len(positions) < 2:
        positions = [(x0, y0), (x0, y0)]
    
    return np.array(positions), (v0x, v0y, ax, ay)

# -----------------------------
# VELOCITY COMPUTATION
# -----------------------------
def compute_velocity_uniform(x0, y0, xt, yt, t):
    dist = np.sqrt((xt-x0)**2 + (yt-y0)**2)
    return dist / t, "v_t = distance / t"

def compute_velocity_acc(x0, y0, xt, yt, a):
    dist = np.sqrt((xt-x0)**2 + (yt-y0)**2)
    v = np.sqrt(2 * a * dist)
    return v, "v_t = sqrt(2 * a * distance)"

# -----------------------------
# DATASET GENERATION
# -----------------------------
def generate_dataset(n_samples=300, output_dir=None, acceleration_values=None):
    """
    Generate synthetic motion dataset.
    
    Args:
        n_samples: Number of video samples to generate
        output_dir: Output directory (defaults to DATA_PATH/synthetic_motion)
        acceleration_values: List of acceleration values to use (defaults to [0.5])
    """
    if output_dir is None:
        output_dir = os.path.join(DATA_PATH, "synthetic_motion")
    if acceleration_values is None:
        acceleration_values = DEFAULT_ACCELERATION_VALUES
    
    os.makedirs(output_dir, exist_ok=True)
    videos_dir = Path(output_dir) / "videos"
    ann_dir = Path(output_dir) / "annotations"
    videos_dir.mkdir(exist_ok=True)
    ann_dir.mkdir(exist_ok=True)

    annotations = []

    print(f"Generating synthetic motion dataset with {n_samples} samples...")
    for idx in tqdm(range(n_samples), desc="Generating videos"):
        num_frames = np.random.randint(20, 80)   # variable length
        is_uniform = np.random.rand() < 0.5      # type of motion

        if is_uniform:
            positions, params = generate_linear_uniform_motion(num_frames)
        else:
            positions, params = generate_linear_acc_motion(num_frames, acceleration_values)

        # Write video
        vid_path = str(videos_dir / f"sample_{idx}.mp4")
        writer = cv2.VideoWriter(
            vid_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE
        )

        for p in positions:
            frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            draw_ball(frame, p)
            writer.write(frame)
        writer.release()

        # Create annotation dictionary
        x0, y0 = positions[0]
        xt, yt = positions[-1]
        actual_frames = len(positions)  # Use actual number of frames generated
        t = actual_frames / FPS

        # Normalize positions to [0, 10] range
        x0_norm, y0_norm = normalize_position(x0, y0)
        xt_norm, yt_norm = normalize_position(xt, yt)

        if is_uniform:
            # magnitude not needed; direction encoded in x0,y0 → xt,yt
            v_t, eq = compute_velocity_uniform(x0, y0, xt, yt, t)
        else:
            # acceleration magnitude from (ax,ay)
            _, _, ax, ay = params
            a = np.sqrt(ax*ax + ay*ay)
            v_t, eq = compute_velocity_acc(x0, y0, xt, yt, a)

        ann = {
            "video": f"videos/sample_{idx}.mp4",
            "motion_type": "uniform" if is_uniform else "accelerated",
            "num_frames": actual_frames,
            "fps": FPS,
            "initial_position": [float(x0_norm), float(y0_norm)],
            "final_position": [float(xt_norm), float(yt_norm)],
            "initial_position_pixels": [float(x0), float(y0)],
            "final_position_pixels": [float(xt), float(yt)],
            "time_final": float(t),
            "velocity_final": float(v_t),
            "velocity_equation": eq,
        }
        
        # Add acceleration to annotation if accelerated motion
        if not is_uniform:
            _, _, ax, ay = params
            a = np.sqrt(ax*ax + ay*ay)
            ann["acceleration"] = float(a)

        with open(ann_dir / f"sample_{idx}.json", "w") as f:
            json.dump(ann, f, indent=4)

        annotations.append(ann)

    # Create train/val/test split
    idxs = list(range(n_samples))
    train, test = train_test_split(idxs, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.25, random_state=42)

    split_dict = {"train": train, "val": val, "test": test}
    with open(Path(output_dir) / "splits.json", "w") as f:
        json.dump(split_dict, f, indent=4)

    print(f"Dataset generation complete! {n_samples} samples created.")
    return output_dir

# -----------------------------
# VISUALIZATION
# -----------------------------
def visualize_sample(idx, output_dir=None):
    """Visualize a sample from the dataset."""
    if output_dir is None:
        output_dir = os.path.join(DATA_PATH, "synthetic_motion")
    
    ann_path = Path(output_dir) / "annotations" / f"sample_{idx}.json"
    with open(ann_path, "r") as f:
        ann = json.load(f)

    print("ANNOTATIONS:")
    print(json.dumps(ann, indent=4))

    cap = cv2.VideoCapture(str(Path(output_dir) / ann["video"]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Sample Video", frame)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic motion dataset')
    parser.add_argument('--n_samples', type=int, default=300, help='Number of samples to generate')
    parser.add_argument('--acceleration_values', type=float, nargs='+', default=[0.5],
                        help='List of acceleration values')
    parser.add_argument('--visualize', action='store_true', help='Visualize first sample after generation')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Generate dataset
    output_dir = generate_dataset(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        acceleration_values=args.acceleration_values
    )
    
    # Only visualize if --visualize flag is passed
    if args.visualize:
        try:
            visualize_sample(0, output_dir)
        except Exception as e:
            print(f"Visualization failed (this is expected on headless servers): {e}")
            print("Videos are saved in the output directory.")
