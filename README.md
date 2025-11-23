# Gesture Controlled Mouse

This repository contains a small prototype that uses MediaPipe for hand tracking and `pyautogui` to control the system mouse with an index-finger gesture.

## Setup

1. Create a Python 3.11 virtual environment (recommended):

```bash
cd "/Users/aryanrajput/Downloads/snake game"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

2. Run the demo:

```bash
source .venv/bin/activate
python main.py
# Or use the provided wrapper:
./run.sh
```

# AR Gesture-Controlled Mouse

This project implements an Augmented Reality (AR) gesture-controlled mouse using computer vision. It uses your webcam to detect hand gestures and overlays AR graphics for feedback, allowing you to control your mouse pointer with gestures on macOS.

## Features
- Real-time hand tracking with MediaPipe
- Mouse control via hand gestures
- AR overlays (e.g., circles, pointers) for visual feedback
- Cross-platform (tested on macOS)

## Setup
1. **Clone the repository**
	```bash
	git clone https://github.com/aryan594-tech/gesture-snake.git
	cd gesture-snake
	```
2. **Create and activate a Python virtual environment**
	```bash
	python3.11 -m venv .venv
	source .venv/bin/activate
	```
3. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```

## macOS Permissions
To enable mouse control and AR overlays, grant Accessibility and Screen Recording permissions to Terminal and Python:
- Go to **System Settings > Privacy & Security > Accessibility**
- Add Terminal and Python
- Go to **System Settings > Privacy & Security > Screen Recording**
- Add Terminal and Python

## Usage
Run the main script:
```bash
source .venv/bin/activate
python main.py
```

## Assets
AR overlay images (e.g., `assets/yellow_circles.png`) are used for visual feedback. You can add or replace images in the `assets/` folder.

## Requirements
- Python 3.11+
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

## Troubleshooting
- If you see errors related to permissions, check System Settings.
- For library compatibility issues, use the pinned versions in `requirements.txt`.

## License
MIT
```

Replace `YOUR_USERNAME/YOUR_REPO` with your repository information.

## Notes
- The `.venv` directory is intentionally ignored; do not commit it.
- If you prefer conda/mambaforge for binary dependencies on macOS, I can provide an `environment.yml`.

Have fun — let me know if you want me to initialize the GitHub repository and push the initial commit for you.
