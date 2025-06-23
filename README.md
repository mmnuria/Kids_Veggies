# Kids&Veggies

**Kids&Veggies** is an interactive educational application for children aged 4 to 8, designed to teach about fruits and vegetables using implicit human-computer interaction and context awareness. It combines facial recognition, speech processing, and augmented reality to create an immersive and playful learning experience.

---

## Author

- **Nuria Manzano Mata**
- Course: Ubiquitous Computing and Ambient Intelligence
- Type: Individual project
- Grade: 10

---

## Technologies Used

- **Facial Recognition**: `face-recognition`, `dlib`, `opencv`, `numpy`
- **Voice Recognition**: `SpeechRecognition`, `PyAudio`, `audioop-lts`
- **Augmented Reality**: `opencv-contrib`, `pygfx`, `trimesh`, `gltflib`
- **3D Rendering**: `rendercanvas`, `pylinalg`, `imageio`
- **Context Awareness**: `dataclasses-json`, `marshmallow`, `arrow`
- **Utilities**: `matplotlib`, `pillow`, `joblib`, `requests`, `yaml`

All dependencies are listed in `requirements.txt`.

---

## Key Features

- **Voice-Driven Navigation**: Children use voice commands like "start", "back", or "exit".
- **Facial Recognition**: Detects and logs users via face vectors—no passwords required.
- **Augmented Reality Games**:
  - *Discover & Name*: Identify food by name using physical ArUco markers.
  - *Find the Fruits*: Name visible 3D fruit models.
  - *Group by Category*: Classify items as fruits or vegetables.
  - *Memory Game*: Repeat a sequence of items shown in AR.
- **User Profiles**:
  - Stores name, language, scores, and game history in `usuarios.json`.
  - Progress tracked over time with statistical summaries.
- **Context-Aware UI**: Application reacts to the presence of a user and adapts interface flow accordingly.
- **Multilingual Support**: Language preference saved (UI not yet fully translated).

---

## Project Structure

KidsAndVeggies/
├── ar/ # AR marker detection and rendering
├── config/ # Camera calibration
├── data/ # User data (JSON)
├── media/ # 3D models for AR games
├── models/ # AR model logic
├── modules/ # Core logic (games, users, etc.)
├── utils/ # Coordinate and image utilities
├── main.py # Application entry point
└── requirements.txt # Dependencies

---

## How It Works

1. **Startup**:
   - Facial recognition starts automatically.
   - User logs in or registers via voice.
2. **Main Menu**:
   - Access account, progress, or start a game via voice commands.
3. **Game Flow**:
   - Choose between *training* or *evaluation*.
   - Select and play one of four games.
   - Voice responses determine score; progress is saved.

Each state is managed via a dynamic state machine with visual and audio feedback.

---

## Achievements

- Fully functional multi-modal interaction system.
- Modular code with reusable components and clear folder separation.
- State-driven UI flow with robust voice and camera input.
- Games aligned with educational goals (categorization, memory, vocabulary).

---

## Possible Improvements

- Real-time language switch for the full UI.
- "Minigames mode" for rapid game sessions.
- Improved texturing of 3D models.

---

## Installation

1. Clone the repo.
2. Set up a Python virtual environment.
3. Install dependencies:
   ´pip install -r requirements.txt´
5. Run the application:
  ´python main.py´

Camera and microphone access are required.

---

## Final Notes

This project was developed from scratch without prior experience in mixed reality or HCI. It stands as a proof of concept with real educational potential, promoting healthy habits through playful engagement.




