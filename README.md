# Speech Recognition Wheelchair

## Overview
This project is a **Python-based simulation** for integrating speech recognition features into a wheelchair. The goal is to enable hands-free control of the wheelchair using voice commands, making mobility easier for individuals with physical disabilities.

## Features
- **Voice Command Processing:** Converts spoken words into executable wheelchair movements.
- **Natural Language Processing:** Recognizes and interprets a variety of voice commands.
- **Real-time Simulation:** Demonstrates how a wheelchair would respond to voice inputs.
- **Machine Learning Integration:** Uses clustering and neural networks for improved decision-making.
- **Data Analysis & Visualization:** Utilizes various libraries for data processing and visualization.
- **Obstacle Detection (Future Scope):** Plans for integrating sensors to enhance navigation safety.

## Technologies Used
- **Python**
- **SpeechRecognition (Google Speech API)**
- **PyAudio**
- **NLTK (Natural Language Processing Toolkit)**
- **Pygame (for simulation visualization)**
- **NumPy & Pandas** (for data manipulation)
- **Matplotlib & Seaborn** (for data visualization)
- **Scikit-learn (KMeans Clustering)**
- **TensorFlow/Keras** (for deep learning applications)
- **OpenCV (cv2)** (for image processing and computer vision)
- **Multithreading & Queues** (for asynchronous processing)
- **HeapQ & Deques** (for optimized data handling)

## Installation
### Prerequisites
Ensure you have Python installed (preferably Python 3.8+). You can install the required dependencies using:

```sh
pip install SpeechRecognition pyaudio nltk pygame numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python
```

If you face issues installing **PyAudio**, try:
```sh
pip install pipwin
pipwin install pyaudio
```

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/Oscah-jay/speech-recognition-wheelchair.git
   cd speech-recognition-wheelchair
   ```
2. Run the program:
   ```sh
   python main.py
   ```
3. Speak commands like:
   - "Go forward"
   - "Turn left"
   - "Turn right"
   - "Stop"

## File Structure
```
📂 speech-recognition-wheelchair
├── 📄 general.py                 # Main script for voice control simulation
├── 📄 voice_test.py                 # Script for voice testing
├── 📂 assets                  # (Optional) Contains images or audio files
├── 📄 wheelchair_simulation.py         # Alternative script for compatibility with basic voice tests
├── 📄 README.md                # Project documentation
```

## Future Enhancements
- Improve speech recognition accuracy with deep learning models.
- Implement a real-time wheelchair navigation system using Raspberry Pi.
- Add obstacle detection using ultrasonic or LiDAR sensors.
- Integrate computer vision for advanced obstacle recognition.
- Implement reinforcement learning for optimized movement patterns.

## Contributions
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.

---
Made with ❤️ by [Oscah-jay](https://github.com/Oscah-jay)

