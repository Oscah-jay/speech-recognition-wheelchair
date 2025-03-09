import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import math
import random
import time

# For voice recognition
try:
    import pyaudio
    import wave
    import struct
    import numpy as np
    from scipy.fftpack import fft
    VOICE_AVAILABLE = True
except ImportError:
    print("PyAudio, wave, or scipy not available. Simulating voice commands instead.")
    VOICE_AVAILABLE = False

class Wheelchair:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0  # linear velocity
        self.omega = 0  # angular velocity
        self.wheel_radius = 0.3  # meters
        self.wheel_base = 0.6  # meters
        self.max_speed = 1.5  # m/s
        self.max_ang_speed = 0.8  # rad/s

    def update(self, v_cmd, omega_cmd, dt):
        # Update velocities
        self.v = v_cmd
        self.omega = omega_cmd
        
        # Update position and orientation
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += self.omega * dt
        
        # Ensure theta stays within 0-2π
        self.theta = self.theta % (2 * math.pi)

class Obstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

class WheelchairSimulation:
    def __init__(self):
        # Simulation parameters
        self.sim_time = 60  # seconds
        self.dt = 0.1  # time step
        self.map_size = 10  # meters
        self.num_obstacles = 5
        self.sensor_range = 2.0  # meters
        self.min_obstacle_dist = 0.8  # minimum safe distance
        
        # Initialize wheelchair
        self.wheelchair = Wheelchair(self.map_size/2, self.map_size/2, 0)
        
        # Create random obstacles
        self.obstacles = []
        for _ in range(self.num_obstacles):
            x = random.uniform(0, self.map_size)
            y = random.uniform(0, self.map_size)
            radius = random.uniform(0.3, 0.5)
            self.obstacles.append(Obstacle(x, y, radius))
        
        # Voice command settings
        self.voice_commands = ['forward', 'backward', 'left', 'right', 'stop']
        self.current_command = 'stop'
        self.voice_recognition_timer = 0
        self.voice_update_interval = 2.0
        
        # Set up figure for animation
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        # Initialize plots
        self.obstacle_patches = []
        for obs in self.obstacles:
            circle = patches.Circle((obs.x, obs.y), obs.radius, fc='r')
            self.obstacle_patches.append(circle)
            self.ax.add_patch(circle)
        
        # Wheelchair representation (triangle)
        whl_size = self.wheelchair.wheel_base / 2
        self.triangle = patches.Polygon([[0, 0], [0, 0], [0, 0]], closed=True, fc='b')
        self.ax.add_patch(self.triangle)
        
        # Sensor lines
        self.sensor_lines = []
        for _ in range(5):
            line, = self.ax.plot([0, 0], [0, 0], 'g-')
            self.sensor_lines.append(line)
        
        # Text for displaying current command
        self.command_text = self.ax.text(0.5, self.map_size-0.5, '', fontsize=14)
        
        # Voice recognition module
        if VOICE_AVAILABLE:
            self.voice_recognizer = VoiceRecognizer()
    
    def update_wheelchair_viz(self):
        # Update triangle representing wheelchair
        whl_size = self.wheelchair.wheel_base / 2
        pts = np.array([[-whl_size, -whl_size/2], 
                        [whl_size, 0], 
                        [-whl_size, whl_size/2]])
        
        # Rotation matrix
        theta = self.wheelchair.theta
        R = np.array([[math.cos(theta), -math.sin(theta)], 
                     [math.sin(theta), math.cos(theta)]])
        
        # Rotate and translate points
        pts_rotated = np.dot(pts, R.T)
        pts_translated = pts_rotated + np.array([self.wheelchair.x, self.wheelchair.y])
        
        # Update triangle
        self.triangle.set_xy(pts_translated)
    
    def update_sensors(self):
        # Update sensor lines
        x, y, theta = self.wheelchair.x, self.wheelchair.y, self.wheelchair.theta
        angles = np.linspace(theta-math.pi/2, theta+math.pi/2, len(self.sensor_lines))
        
        for i, line in enumerate(self.sensor_lines):
            end_x = x + self.sensor_range * math.cos(angles[i])
            end_y = y + self.sensor_range * math.sin(angles[i])
            line.set_data([x, end_x], [y, end_y])
    
    def recognize_voice_command(self):
        if VOICE_AVAILABLE:
            return self.voice_recognizer.recognize_command()
        else:
            # Simulate voice commands
            if random.random() < 0.8:
                return random.choice(self.voice_commands)
            else:
                return 'stop'
    
    def detect_obstacles(self):
        # Check distance to each obstacle
        x, y, theta = self.wheelchair.x, self.wheelchair.y, self.wheelchair.theta
        detected = False
        min_dist = self.sensor_range
        avoid_angle = 0
        
        for obs in self.obstacles:
            dist = math.sqrt((obs.x - x)**2 + (obs.y - y)**2) - obs.radius
            
            if dist < min_dist:
                min_dist = dist
                detected = True
                
                # Calculate angle to obstacle
                obstacle_angle = math.atan2(obs.y - y, obs.x - x)
                angle_diff = ((obstacle_angle - theta + math.pi) % (2*math.pi)) - math.pi
                
                # Turn away from obstacle
                if angle_diff > 0:
                    avoid_angle = -1  # Turn right
                else:
                    avoid_angle = 1   # Turn left
        
        return detected, min_dist, avoid_angle
    
    def animation_step(self, frame):
        # Update voice recognition timer
        self.voice_recognition_timer += self.dt
        if self.voice_recognition_timer >= self.voice_update_interval:
            self.current_command = self.recognize_voice_command()
            self.voice_recognition_timer = 0
        
        # Convert voice commands to motion commands
        v_cmd = 0
        omega_cmd = 0
        
        if self.current_command == 'forward':
            v_cmd = self.wheelchair.max_speed
        elif self.current_command == 'backward':
            v_cmd = -self.wheelchair.max_speed
        elif self.current_command == 'left':
            omega_cmd = self.wheelchair.max_ang_speed
        elif self.current_command == 'right':
            omega_cmd = -self.wheelchair.max_ang_speed
        
        # Detect obstacles
        obstacle_detected, obstacle_dist, avoid_dir = self.detect_obstacles()
        
        # Obstacle avoidance logic
        if obstacle_detected and obstacle_dist < self.min_obstacle_dist:
            # Override commands for safety
            if v_cmd > 0:
                # If moving forward toward obstacle, stop or turn
                v_cmd = 0
                omega_cmd = self.wheelchair.max_ang_speed * avoid_dir
            elif v_cmd < 0:
                # If moving backward toward obstacle, stop
                v_cmd = 0
        
        # Update wheelchair state
        self.wheelchair.update(v_cmd, omega_cmd, self.dt)
        
        # Ensure wheelchair stays within bounds
        self.wheelchair.x = max(min(self.wheelchair.x, self.map_size), 0)
        self.wheelchair.y = max(min(self.wheelchair.y, self.map_size), 0)
        
        # Update visualization
        self.update_wheelchair_viz()
        self.update_sensors()
        self.command_text.set_text(f'Command: {self.current_command}')
        
        # Return all updated objects for animation
        return [self.triangle] + self.sensor_lines + [self.command_text]
    
    def run(self):
        # Create animation
        num_frames = int(self.sim_time / self.dt)
        self.animation = FuncAnimation(
            self.fig, self.animation_step, frames=num_frames,
            interval=self.dt*1000, blit=True, repeat=False
        )
        
        plt.title('Voice-Controlled Wheelchair Simulation')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.show()


class VoiceRecognizer:
    def __init__(self):
        # Initialize PyAudio
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 3 
        
        self.audio = pyaudio.PyAudio()
    
    def record_audio(self):
        # Open audio stream
        stream = self.audio.open(format=self.format,
                             channels=self.channels,
                             rate=self.rate,
                             input=True,
                             frames_per_buffer=self.chunk)
        
        print("Recording...")
        
        # Record audio
        frames = []
        for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        print("Recording complete.")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Convert frames to numpy array
        audio_data = b''.join(frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        
        return audio_data
    
    def recognize_command(self):
        try:
            # Record audio
            audio_data = self.record_audio()
            
            # Simple energy-based detection
            energy = np.mean(np.abs(audio_data))
            
            # Process with FFT for frequency analysis
            fft_data = fft(audio_data)
            fft_data = np.abs(fft_data[:len(fft_data)//2])
            
            # Simple frequency band analysis
            freq_bins = len(fft_data)
            low_freq = int(freq_bins * 100 / self.rate)
            mid_freq_low = int(freq_bins * 500 / self.rate)
            mid_freq_high = int(freq_bins * 1500 / self.rate)
            high_freq_low = int(freq_bins * 1500 / self.rate)
            high_freq = int(freq_bins * 3000 / self.rate)
            
            low_energy = np.sum(fft_data[low_freq:mid_freq_low])
            mid_energy = np.sum(fft_data[mid_freq_low:mid_freq_high])
            high_energy = np.sum(fft_data[high_freq_low:high_freq])
            
            # Simple command classification
            if energy < 1000:  # Threshold needs to be calibrated
                return 'stop'
            elif high_energy > mid_energy and high_energy > low_energy:
                return 'forward'
            elif low_energy > mid_energy and low_energy > high_energy:
                return 'stop'
            elif mid_energy > low_energy and mid_energy > high_energy * 1.5:
                # Further classify left vs right based on frequency distribution
                mid_low = np.sum(fft_data[mid_freq_low:int((mid_freq_low+mid_freq_high)/2)])
                mid_high = np.sum(fft_data[int((mid_freq_low+mid_freq_high)/2):mid_freq_high])
                if mid_high > mid_low:
                    return 'right'
                else:
                    return 'left'
            else:
                return 'backward'
                
        except Exception as e:
            print(f"Error in voice recognition: {e}")
            return 'stop'  # Default to stop for safety
    
    def __del__(self):
        # Clean up PyAudio
        self.audio.terminate()


# Main execution
if __name__ == "__main__":
    print("Voice-Controlled Wheelchair Simulation")
    print("Commands: forward, backward, left, right, stop")
    
    if not VOICE_AVAILABLE:
        print("Running with simulated voice commands.")
    
    sim = WheelchairSimulation()
    sim.run()