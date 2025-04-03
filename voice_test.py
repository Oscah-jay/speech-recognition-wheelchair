# Speech-Controlled Wheelchair Simulation Project
# Based on SDLC Methodology

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import random
import json
import time
import threading
import queue
from tensorflow import keras
import tensorflow as tf
import cv2
import os
from collections import deque
import math
import heapq

# ============================================================
# 1. REQUIREMENTS ANALYSIS SIMULATION
# ============================================================

def create_user_personas():
    """Create detailed user personas based on requirements analysis"""
    personas = [
        {
            "id": 1,
            "name": "Oscah",
            "age": 45,
            "disability": "Spinal cord injury (C5-C6)",
            "tech_proficiency": "Medium",
            "environment": "Urban apartment, works in office",
            "pain_points": [
                "Difficulty navigating crowded spaces",
                "Limited upper body control for manual wheelchair",
                "Fatigue when using voice commands for extended periods"
            ],
            "needs": [
                "Hands-free control system",
                "Precise navigation in tight spaces",
                "Long battery life",
                "Effective operation in noisy environments"
            ]
        },
        {
            "id": 2,
            "name": "Njoro",
            "age": 67,
            "disability": "Multiple Sclerosis",
            "tech_proficiency": "Low",
            "environment": "Suburban home, frequent hospital visits",
            "pain_points": [
                "Hand tremors limit use of joystick controls",
                "Difficulty with complex interfaces",
                "Voice can weaken during the day",
                "Anxiety about wheelchair operation in public"
            ],
            "needs": [
                "Simple voice command interface",
                "Highly sensitive speech recognition",
                "Gentle acceleration and braking",
                "Obstacle detection and avoidance"
            ]
        },
        {
            "id": 3,
            "name": "Jamal",
            "age": 22,
            "disability": "Muscular Dystrophy",
            "tech_proficiency": "High",
            "environment": "University campus, shared dormitory",
            "pain_points": [
                "Progressive weakness makes joystick use difficult",
                "Needs to navigate complex campus environment",
                "Wants independence from personal assistants",
                "Battery anxiety on long days"
            ],
            "needs": [
                "Customizable voice commands",
                "Integration with smartphone",
                "Advanced navigation capabilities",
                "Energy-efficient system",
                "Ability to navigate between buildings autonomously"
            ]
        }
    ]
    return personas

def generate_survey_data(n_respondents=50):
    """Generate simulated survey data from potential users"""
    np.random.seed(42)
    
    columns = [
        "respondent_id",
        "age",
        "disability_type",
        "tech_proficiency",
        "important_speech_accuracy",
        "important_battery_life",
        "important_obstacle_detection",
        "important_ease_of_use",
        "important_customization",
        "current_mobility_satisfaction",
        "willing_to_use_voice_control"
    ]
    
    survey_data = pd.DataFrame(columns=columns)
    
    for i in range(n_respondents):
        row = {
            "respondent_id": i + 1,
            "age": np.random.randint(18, 85),
            "disability_type": np.random.choice(["Spinal Cord Injury", "MS", "ALS", "Muscular Dystrophy", "Cerebral Palsy", "Other"]),
            "tech_proficiency": np.random.choice(["Low", "Medium", "High"], p=[0.3, 0.5, 0.2]),
            "important_speech_accuracy": np.random.randint(1, 11),
            "important_battery_life": np.random.randint(1, 11),
            "important_obstacle_detection": np.random.randint(1, 11),
            "important_ease_of_use": np.random.randint(1, 11),
            "important_customization": np.random.randint(1, 11),
            "current_mobility_satisfaction": np.random.randint(1, 11),
            "willing_to_use_voice_control": np.random.choice(["Yes", "Maybe", "No"], p=[0.7, 0.2, 0.1])
        }
        survey_data = pd.concat([survey_data, pd.DataFrame([row])], ignore_index=True)
    
    return survey_data

def analyze_survey_data(survey_data):
    """Analyze survey data to extract requirements"""
    important_features = [
        "important_speech_accuracy",
        "important_battery_life",
        "important_obstacle_detection",
        "important_ease_of_use",
        "important_customization"
    ]
    
    feature_importance = survey_data[important_features].mean().sort_values(ascending=False)
    
    # Segment users based on their preferences
    X = survey_data[important_features].values
    kmeans = KMeans(n_clusters=3, random_state=42)
    survey_data['user_segment'] = kmeans.fit_predict(X)
    
    # Visualize the results
    def visualize_survey_results(survey_data, feature_importance):
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot feature importance
        feature_importance.plot(kind='bar', ax=ax1)
        ax1.set_title('Feature Importance Rankings')
        ax1.set_ylabel('Average Importance (1-10)')
        
        # Plot user segments
        sns.scatterplot(
            x="important_speech_accuracy", 
            y="important_ease_of_use",
            hue="user_segment",
            data=survey_data,
            ax=ax2
        )
        ax2.set_title('User Segments')
        
        plt.tight_layout()
        plt.savefig('survey_analysis.png')
        plt.close()
    
    try:
        visualize_survey_results(survey_data, feature_importance)
    except Exception as e:
        print(f"Could not visualize survey results: {e}")
    
    return {
        "feature_importance": feature_importance,
        "segmented_data": survey_data,
        "cluster_centers": kmeans.cluster_centers_
    }

def define_requirements(analysis_results):
    """Define technical and functional requirements based on analysis"""
    requirements = {
        "functional_requirements": [
            {
                "id": "FR1",
                "title": "Speech Command Recognition",
                "description": "System must recognize at least 20 distinct voice commands with accuracy >95% in quiet environments",
                "priority": "High",
                "source": "User interviews and survey analysis"
            },
            {
                "id": "FR2",
                "title": "Obstacle Detection and Avoidance",
                "description": "System must detect obstacles within 3 meters and automatically adjust path",
                "priority": "High",
                "source": "Safety requirements and user needs"
            },
            {
                "id": "FR3",
                "title": "User Interface Customization",
                "description": "Users must be able to customize and add voice commands",
                "priority": "Medium",
                "source": "User segment 3 preferences"
            },
            {
                "id": "FR4",
                "title": "Command Confirmation",
                "description": "System provides audio feedback confirming recognized commands",
                "priority": "Medium",
                "source": "User testing feedback"
            },
            {
                "id": "FR5",
                "title": "Speed Control",
                "description": "Variable speed control through voice commands (slow, normal, fast)",
                "priority": "High",
                "source": "Safety requirements and user feedback"
            }
        ],
        "technical_requirements": [
            {
                "id": "TR1",
                "title": "Speech Recognition Accuracy",
                "description": ">95% accuracy in quiet environments, >85% in noisy environments",
                "priority": "High",
                "source": "Survey analysis"
            },
            {
                "id": "TR2",
                "title": "Battery Life",
                "description": "Minimum 8 hours of continuous operation",
                "priority": "High",
                "source": "Feature importance analysis"
            },
            {
                "id": "TR3",
                "title": "Response Time",
                "description": "System must respond to voice commands within 500ms",
                "priority": "High",
                "source": "Safety requirements"
            },
            {
                "id": "TR4",
                "title": "Sensor Integration",
                "description": "Integration of ultrasonic, infrared sensors, and camera for 360° coverage",
                "priority": "Medium",
                "source": "Technical analysis"
            },
            {
                "id": "TR5",
                "title": "System Weight",
                "description": "Additional components must not add more than 2kg to wheelchair",
                "priority": "Medium",
                "source": "User mobility concerns"
            }
        ]
    }
    return requirements

def create_journey_map(persona_id, personas):
    """Create user journey map for a specific persona"""
    persona = next((p for p in personas if p["id"] == persona_id), None)
    
    if not persona:
        return None
    
    journey_points = [
        {
            "stage": "Setup",
            "actions": ["Power on system", "Calibrate voice recognition", "Set destination"],
            "emotions": "Neutral to Slightly Anxious",
            "pain_points": "Initial setup can be complex, especially for low tech proficiency users",
            "opportunities": "Simplified startup sequence, preset configurations"
        },
        {
            "stage": "Navigation Start",
            "actions": ["Issue voice command to start", "Confirm direction", "Begin movement"],
            "emotions": "Focused, Slight apprehension",
            "pain_points": "Uncertainty about system understanding commands correctly",
            "opportunities": "Clear feedback mechanisms, confidence indicators for commands"
        },
        {
            "stage": "Ongoing Navigation",
            "actions": ["Issue directional commands", "Adjust speed", "Navigate around obstacles"],
            "emotions": "Engaged to Confident",
            "pain_points": "Repeated commands in noisy environments, fatigue from constant commands",
            "opportunities": "Contextual command understanding, route memory"
        },
        {
            "stage": "Obstacle Encounter",
            "actions": ["System detects obstacle", "System slows/stops", "User redirects"],
            "emotions": "Momentary concern to Relief",
            "pain_points": "Unexpected stops, ambiguity about why system stopped",
            "opportunities": "Clear explanation of detected obstacles, suggested alternative paths"
        },
        {
            "stage": "Destination Arrival",
            "actions": ["System announces arrival", "User confirms stop", "System powers down or enters standby"],
            "emotions": "Satisfaction, Relief",
            "pain_points": "Precise positioning at destination, especially in tight spaces",
            "opportunities": "Fine-tuning position commands, destination memory"
        }
    ]
    
    return {
        "persona": persona,
        "journey": journey_points
    }

def run_requirements_analysis():
    """Run the full requirements analysis process"""
    print("Starting Requirements Analysis...")
    
    # Create personas
    personas = create_user_personas()
    print(f"Created {len(personas)} user personas")
    
    # Generate and analyze survey data
    survey_data = generate_survey_data(50)
    print(f"Generated survey data with {len(survey_data)} responses")
    
    analysis = analyze_survey_data(survey_data)
    print("Analyzed survey data")
    
    # Define requirements
    requirements = define_requirements(analysis)
    print(f"Defined {len(requirements['functional_requirements'])} functional requirements and {len(requirements['technical_requirements'])} technical requirements")
    
    # Create journey maps
    journey_maps = [create_journey_map(persona["id"], personas) for persona in personas]
    print(f"Created {len(journey_maps)} user journey maps")
    
    # Save all artifacts
    with open('requirements_analysis.json', 'w') as f:
        json.dump({
            "personas": personas,
            "requirements": requirements,
            "journey_maps": journey_maps,
            "feature_importance": analysis["feature_importance"].to_dict(),
        }, f, indent=2)
    
    print("Requirements Analysis completed and saved to requirements_analysis.json")
    return personas, requirements, journey_maps, analysis


# ============================================================
# 2. SYSTEM DESIGN SIMULATION
# ============================================================

class SystemArchitecture:
    """Class to represent and visualize the system architecture"""
    def __init__(self):
        self.components = {
            "speech_recognition": {
                "description": "Processes and interprets voice commands",
                "inputs": ["microphone_array"],
                "outputs": ["command_processor"],
                "technologies": ["TensorFlow", "PyTorch", "NLP models"],
                "submodules": ["noise_filtering", "speech_to_text", "intent_recognition", "command_verification"]
            },
            "sensor_system": {
                "description": "Collects environmental data for navigation and safety",
                "inputs": ["environment"],
                "outputs": ["navigation_system", "safety_system"],
                "technologies": ["Ultrasonic sensors", "Infrared sensors", "Camera", "LiDAR (optional)"],
                "submodules": ["obstacle_detection", "distance_measurement", "terrain_analysis"]
            },
            "navigation_system": {
                "description": "Plans and executes movement based on commands and sensor data",
                "inputs": ["command_processor", "sensor_system"],
                "outputs": ["motor_control_system"],
                "technologies": ["Path planning algorithms", "Computer vision", "SLAM"],
                "submodules": ["path_planning", "mapping", "localization"]
            },
            "safety_system": {
                "description": "Ensures safe operation under all conditions",
                "inputs": ["sensor_system", "motor_control_system"],
                "outputs": ["motor_control_system", "user_interface"],
                "technologies": ["Rule-based system", "Emergency protocols"],
                "submodules": ["collision_prevention", "stability_control", "emergency_stop"]
            },
            "motor_control_system": {
                "description": "Controls wheelchair motors for movement",
                "inputs": ["navigation_system", "safety_system"],
                "outputs": ["wheelchair_motors"],
                "technologies": ["PWM control", "Motor drivers"],
                "submodules": ["speed_control", "direction_control", "motor_feedback"]
            },
            "user_interface": {
                "description": "Provides feedback and control options to user",
                "inputs": ["speech_recognition", "safety_system", "power_management"],
                "outputs": ["user_feedback", "customization_options"],
                "technologies": ["Audio feedback", "Optional visual display", "Haptic feedback"],
                "submodules": ["command_confirmation", "status_indicators", "settings_menu"]
            },
            "power_management": {
                "description": "Manages system power consumption and battery",
                "inputs": ["motor_control_system", "all_systems"],
                "outputs": ["power_distribution", "user_interface"],
                "technologies": ["Efficient power controllers", "Battery management system"],
                "submodules": ["power_monitoring", "battery_optimization", "charging_control"]
            }
        }
        
        # Define component connections
        self.connections = [
            {"from": "speech_recognition", "to": "command_processor"},
            {"from": "command_processor", "to": "navigation_system"},
            {"from": "sensor_system", "to": "navigation_system"},
            {"from": "sensor_system", "to": "safety_system"},
            {"from": "navigation_system", "to": "motor_control_system"},
            {"from": "safety_system", "to": "motor_control_system"},
            {"from": "safety_system", "to": "user_interface"},
            {"from": "motor_control_system", "to": "wheelchair_motors"},
            {"from": "power_management", "to": "all_systems"},
            {"from": "power_management", "to": "user_interface"}
        ]
    
    def generate_architecture_diagram(self):
        """Generate a visualization of the system architecture"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes (components)
            for component in self.components:
                G.add_node(component)
            
            # Add edges (connections)
            for connection in self.connections:
                from_comp = connection["from"]
                to_comp = connection["to"]
                if to_comp != "all_systems":  # Skip generic connections
                    G.add_edge(from_comp, to_comp)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                    node_size=3000, font_size=10, font_weight='bold',
                    arrowsize=20, arrows=True)
            
            plt.title('System Architecture', size=15)
            plt.tight_layout()
            plt.savefig('system_architecture.png')
            plt.close()
            
            print("Architecture diagram generated and saved as system_architecture.png")
            return True
        except Exception as e:
            print(f"Could not generate architecture diagram: {e}")
            return False
    
    def export_architecture(self):
        """Export the architecture to a JSON file"""
        with open('system_architecture.json', 'w') as f:
            json.dump({
                "components": self.components,
                "connections": self.connections
            }, f, indent=2)
        print("System architecture exported to system_architecture.json")

class UIPrototype:
    """Class to represent and create UI prototypes"""
    def __init__(self):
        self.screens = {
            "main_screen": {
                "elements": [
                    {"type": "status_indicator", "name": "battery_level", "position": "top-right"},
                    {"type": "status_indicator", "name": "speech_recognition_status", "position": "top-center"},
                    {"type": "button", "name": "emergency_stop", "position": "center", "color": "red"},
                    {"type": "indicator", "name": "current_speed", "position": "bottom-left"},
                    {"type": "indicator", "name": "obstacles_detected", "position": "bottom-right"}
                ],
                "voice_commands": [
                    "Go forward", "Go backward", "Turn left", "Turn right", "Stop",
                    "Speed up", "Slow down", "Emergency stop"
                ]
            },
            "settings_screen": {
                "elements": [
                    {"type": "slider", "name": "voice_sensitivity", "position": "top", "range": [1, 10]},
                    {"type": "toggle", "name": "noise_filtering", "position": "middle-top", "default": True},
                    {"type": "button", "name": "calibrate_voice", "position": "middle", "color": "blue"},
                    {"type": "menu", "name": "command_customization", "position": "middle-bottom"},
                    {"type": "button", "name": "save_settings", "position": "bottom", "color": "green"}
                ],
                "voice_commands": [
                    "Open settings", "Save settings", "Calibrate voice", "Adjust sensitivity",
                    "Toggle noise filter", "Back to main screen"
                ]
            }
        }
    
    def get_command_list(self):
        """Get a complete list of voice commands across all screens"""
        commands = []
        for screen in self.screens.values():
            commands.extend(screen["voice_commands"])
        return list(set(commands))  # Remove duplicates
    
    def generate_ui_mockup(self):
        """Generate a simple visual mockup of the UI"""
        try:
            import cv2
            import numpy as np
            
            # Create empty images for each screen
            screen_height, screen_width = 400, 600
            
            for screen_name, screen_data in self.screens.items():
                # Create a blank screen
                screen = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 240
                
                # Draw title
                title = screen_name.replace('_', ' ').title()
                cv2.putText(screen, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 0), 2, cv2.LINE_AA)
                
                # Draw elements
                y_offset = 100
                for element in screen_data["elements"]:
                    element_name = element["name"].replace('_', ' ').title()
                    element_type = element["type"]
                    
                    # Determine position
                    if "position" in element:
                        position = element["position"]
                        if position == "top-right":
                            pos = (screen_width - 200, y_offset)
                        elif position == "top-center":
                            pos = (screen_width // 2 - 100, y_offset)
                        elif position == "center":
                            pos = (screen_width // 2 - 100, screen_height // 2)
                        elif position == "bottom-left":
                            pos = (50, screen_height - 50)
                        elif position == "bottom-right":
                            pos = (screen_width - 200, screen_height - 50)
                        elif position == "top":
                            pos = (50, y_offset)
                            y_offset += 50
                        elif position == "middle-top":
                            pos = (50, y_offset)
                            y_offset += 50
                        elif position == "middle":
                            pos = (50, y_offset)
                            y_offset += 50
                        elif position == "middle-bottom":
                            pos = (50, y_offset)
                            y_offset += 50
                        elif position == "bottom":
                            pos = (50, screen_height - 100)
                        else:
                            pos = (50, y_offset)
                            y_offset += 50
                    else:
                        pos = (50, y_offset)
                        y_offset += 50
                    
                    # Draw different types of elements
                    if element_type == "button":
                        color = (0, 0, 255)  # Red for emergency, otherwise blue
                        if "color" in element:
                            if element["color"] == "red":
                                color = (0, 0, 255)
                            elif element["color"] == "blue":
                                color = (255, 0, 0)
                            elif element["color"] == "green":
                                color = (0, 255, 0)
                        
                        cv2.rectangle(screen, pos, (pos[0] + 150, pos[1] + 40), color, -1)
                        cv2.putText(screen, element_name, (pos[0] + 10, pos[1] + 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    elif element_type == "status_indicator":
                        cv2.rectangle(screen, pos, (pos[0] + 120, pos[1] + 30), (0, 0, 0), 1)
                        cv2.putText(screen, element_name, (pos[0] + 5, pos[1] + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    elif element_type == "indicator":
                        cv2.rectangle(screen, pos, (pos[0] + 120, pos[1] + 30), (0, 0, 0), 1)
                        cv2.putText(screen, element_name, (pos[0] + 5, pos[1] + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    elif element_type == "slider":
                        cv2.rectangle(screen, pos, (pos[0] + 200, pos[1] + 20), (200, 200, 200), -1)
                        cv2.rectangle(screen, pos, (pos[0] + 100, pos[1] + 20), (0, 255, 0), -1)
                        cv2.circle(screen, (pos[0] + 100, pos[1] + 10), 10, (0, 0, 255), -1)
                        cv2.putText(screen, element_name, (pos[0], pos[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    elif element_type == "toggle":
                        cv2.rectangle(screen, pos, (pos[0] + 60, pos[1] + 30), (200, 200, 200), -1)
                        cv2.rectangle(screen, pos, (pos[0] + 30, pos[1] + 30), (0, 255, 0), -1)
                        cv2.putText(screen, element_name, (pos[0], pos[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    elif element_type == "menu":
                        cv2.rectangle(screen, pos, (pos[0] + 180, pos[1] + 120), (220, 220, 220), -1)
                        cv2.putText(screen, element_name, (pos[0] + 5, pos[1] + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.line(screen, (pos[0], pos[1] + 30), (pos[0] + 180, pos[1] + 30), (0, 0, 0), 1)
                        cv2.putText(screen, "Command 1", (pos[0] + 5, pos[1] + 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(screen, "Command 2", (pos[0] + 5, pos[1] + 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(screen, "Command 3", (pos[0] + 5, pos[1] + 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(screen, "Add New...", (pos[0] + 5, pos[1] + 110), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Draw voice commands area
                cv2.rectangle(screen, (50, screen_height - 150), (screen_width - 50, screen_height - 50), 
                             (220, 220, 220), -1)
                cv2.putText(screen, "Voice Commands:", (60, screen_height - 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                # List some commands
                commands = screen_data["voice_commands"][:4]  # Show just a few
                for i, cmd in enumerate(commands):
                    cv2.putText(screen, f"• {cmd}", (70, screen_height - 110 + i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Save the screen
                cv2.imwrite(f'{screen_name}_mockup.png', screen)
            
            print("UI mockups generated")
            return True
        except Exception as e:
            print(f"Could not generate UI mockups: {e}")
            return False
    
    def export_ui_design(self):
        """Export the UI design to a JSON file"""
        ui_data = {
            "screens": self.screens,
            "command_list": self.get_command_list()
        }
        
        with open('ui_design.json', 'w') as f:
            json.dump(ui_data, f, indent=2)
        
        print("UI design exported to ui_design.json")

def run_system_design():
    """Run the system design phase"""
    print("Starting System Design...")
    
    # Create system architecture
    architecture = SystemArchitecture()
    architecture.generate_architecture_diagram()
    architecture.export_architecture()
    
    # Create UI prototype
    ui = UIPrototype()
    ui.generate_ui_mockup()
    ui.export_ui_design()
    
    print("System Design completed")
    return architecture, ui


# ============================================================
# 3. IMPLEMENTATION SIMULATION
# ============================================================

class SpeechRecognitionSimulator:
    """Simulates speech recognition for the wheelchair control system"""
    def __init__(self, command_list=None, noise_level="low"):
        self.noise_levels = {
            "low": 0.95,
            "medium": 0.85,
            "high": 0.75
        }
        self.accuracy = self.noise_levels.get(noise_level, 0.9)
        
        # Default command list if none provided
        self.command_list = command_list or [
            "go forward", "go backward", "stop", "turn left", "turn right", 
            "speed up", "slow down", "emergency stop", "open settings",
            "save settings", "exit", "help"
        ]
        
        # Create confusion matrix for commands
        self.command_confusion = {}
        for cmd in self.command_list:
            similar_commands = self._find_similar_commands(cmd)
            self.command_confusion[cmd] = similar_commands
        
        # Command queue for async operation
        self.command_queue = queue.Queue()
        
        # Create a simulated model
        self.model = self._create_simulated_model()
        
        # Statistics
        self.stats = {
            "commands_processed": 0,
            "commands_recognized_correctly": 0,
            "processing_times": []
        }
    
    def _create_simulated_model(self):
        """Create a simulated TensorFlow model for speech recognition"""
        # Create a simple model architecture
        inputs = keras.Input(shape=(13,))
        x = keras.layers.Dense(64, activation="relu")(inputs)
        x = keras.layers.Dense(32, activation="relu")(x)
        outputs = keras.layers.Dense(len(self.command_list), activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _find_similar_commands(self, command):
        """Find commands that could be confused with the given command"""
        similar = {}
        for cmd in self.command_list:
            if cmd != command:
                # Simple similarity: commands that share first word or have similar length
                if cmd.split()[0] == command.split()[0] or abs(len(cmd) - len(command)) < 3:
                    # Assign a confusion probability, higher for more similar commands
                    similarity = 0.1 + 0.1 * (1 - abs(len(cmd) - len(command)) / max(len(cmd), len(command)))
                    similar[cmd] = similarity
        
        # Normalize to ensure sum of probabilities doesn't exceed 1-accuracy
        total = sum(similar.values())
        if total > 0:
            error_budget = 1 - self.accuracy
            normalize_factor = error_budget / total
            similar = {k: v * normalize_factor for k, v in similar.items()}
        
        return similar
    
    def recognize_command(self, audio_input):
        """Simulate recognition of a command from audio input"""
        start_time = time.time()
        
        # Simulate processing delay
        time.sleep(0.2 + random.random() * 0.3)  # 200-500ms delay
        
        self.stats["commands_processed"] += 1
        
        # If the command is in our list, simulate recognition with accuracy probability
        if audio_input in self.command_list:
            if random.random() < self.accuracy:
                recognized = audio_input
                self.stats["commands_recognized_correctly"] += 1
            else:
                # Simulate misrecognition based on confusion matrix
                confusions = self.command_confusion[audio_input]
                if confusions and random.random() < 0.8:  # 80% chance to confuse with similar command
                    confused_commands = list(confusions.keys())
                    confused_probs = list(confusions.values())
                    total_prob = sum(confused_probs)
                    if total_prob < 1e-10:  # Guard against division by zero
                        recognized = random.choice(self.command_list)
                    else:
                        normalized_probs = [p/total_prob for p in confused_probs]
                        recognized = np.random.choice(confused_commands, p=normalized_probs)
                else:
                    # Random misrecognition
                    recognized = random.choice([c for c in self.command_list if c != audio_input])
        else:
            # If not in our list, simulate a random command or None
            if random.random() < 0.5:
                recognized = random.choice(self.command_list)
            else:
                recognized = None
        
        end_time = time.time()
        processing_time = end_time - start_time
        self.stats["processing_times"].append(processing_time)
        
        return {
            "input": audio_input,
            "recognized": recognized,
            "confidence": random.uniform(0.6, 0.99) if recognized else random.uniform(0.1, 0.5),
            "processing_time": processing_time
        }
    
    def start_listening(self):
        """Start listening for commands in a separate thread"""
        def listener_thread():
            while True:
                # In a real system, this would be audio processing
                # Here we just wait for commands added to the queue
                try:
                    command = self.command_queue.get(timeout=1.0)
                    if command == "STOP_LISTENING":
                        break
                    
                    result = self.recognize_command(command)
                    print(f"Recognized: {result['recognized']} (Confidence: {result['confidence']:.2f})")
                
                except queue.Empty:
                    continue
        
        self.listener = threading.Thread(target=listener_thread)
        self.listener.daemon = True
        self.listener.start()
        print("Speech recognition system is now listening...")
    
    def stop_listening(self):
        """Stop the listening thread"""
        self.command_queue.put("STOP_LISTENING")
        self.listener.join()
        print("Speech recognition system stopped")
    
    def process_command(self, command):
        """Process a voice command"""
        self.command_queue.put(command)
    
    def get_recognition_statistics(self):
        """Get statistics about the speech recognition system"""
        avg_time = sum(self.stats["processing_times"]) / max(len(self.stats["processing_times"]), 1)
        accuracy = self.stats["commands_recognized_correctly"] / max(self.stats["commands_processed"], 1)
        
        return {
            "total_commands": self.stats["commands_processed"],
            "correctly_recognized": self.stats["commands_recognized_correctly"],
            "accuracy": accuracy,
            "average_processing_time": avg_time
        }


class SensorSystemSimulator:
    """Simulates the sensor system for obstacle detection"""
    def __init__(self, environment_complexity="medium"):
        self.complexity = environment_complexity
        self.obstacle_densities = {
            "low": 0.1,        # 10% chance of obstacle in sensing range
            "medium": 0.25,    # 25% chance
            "high": 0.4        # 40% chance
        }
        self.obstacle_density = self.obstacle_densities.get(environment_complexity, 0.25)
        
        # Sensor ranges in meters
        self.sensor_ranges = {
            "ultrasonic_front": 3.0,
            "ultrasonic_rear": 2.0,
            "infrared_left": 1.5,
            "infrared_right": 1.5,
            "camera": 5.0  # For object recognition
        }
        
        # Sensor accuracies (probability of correct detection)
        self.sensor_accuracies = {
            "ultrasonic_front": 0.95,
            "ultrasonic_rear": 0.92,
            "infrared_left": 0.9,
            "infrared_right": 0.9,
            "camera": 0.85
        }
        
        # Current sensor data
        self.current_readings = {sensor: None for sensor in self.sensor_ranges}
        
        # Simulated environment with obstacles
        self.environment = self._generate_environment()
        
        # Start the sensor simulation
        self.running = False
        self.sensor_thread = None
    
    def _generate_environment(self):
        """Generate a simulated environment with obstacles"""
        # Create a 10x10 meter area around the wheelchair
        environment = {
            "obstacles": [],
            "dimensions": (10, 10),
            "wheelchair_position": (5, 5),  # Center of the environment
            "wheelchair_orientation": 0     # 0 degrees = facing "north"
        }
        
        # Add random obstacles
        num_obstacles = int(self.obstacle_density * 20)  # Up to 20 obstacles
        for _ in range(num_obstacles):
            obstacle = {
                "position": (random.uniform(0, 10), random.uniform(0, 10)),
                "size": random.uniform(0.3, 1.0),  # Obstacle diameter in meters
                "type": random.choice(["static", "static", "static", "moving"])  # Mostly static obstacles
            }
            environment["obstacles"].append(obstacle)
        
        return environment
    
    def update_wheelchair_position(self, x, y, orientation):
        """Update the wheelchair's position in the environment"""
        self.environment["wheelchair_position"] = (x, y)
        self.environment["wheelchair_orientation"] = orientation
    
    def _get_sensor_reading(self, sensor_name):
        """Get a simulated reading from the specified sensor"""
        # Extract wheelchair position and orientation
        wx, wy = self.environment["wheelchair_position"]
        orientation = self.environment["wheelchair_orientation"]
        
        # Determine sensor direction relative to wheelchair orientation
        sensor_directions = {
            "ultrasonic_front": orientation,
            "ultrasonic_rear": (orientation + 180) % 360,
            "infrared_left": (orientation + 270) % 360,
            "infrared_right": (orientation + 90) % 360,
            "camera": orientation  # Camera faces forward
        }
        
        sensor_direction = sensor_directions[sensor_name]
        sensor_range = self.sensor_ranges[sensor_name]
        
        # Check for obstacles in the sensor's field of view
        nearest_obstacle_distance = float('inf')
        nearest_obstacle = None
        
        for obstacle in self.environment["obstacles"]:
            ox, oy = obstacle["position"]
            obstacle_size = obstacle["size"]
            
            # Calculate distance to obstacle center
            distance = ((ox - wx) ** 2 + (oy - wy) ** 2) ** 0.5
            
            # Adjust for obstacle size (edge of obstacle)
            distance -= obstacle_size / 2
            
            # Skip if beyond sensor range
            if distance > sensor_range:
                continue
            
            # Calculate angle to obstacle
            angle = math.degrees(math.atan2(oy - wy, ox - wx)) % 360
            
            # Check if obstacle is in sensor's field of view
            angle_diff = min((angle - sensor_direction) % 360, (sensor_direction - angle) % 360)
            
            # Different sensors have different fields of view
            field_of_view = {
                "ultrasonic_front": 30,
                "ultrasonic_rear": 30,
                "infrared_left": 15,
                "infrared_right": 15,
                "camera": 60
            }
            
            if angle_diff <= field_of_view[sensor_name] / 2:
                if distance < nearest_obstacle_distance:
                    nearest_obstacle_distance = distance
                    nearest_obstacle = obstacle
        
        # Simulate sensor accuracy
        if nearest_obstacle and random.random() < self.sensor_accuracies[sensor_name]:
            # Add some noise to the distance reading
            noise = random.uniform(-0.05, 0.05) * nearest_obstacle_distance
            reading = max(0, nearest_obstacle_distance + noise)
            
            return {
                "distance": reading,
                "detected": True,
                "obstacle_type": nearest_obstacle["type"] if sensor_name == "camera" else None
            }
        elif nearest_obstacle:
            # Failed to detect an existing obstacle (false negative)
            return {
                "distance": None,
                "detected": False,
                "obstacle_type": None
            }
        else:
            # No obstacle in range
            # Small chance of false positive
            if random.random() < 0.02:  # 2% chance of false positive
                return {
                    "distance": random.uniform(0, sensor_range),
                    "detected": True,
                    "obstacle_type": random.choice(["static", "moving"]) if sensor_name == "camera" else None
                }
            else:
                return {
                    "distance": None,
                    "detected": False,
                    "obstacle_type": None
                }
    
    def get_all_readings(self):
        """Get readings from all sensors"""
        readings = {}
        for sensor in self.sensor_ranges.keys():
            readings[sensor] = self._get_sensor_reading(sensor)
        
        return readings
    
    def start_sensing(self, update_frequency=10):
        """Start the sensor system with the given update frequency in Hz"""
        if self.running:
            return
        
        self.running = True
        
        def sensor_thread_func():
            while self.running:
                # Update sensor readings
                self.current_readings = self.get_all_readings()
                
                # Sleep for the specified interval
                time.sleep(1.0 / update_frequency)
        
        self.sensor_thread = threading.Thread(target=sensor_thread_func)
        self.sensor_thread.daemon = True
        self.sensor_thread.start()
        print("Sensor system started")
    
    def stop_sensing(self):
        """Stop the sensor system"""
        self.running = False
        if self.sensor_thread:
            self.sensor_thread.join()
        print("Sensor system stopped")
    
    def get_obstacle_map(self, resolution=20):
        """Create a simple 2D map of obstacles around the wheelchair"""
        # Create an empty grid
        grid_size = resolution + 1
        grid = np.zeros((grid_size, grid_size))
        
        # Map dimensions
        map_size = 10.0  # 10x10 meter area
        cell_size = map_size / resolution
        
        # Place obstacles on the grid
        for obstacle in self.environment["obstacles"]:
            ox, oy = obstacle["position"]
            size = obstacle["size"]
            
            # Convert to grid coordinates
            grid_x = int(ox / cell_size)
            grid_y = int(oy / cell_size)
            
            # Ensure within bounds
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                grid[grid_y, grid_x] = 1
                
                # Mark adjacent cells for larger obstacles
                if size > cell_size:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = grid_x + dx, grid_y + dy
                            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                grid[ny, nx] = 1
        
        # Mark wheelchair position
        wx, wy = self.environment["wheelchair_position"]
        grid_wx = int(wx / cell_size)
        grid_wy = int(wy / cell_size)
        
        if 0 <= grid_wx < grid_size and 0 <= grid_wy < grid_size:
            grid[grid_wy, grid_wx] = 2  # Use 2 to represent wheelchair
        
        return grid


class NavigationSystem:
    """Simulates the navigation system for the wheelchair"""
    def __init__(self, sensor_system):
        self.sensor_system = sensor_system
        self.current_command = None
        self.status = "idle"
        self.current_speed = 0.0
        self.max_speed = 2.0  # meters per second
        self.current_direction = 0.0  # degrees, 0 = forward
        
        # Safety parameters
        self.min_safe_distance = 0.5  # meters
        self.slow_down_distance = 1.5  # meters
        
        # Navigation log
        self.log = deque(maxlen=100)
        
        # Path planning grid
        self.grid_resolution = 20
        self.path = []
    
    def process_command(self, command, confidence):
        """Process a voice command for navigation"""
        self.current_command = command
        self.log.append({
            "timestamp": time.time(),
            "command": command,
            "confidence": confidence
        })
        
        # Basic command parsing
        if command == "go forward":
            self.status = "moving"
            self.current_direction = 0.0
            self._accelerate()
        elif command == "go backward":
            self.status = "moving"
            self.current_direction = 180.0
            self._accelerate()
        elif command == "turn left":
            self.status = "turning"
            self.current_direction = (self.current_direction - 90) % 360
            self.current_speed = 0.5
        elif command == "turn right":
            self.status = "turning"
            self.current_direction = (self.current_direction + 90) % 360
            self.current_speed = 0.5
        elif command == "stop":
            self.status = "stopping"
            self._decelerate_to_stop()
        elif command == "emergency stop":
            self.status = "emergency_stop"
            self._emergency_stop()
        elif command == "speed up":
            self._accelerate()
        elif command == "slow down":
            self._decelerate()
        else:
            return False
        
        return True
    
    def _accelerate(self):
        """Increase speed gradually"""
        self.current_speed = min(self.max_speed, self.current_speed + 0.5)
    
    def _decelerate(self):
        """Decrease speed gradually"""
        self.current_speed = max(0.0, self.current_speed - 0.5)
        if self.current_speed < 0.1:
            self.current_speed = 0.0
            self.status = "idle"
    
    def _decelerate_to_stop(self):
        """Decelerate until stopped"""
        self.current_speed = max(0.0, self.current_speed - 1.0)
        if self.current_speed < 0.1:
            self.current_speed = 0.0
            self.status = "idle"
    
    def _emergency_stop(self):
        """Immediate stop"""
        self.current_speed = 0.0
        self.status = "idle"
    
    def update_navigation(self):
        """Update navigation based on current commands and sensor data"""
        # Get current sensor readings
        readings = self.sensor_system.current_readings
        
        # Safety check - detect obstacles in path
        front_sensor = readings.get("ultrasonic_front", {})
        rear_sensor = readings.get("ultrasonic_rear", {})
        left_sensor = readings.get("infrared_left", {})
        right_sensor = readings.get("infrared_right", {})
        
        # Check for obstacles in the direction of travel
        if self.current_direction < 45 or self.current_direction > 315:
            # Moving forward
            if front_sensor.get("detected", False):
                distance = front_sensor.get("distance", float('inf'))
                if distance < self.min_safe_distance:
                    self._emergency_stop()
                    return "Emergency stop: obstacle detected"
                elif distance < self.slow_down_distance:
                    self._decelerate()
                    return "Slowing down: obstacle ahead"
        
        elif 135 < self.current_direction < 225:
            # Moving backward
            if rear_sensor.get("detected", False):
                distance = rear_sensor.get("distance", float('inf'))
                if distance < self.min_safe_distance:
                    self._emergency_stop()
                    return "Emergency stop: obstacle detected behind"
                elif distance < self.slow_down_distance:
                    self._decelerate()
                    return "Slowing down: obstacle behind"
        
        elif 45 <= self.current_direction <= 135:
            # Turning/moving right
            if right_sensor.get("detected", False):
                distance = right_sensor.get("distance", float('inf'))
                if distance < self.min_safe_distance:
                    self._emergency_stop()
                    return "Emergency stop: obstacle detected to the right"
        
        elif 225 <= self.current_direction <= 315:
            # Turning/moving left
            if left_sensor.get("detected", False):
                distance = left_sensor.get("distance", float('inf'))
                if distance < self.min_safe_distance:
                    self._emergency_stop()
                    return "Emergency stop: obstacle detected to the left"
        
        # Normal operation
        return f"Navigating: Speed {self.current_speed:.1f} m/s, Direction {self.current_direction:.0f}°"
    
    def get_navigation_status(self):
        """Get the current navigation status"""
        return {
            "status": self.status,
            "speed": self.current_speed,
            "direction": self.current_direction,
            "command": self.current_command
        }
    
    def plan_path(self, destination_x, destination_y):
        """Plan a path to the destination using A* algorithm"""
        # Get the current obstacle map
        grid = self.sensor_system.get_obstacle_map(self.grid_resolution)
        
        # Extract wheelchair position
        wx, wy = self.sensor_system.environment["wheelchair_position"]
        
        # Convert to grid coordinates
        cell_size = 10.0 / self.grid_resolution
        start_x = int(wx / cell_size)
        start_y = int(wy / cell_size)
        end_x = int(destination_x / cell_size)
        end_y = int(destination_y / cell_size)
        
        # Ensure coordinates are within grid bounds
        grid_size = self.grid_resolution + 1
        start_x = max(0, min(start_x, grid_size - 1))
        start_y = max(0, min(start_y, grid_size - 1))
        end_x = max(0, min(end_x, grid_size - 1))
        end_y = max(0, min(end_y, grid_size - 1))
        
        # Run A* algorithm
        self.path = self._a_star_search(grid, (start_y, start_x), (end_y, end_x))
        
        # Convert path back to world coordinates
        world_path = []
        for y, x in self.path:
            world_x = (x + 0.5) * cell_size
            world_y = (y + 0.5) * cell_size
            world_path.append((world_x, world_y))
        
        return world_path
    
    def _a_star_search(self, grid, start, end):
        """A* search algorithm for path finding"""
        # Heuristic function (Manhattan distance)
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # A* algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        open_set_hash = {start}
        
        grid_size = grid.shape[0]
        
        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            # Explore neighbors
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-connected grid
                neighbor = (current[0] + dy, current[1] + dx)
                
                # Check if valid position
                if (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size and
                        grid[neighbor] != 1):  # Not an obstacle
                    
                    tentative_g_score = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                        
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
        
        # No path found
        return []


class WheelchairSimulator:
    """Main simulator for the speech-controlled wheelchair system"""
    def __init__(self, environment_complexity="medium", noise_level="low"):
        print("Initializing Speech-Controlled Wheelchair Simulator...")
        
        # Load UI design for command list
        try:
            with open('ui_design.json', 'r') as f:
                ui_data = json.load(f)
                command_list = ui_data.get("command_list", None)
        except (FileNotFoundError, json.JSONDecodeError):
            command_list = None
            print("UI design file not found or invalid, using default command list")
        
        # Initialize subsystems
        self.speech_system = SpeechRecognitionSimulator(command_list, noise_level)
        self.sensor_system = SensorSystemSimulator(environment_complexity)
        self.navigation_system = NavigationSystem(self.sensor_system)
        
        # Simulation state
        self.running = False
        self.simulation_thread = None
        self.simulation_speed = 1.0  # Real-time by default
        
        # Wheelchair state
        self.position = (5.0, 5.0)  # meters
        self.orientation = 0.0  # degrees
        self.battery_level = 100.0  # percent
        
        # Performance metrics
        self.metrics = {
            "commands_processed": 0,
            "successful_commands": 0,
            "emergency_stops": 0,
            "obstacles_avoided": 0,
            "distance_traveled": 0.0,
            "battery_consumption": 0.0
        }
        
        # Simulation log
        self.log = []
        
        print("Wheelchair Simulator initialized successfully")
    
    def start_simulation(self):
        """Start the wheelchair simulation"""
        if self.running:
            return False
        
        self.running = True
        
        # Start subsystems
        self.speech_system.start_listening()
        self.sensor_system.start_sensing()
        
        # Start simulation loop
        def simulation_loop():
            last_update = time.time()
            
            while self.running:
                # Calculate time delta
                now = time.time()
                dt = (now - last_update) * self.simulation_speed
                last_update = now
                
                # Update navigation based on commands and sensor data
                nav_status = self.navigation_system.update_navigation()
                
                # Update wheelchair position based on navigation
                self._update_position(dt)
                
                # Update battery level
                self._update_battery(dt)
                
                # Log state
                self._log_state(nav_status)
                
                # Sleep to control simulation speed
                time.sleep(0.1 / self.simulation_speed)
        
        self.simulation_thread = threading.Thread(target=simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        print("Wheelchair simulation started")
        return True
    
    def stop_simulation(self):
        """Stop the simulation"""
        if not self.running:
            return False
        
        self.running = False
        
        # Stop subsystems
        self.speech_system.stop_listening()
        self.sensor_system.stop_sensing()
        
        # Wait for simulation thread to end
        if self.simulation_thread:
            self.simulation_thread.join()
        
        print("Wheelchair simulation stopped")
        return True
    
    def process_voice_command(self, command_text):
        """Process a voice command through the speech recognition system"""
        # Log the raw command
        self.log.append({
            "timestamp": time.time(),
            "event": "voice_input",
            "data": command_text
        })
        
        # Process through speech recognition
        result = self.speech_system.recognize_command(command_text)
        
        # Log recognition result
        self.log.append({
            "timestamp": time.time(),
            "event": "speech_recognition",
            "data": result
        })
        
        self.metrics["commands_processed"] += 1
        
        # If command recognized, pass to navigation system
        if result["recognized"]:
            success = self.navigation_system.process_command(
                result["recognized"], result["confidence"])
            
            if success:
                self.metrics["successful_commands"] += 1
                
                # Log navigation command
                self.log.append({
                    "timestamp": time.time(),
                    "event": "navigation_command",
                    "data": {
                        "command": result["recognized"],
                        "confidence": result["confidence"]
                    }
                })
        
        return result
    
    def _update_position(self, dt):
        """Update wheelchair position based on navigation commands"""
        nav_status = self.navigation_system.get_navigation_status()
        speed = nav_status["speed"]
        direction_rad = math.radians(nav_status["direction"])
        
        # Update position
        if speed > 0:
            dx = speed * dt * math.cos(direction_rad)
            dy = speed * dt * math.sin(direction_rad)
            
            # Check for boundary constraints (keep wheelchair in the 10x10 area)
            new_x = max(0, min(10, self.position[0] + dx))
            new_y = max(0, min(10, self.position[1] + dy))
            
            # Calculate actual distance traveled
            distance = ((new_x - self.position[0])**2 + (new_y - self.position[1])**2)**0.5
            self.metrics["distance_traveled"] += distance
            
            self.position = (new_x, new_y)
        
        # Update orientation (simplified - just use the direction)
        self.orientation = nav_status["direction"]
        
        # Update sensor system with new position
        self.sensor_system.update_wheelchair_position(
            self.position[0], self.position[1], self.orientation)
    
    def _update_battery(self, dt):
        """Update battery level based on usage"""
        nav_status = self.navigation_system.get_navigation_status()
        
        # Battery consumption depends on speed and activity
        base_consumption = 0.1  # % per second at idle
        
        if nav_status["status"] == "idle":
            consumption = base_consumption
        elif nav_status["status"] == "emergency_stop":
            consumption = base_consumption * 2
        else:
            # Consumption increases with speed
            speed_factor = nav_status["speed"] / self.navigation_system.max_speed
            consumption = base_consumption + (0.4 * speed_factor)  # Up to 0.5% per second at max speed
        
        # Apply time delta
        consumption_this_update = consumption * dt
        self.battery_level = max(0, self.battery_level - consumption_this_update)
        self.metrics["battery_consumption"] += consumption_this_update
        
        # Handle battery depletion
        if self.battery_level < 1.0 and nav_status["status"] != "idle":
            self.navigation_system._emergency_stop()
            self.log.append({
                "timestamp": time.time(),
                "event": "battery_critical",
                "data": {"battery_level": self.battery_level}
            })
    
    def _log_state(self, nav_status):
        """Log the current state of the wheelchair"""
        if len(self.log) > 0 and time.time() - self.log[-1]["timestamp"] < 1.0:
            # Don't log too frequently
            return
        
        self.log.append({
            "timestamp": time.time(),
            "event": "state_update",
            "data": {
                "position": self.position,
                "orientation": self.orientation,
                "battery_level": self.battery_level,
                "navigation": self.navigation_system.get_navigation_status(),
                "sensors": {k: v for k, v in self.sensor_system.current_readings.items() if v}
            }
        })
    
    def get_simulation_status(self):
        """Get the current status of the wheelchair simulation"""
        return {
            "position": self.position,
            "orientation": self.orientation,
            "battery_level": self.battery_level,
            "navigation": self.navigation_system.get_navigation_status(),
            "metrics": self.metrics,
            "running": self.running
        }
    
    def get_obstacle_map(self):
        """Get the current obstacle map"""
        return self.sensor_system.get_obstacle_map()
    
    def set_simulation_speed(self, speed):
        """Set the simulation speed (1.0 = real-time)"""
        if speed > 0:
            self.simulation_speed = speed
            return True
        return False
    
    def navigate_to_position(self, x, y):
        """Plan and execute navigation to a target position"""
        # Check if position is valid
        if not (0 <= x <= 10 and 0 <= y <= 10):
            return False, "Invalid target position"
        
        # Plan path
        path = self.navigation_system.plan_path(x, y)
        
        if not path:
            return False, "No valid path found"
        
        # Log the navigation plan
        self.log.append({
            "timestamp": time.time(),
            "event": "path_planning",
            "data": {
                "start": self.position,
                "target": (x, y),
                "path": path
            }
        })
        
        # In a real system, this would trigger autonomous navigation
        # For the simulator, we'll just log the plan
        
        return True, path
    
    def get_log_events(self, event_type=None, limit=50):
        """Get log events, optionally filtered by type"""
        if event_type:
            filtered_log = [entry for entry in self.log if entry["event"] == event_type]
        else:
            filtered_log = self.log
        
        # Return most recent entries first
        return list(reversed(filtered_log[-limit:]))
    
    def export_logs(self, filename="wheelchair_simulation_log.json"):
        """Export logs to a JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                "log": self.log,
                "metrics": self.metrics,
                "simulation_info": {
                    "time": time.time(),
                    "battery_level": self.battery_level,
                    "distance_traveled": self.metrics["distance_traveled"]
                }
            }, f, indent=2)
        
        return filename
    
    # Add to your Python simulator
import socket
import json

class UnityInterface:
    def __init__(self, simulator, host='127.0.0.1', port=5555):
        self.simulator = simulator
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.socket.listen(1)
        self.client = None
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._communication_loop)
        self.thread.daemon = True
        self.thread.start()
        print(f"Unity interface started on port 5555. Waiting for connection...")
    
    def _communication_loop(self):
        self.socket.settimeout(1.0)
        
        # Wait for Unity to connect
        while self.running and not self.client:
            try:
                self.client, address = self.socket.accept()
                print(f"Unity connected from {address}")
            except socket.timeout:
                continue
        
        if self.client:
            self.client.settimeout(0.1)
            
            # Communication loop
            while self.running:
                try:
                    # Send state to Unity
                    state = self._get_state_for_unity()
                    self.client.sendall((json.dumps(state) + '\n').encode('utf-8'))
                    
                    # Receive commands from Unity
                    try:
                        data = self.client.recv(4096)
                        if data:
                            self._process_unity_command(data)
                    except socket.timeout:
                        pass
                    
                    time.sleep(0.05)  # Update at 20Hz
                    
                except Exception as e:
                    print(f"Communication error: {e}")
                    break
    
    def _get_state_for_unity(self):
        """Get the current state to send to Unity"""
        status = self.simulator.get_simulation_status()
        obstacles = self.simulator.sensor_system.environment["obstacles"]
        
        return {
            "wheelchair": {
                "position": status["position"],
                "orientation": status["orientation"],
                "battery": status["battery_level"],
                "speed": status["navigation"]["speed"],
                "status": status["navigation"]["status"]
            },
            "obstacles": obstacles,
            "sensors": {k: v for k, v in self.simulator.sensor_system.current_readings.items() if v and v.get("detected", False)}
        }
    
    def _process_unity_command(self, data):
        """Process commands from Unity"""
        try:
            command = json.loads(data.decode('utf-8'))
            
            if command["type"] == "voice":
                self.simulator.process_voice_command(command["text"])
            elif command["type"] == "goto":
                self.simulator.navigate_to_position(command["x"], command["y"])
            # Add more command types as needed
            
        except json.JSONDecodeError:
            print("Invalid command from Unity")
    
    def stop(self):
        self.running = False
        if self.client:
            self.client.close()
        self.socket.close()


def main():
    """Main function to run the wheelchair simulator with Unity interface only."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Speech-Controlled Wheelchair Simulator')
    parser.add_argument('--environment', choices=['low', 'medium', 'high'], default='medium',
                        help='Environmental complexity (obstacle density)')
    parser.add_argument('--noise', choices=['low', 'medium', 'high'], default='low',
                        help='Speech recognition noise level')
    parser.add_argument('--sim-speed', type=float, default=1.0,
                        help='Simulation speed multiplier (1.0 = real-time)')
    args = parser.parse_args()
    
    # Create simulator
    simulator = WheelchairSimulator(
        environment_complexity=args.environment,
        noise_level=args.noise
    )
    
    # Set simulation speed
    simulator.set_simulation_speed(args.sim_speed)
    
    # Start simulation
    simulator.start_simulation()
    
    # Start Unity interface
    unity_interface = UnityInterface(simulator)
    unity_interface.start()
    print("Unity interface started. Run Unity project to visualize.")
    
    try:
        while True:
            pass  # Remove CLI interactions and let Unity handle commands
    
    except KeyboardInterrupt:
        print("\nExiting simulator...")
    
    finally:
        # Stop simulation
        simulator.stop_simulation()



if __name__ == "__main__":
    main()


       
