import os
import json
import datetime
from dotenv import load_dotenv
from pathlib import Path

# Ensure data directory exists
os.makedirs("Data", exist_ok=True)

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the chat system."""
    
    def __init__(self):
        # Load from environment variables
        self.username = os.getenv("Username", "User")
        self.assistant_name = os.getenv("Assistantname", "AIAssistant")
        self.groq_api_key = os.getenv("GroqAPIKey")
        self.runpod_endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "")
        self.runpod_api_key = os.getenv("RUNPOD_API_KEY", "")
        
        # Models
        self.default_model = "llama3-70b-8192"
        self.alt_model = "localmodels/Llama-2-7B-Chat-ggml"
        self.model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
        
        # General settings
        self.history_file = "Data/ChatLog.json"
        self.temperature = 0.7
        self.max_tokens = 1024
        self.top_p = 1.0
        self.top_k = 50
        self.repetition_penalty = 1.2
        
        # Check if Groq API key is available
        if not self.groq_api_key:
            print("⚠️ Warning: GroqAPIKey not found in environment variables.")
            print("   Some features may be limited. Please set up your API keys.")

    def current_time_info(self):
        """Get formatted current date and time information."""
        now = datetime.datetime.now()
        
        return {
            "day": now.strftime("%A"),
            "date": now.strftime("%d"),
            "month": now.strftime("%B"),
            "year": now.strftime("%Y"),
            "hour": now.strftime("%H"),
            "minute": now.strftime("%M"),
            "second": now.strftime("%S"),
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_realtime_info(self):
        """Get formatted real-time information as a string."""
        time_info = self.current_time_info()
        
        data = f"Real-time information:\n"
        data += f"Day: {time_info['day']}\n"
        data += f"Date: {time_info['date']}\n"
        data += f"Month: {time_info['month']}\n"
        data += f"Year: {time_info['year']}\n"
        data += f"Time: {time_info['hour']}:{time_info['minute']}:{time_info['second']}\n"
        
        return data


def load_chat_history(filename="Data/ChatLog.json"):
    """Load chat history from file with error handling."""
    try:
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r") as f:
                return json.load(f)
        return []
    except json.JSONDecodeError:
        return []
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []


def save_chat_history(messages, filename="Data/ChatLog.json"):
    """Save chat history to file with error handling."""
    try:
        with open(filename, "w") as f:
            json.dump(messages, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving chat history: {e}")
        return False


def format_answer(answer):
    """Format the AI response by removing empty lines."""
    if not answer:
        return ""
    
    lines = answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)