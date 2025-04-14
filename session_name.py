import os
import json
from datetime import datetime
from typing import List, Dict
import sys

class Message:
    def __init__(self, sender: str, content: str, timestamp: datetime = None):
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(
            sender=data["sender"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

class Session:
    def __init__(self, session_id: str = None, topic: str = "General", mode: str = None):
      self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
      self.messages: List[Message] = []
      self.topic = topic

    # Add this line
      self.session_dir = os.path.join(os.path.expanduser("~"), ".security_advisor", "sessions")

    # Auto-detect mode if not passed
      if mode:
        self.mode = mode
      elif "security_mode" in sys.argv[0]:
        self.mode = "security"
      else:
        self.mode = "normal"

  # Use security_advisor for consistency

    def add_message(self, sender: str, content: str) -> None:
        self.messages.append(Message(sender, content))

    def save(self) -> None:
        os.makedirs(self.session_dir, exist_ok=True)
        session_data = {
            "session_id": self.session_id,
            "topic": self.topic,
            "mode": self.mode,
            "messages": [msg.to_dict() for msg in self.messages],
            "last_updated": datetime.now().isoformat()
        }
        with open(os.path.join(self.session_dir, f"{self.session_id}.json"), "w") as f:
            json.dump(session_data, f, indent=2)

    @classmethod
    def load(cls, session_id: str) -> 'Session':
        session = cls(session_id)  # Default topic and mode will be overwritten
        session_file = os.path.join(cls.session_dir, f"{session_id}.json")
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                data = json.load(f)
            session.session_id = data["session_id"]
            session.topic = data.get("topic", "General")
            session.mode = data.get("mode", "normal")
            session.messages = [Message.from_dict(msg) for msg in data["messages"]]
        return session

    @classmethod
    def list_sessions(cls) -> List[Dict]:
        session_dir = os.path.join(os.path.expanduser("~"), ".security_advisor", "sessions")
        os.makedirs(session_dir, exist_ok=True)
        sessions = []
        for file in os.listdir(session_dir):
            if file.endswith(".json"):
                with open(os.path.join(session_dir, file), "r") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "topic": data.get("topic", "General"),
                    "mode": data.get("mode", "normal"),
                    "last_updated": datetime.fromisoformat(data["last_updated"]),
                    "message_count": len(data["messages"])
                })
        return sorted(sessions, key=lambda x: x["last_updated"], reverse=True)

    @classmethod
    def display_history(cls) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()

        sessions = cls.list_sessions()
        if not sessions:
            console.print("[yellow]No saved sessions found.[/yellow]")
            return

        table = Table(title="Session History", box=box.SIMPLE)
        table.add_column("Session ID", style="cyan")
        table.add_column("Topic", style="green")
        table.add_column("Mode", style="magenta")
        table.add_column("Last Updated", style="white")
        table.add_column("Messages", style="yellow")

        for session in sessions:
            table.add_row(
                session["session_id"],
                session["topic"],
                session["mode"].upper(),
                session["last_updated"].strftime("%Y-%m-%d %H:%M:%S"),
                str(session["message_count"])
            )

        console.print(table)