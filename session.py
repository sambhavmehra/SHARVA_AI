import os
import json
from datetime import datetime
from typing import List, Dict

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
    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.messages: List[Message] = []
        self.session_dir = os.path.join(os.path.expanduser("~"), ".security_advisor", "sessions")

    def add_message(self, sender: str, content: str) -> None:
        self.messages.append(Message(sender, content))

    def save(self) -> None:
        os.makedirs(self.session_dir, exist_ok=True)
        session_data = {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "last_updated": datetime.now().isoformat()
        }
        with open(os.path.join(self.session_dir, f"{self.session_id}.json"), "w") as f:
            json.dump(session_data, f, indent=2)

    @classmethod
    def load(cls, session_id: str) -> 'Session':
        session = cls(session_id)
        session_file = os.path.join(session.session_dir, f"{session_id}.json")
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                data = json.load(f)
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
                    "last_updated": datetime.fromisoformat(data["last_updated"]),
                    "message_count": len(data["messages"])
                })
        return sorted(sessions, key=lambda x: x["last_updated"], reverse=True)