from logging import config
import os
import sys
import json
import shutil
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich import box
from rich.theme import Theme
from rich.style import Style
from rich.text import Text
from rich.layout import Layout
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import clear as pt_clear
from prompt_toolkit.key_binding import KeyBindings
from config import Config

from realtime_engine import RealTimeEngine
from chat_engine import ChatEngine

# Define custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "user": "bold green",
    "assistant": "bold blue",
    "highlight": "bold magenta",
    "command": "italic cyan",
    "grok": "bold purple",  # New style for grok analysis
    "realtime": "bold cyan", # New style for realtime data
})

console = Console(theme=custom_theme)

class Message:
    """Class to represent a chat message."""
    def __init__(self, sender: str, content: str, timestamp: datetime = None, metadata: Dict = None):
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}  # Add metadata for tracking analysis type
       
    def to_dict(self) -> Dict:
        """Convert message to dictionary for serialization."""
        return {
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Create message from dictionary."""
        message = cls(
            sender=data["sender"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        message.metadata = data.get("metadata", {})
        return message


class Session:
    """Class to manage chat sessions."""
    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.messages: List[Message] = []
        self.session_dir = os.path.join(os.path.expanduser("~"), ".normal_advisor", "sessions")
        self.topic = ""
    
    def add_message(self, sender: str, content: str, metadata: Dict = None) -> None:
        """Add a message to the session with optional metadata."""
        self.messages.append(Message(sender, content, metadata=metadata))
    
    def save(self) -> None:
        """Save the session to disk."""
        os.makedirs(self.session_dir, exist_ok=True)
        
        session_data = {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "topic": self.topic,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(os.path.join(self.session_dir, f"{self.session_id}.json"), "w") as f:
            json.dump(session_data, f, indent=2)
    
    @classmethod
    def load(cls, session_id: str) -> 'Session':
        """Load a session from disk."""
        session = cls(session_id)
        session_file = os.path.join(session.session_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            raise FileNotFoundError(f"Session file not found: {session_file}")
        
        with open(session_file, "r") as f:
            data = json.load(f)
        
        session.messages = [Message.from_dict(msg) for msg in data["messages"]]
        session.topic = data.get("topic", "")
        
        return session
    
    @classmethod
    def list_sessions(cls) -> List[Dict]:
        """List all saved sessions."""
        session_dir = os.path.join(os.path.expanduser("~"), ".normal_advisor", "sessions")
        os.makedirs(session_dir, exist_ok=True)
        
        sessions = []
        for file in os.listdir(session_dir):
            if file.endswith(".json"):
                try:
                    with open(os.path.join(session_dir, file), "r") as f:
                        data = json.load(f)
                    sessions.append({
                        "session_id": data["session_id"],
                        "topic": data.get("topic", "Untitled"),
                        "last_updated": datetime.fromisoformat(data["last_updated"]),
                        "message_count": len(data["messages"])
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    console.print(f"[error]Error reading session file {file}: {str(e)}[/error]")
        
        return sorted(sessions, key=lambda x: x["last_updated"], reverse=True)


class GrokAnalyzer:
    """Class to provide advanced analysis capabilities similar to 'grok'."""
    
    def __init__(self):
        self.patterns = {
            "code_analysis": re.compile(r"(analyze|review|explain)\s+(code|function|class|script|algorithm)", re.IGNORECASE),
            "data_analysis": re.compile(r"(analyze|examine|study)\s+(data|dataset|numbers|statistics)", re.IGNORECASE),
            "complex_problem": re.compile(r"(solve|calculate|compute|figure out)\s+.{10,}", re.IGNORECASE),
            "technical_question": re.compile(r"(how|what|why|when)\s+(does|is|are|can|should|would|could)\s+.{5,}\?", re.IGNORECASE),
            "conceptual_question": re.compile(r"(explain|describe|define|clarify|elaborate)\s+.{10,}", re.IGNORECASE)
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query to determine if it needs deep thinking (grok-like analysis)."""
        analysis = {
            "requires_grok": False,
            "analysis_type": [],
            "complexity": 0.0
        }
        
        # Check for pattern matches
        for pattern_type, pattern in self.patterns.items():
            if pattern.search(query):
                analysis["analysis_type"].append(pattern_type)
                analysis["complexity"] += 0.2  # Increase complexity for each matched pattern
        
        # Check for other complexity indicators
        if len(query.split()) > 20:
            analysis["complexity"] += 0.2  # Longer queries might be more complex
        
        if "?" in query and len(query) > 100:
            analysis["complexity"] += 0.1  # Detailed questions
            
        if any(keyword in query.lower() for keyword in ["complex", "difficult", "challenging", "advanced", "expert"]):
            analysis["complexity"] += 0.2  # Explicitly mentioned complexity
        
        # Determine if grok analysis is needed
        analysis["requires_grok"] = analysis["complexity"] > 0.3 or len(analysis["analysis_type"]) > 0
        
        return analysis


class RealTimeProcessor:
    """Enhanced class to process real-time queries and integrate with grok analysis."""
    
    def __init__(self, realtime_engine, chat_engine):
        self.realtime_engine = realtime_engine
        self.chat_engine = chat_engine
        self.grok_analyzer = GrokAnalyzer()
        
        # Define real-time query patterns
        self.realtime_patterns = [
            re.compile(r"(current|latest|recent|today's|now|right now|live)\s+(news|weather|stock|price|event|update)", re.IGNORECASE),
            re.compile(r"what('s| is) happening (now|today|right now|at the moment)", re.IGNORECASE),
            re.compile(r"(how is|what is|tell me about)\s+.{1,20}\s+(today|now|currently|at present)", re.IGNORECASE),
            re.compile(r"(track|monitor|check|get|fetch)\s+.{1,20}\s+(status|price|value|rate|score)", re.IGNORECASE)
        ]
    
    def is_realtime_query(self, query: str) -> bool:
        """Check if the query requires real-time data."""
        return any(pattern.search(query) for pattern in self.realtime_patterns)
    
    def process_query(self, query: str, mode: str = "normal") -> Tuple[str, Dict]:
        """Process a query with intelligent routing between real-time and grok analysis."""
        # Analyze if the query needs grok-like deep thinking
        grok_analysis = self.grok_analyzer.analyze_query(query)
        
        # Check if it's a real-time query
        is_realtime = self.is_realtime_query(query)
        
        metadata = {
            "analysis": {
                "is_realtime": is_realtime,
                "grok_analysis": grok_analysis
            },
            "processing": {
                "mode": mode,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Process based on analysis results
        if is_realtime and grok_analysis["requires_grok"]:
            # Handle complex real-time query that needs both real-time data and grok analysis
            console.print("[bold magenta]🧠 Deep Analysis of Real-Time Data Required[/bold magenta]")
            
            # Step 1: Get real-time data
            with console.status("[realtime]Fetching real-time data...[/realtime]"):
                realtime_data = self.realtime_engine.process_realtime_query(query, mode=mode)
            
            # Step 2: Apply grok-like analysis to the real-time data
            with console.status("[grok]Performing deep analysis...[/grok]"):
                analysis_prompt = f"Analyze this real-time information deeply and explain it thoroughly: {realtime_data}"
                deep_analysis = self.chat_engine.process_normal_query(analysis_prompt)
            
            # Step 3: Merge the results with clear section breaks
            response = f"## Real-Time Data Results\n\n{realtime_data}\n\n## Deep Analysis\n\n{deep_analysis}"
            
        elif is_realtime:
            # Handle regular real-time query
            console.print("[realtime]🛰️ Real-Time Query Processing[/realtime]")
            with console.status("[realtime]Fetching real-time data...[/realtime]"):
                response = self.realtime_engine.process_realtime_query(query, mode=mode)
            
        elif grok_analysis["requires_grok"]:
            # Handle query that needs grok-like analysis but not real-time data
            console.print(f"[grok]🧠 Deep Analysis Required ({', '.join(grok_analysis['analysis_type'])})[/grok]")
            with console.status("[grok]Performing deep analysis...[/grok]"):
                grok_prompt = f"Analyze this question thoroughly with deep thinking and detailed explanation: {query}"
                response = self.chat_engine.process_normal_query(grok_prompt)
            
        else:
            # Handle regular query
            console.print("[info]Processing standard query[/info]")
            with console.status("[success]Thinking...[/success]"):
                response = self.chat_engine.process_normal_query(query)
        
        return response, metadata


class NormalMode:
    def __init__(self, config):
        self.config = config
        
        # Import here to avoid circular imports
        from chat_engine import ChatEngine
        self.chat_engine = ChatEngine(config)
        
        self.realtime_engine = RealTimeEngine(self.chat_engine)
        
        # Initialize the enhanced real-time processor with grok capabilities
        self.query_processor = RealTimeProcessor(self.realtime_engine, self.chat_engine)
        
        # Create command completer with new grok command
        self.commands = [
            "quit", "exit", "clear", "help", "switch", "search",
            "save", "load", "sessions", "theme", "keyboard", "export", "history",
            "analyze", "grok"  # New commands for enhanced analysis
        ]
        self.completer = WordCompleter(self.commands)
        
        # Setup prompt session with history
        history_file = os.path.join(os.path.expanduser("~"), ".normal_advisor", "history")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        self.prompt_session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer
        )
        
        # Set up keybindings
        self.kb = KeyBindings()
        
        @self.kb.add('c-l')
        def clear_screen(event):
            self.clear_screen()
        
        @self.kb.add('c-s')
        def save_session(event):
            self.save_current_session()
            console.print("[info]Session saved[/info]")
        
        @self.kb.add('c-g')  # New keybinding for grok mode
        def toggle_grok(event):
            console.print("[grok]🧠 Grok analysis mode toggled[/grok]")
        
        # Initialize session
        self.current_session = Session()

        # Default theme settings
        self.theme_name = "default"
        self.available_themes = {
            "default": {
                "user_color": "bold green",
                "assistant_color": "bold blue",
                "banner_color": "blue",
                "highlight_color": "cyan",
                "grok_color": "bold purple",
                "realtime_color": "bold cyan"
            },
            "dark": {
                "user_color": "bold yellow",
                "assistant_color": "bold purple",
                "banner_color": "dark_blue",
                "highlight_color": "bright_cyan",
                "grok_color": "magenta",
                "realtime_color": "cyan"
            },
            "light": {
                "user_color": "dark_green",
                "assistant_color": "dark_blue",
                "banner_color": "cyan",
                "highlight_color": "magenta",
                "grok_color": "purple",
                "realtime_color": "blue"
            }
        }
        self.current_theme = self.available_themes["default"]

    def display_banner(self):
       """Display an enhanced normal mode banner for SHARVA with selective color styling."""
   
       blink = "\033[5m"  # ANSI blink code
       reset = "\033[0m"  # ANSI reset code
       console.clear()

       # Get current terminal width for centering
       width = shutil.get_terminal_size().columns

       banner_art = r"""
     ____       __  __       ___        ____     _    __       ___ 
   / ___/      / / / /      /   |      / __ \   | |  / /      /   |
   \__ \      / /_/ /      / /| |     / /_/ /   | | / /      / /| |
  ___/ /  _  / __  /   _  / ___ | _  / _, _/  _ | |/ /   _  / ___ |
 /____/  (_)/_/ /_/   (_)/_/  |_|(_)/_/ |_|  (_)|___/   (_)/_/  |_|
       [""" + blink + " NORMAL MODE ACTIVATED " + reset + r"""]

          Developed by Sambhav Mehra
    """

       # Center each line based on the terminal width
       centered_banner = "\n".join(line.center(width) for line in banner_art.strip("\n").splitlines())

       console.print(
          Panel(
            # Use a different color for the banner ASCII art (e.g., bold magenta)
            Text(centered_banner, style="bold magenta"),
            # Title is styled in blue as required
            title="[bold blue]SHARVA AI | Normal Mode v4.1.0[/bold blue]",
            # Subtitle uses a different color (italic yellow)
            subtitle="[italic yellow]Smart Hacker's Assistant for Reconnaissance & Vulnerability Assessment[/italic yellow]",
            # Border is blue
            border_style="blue",
            box=box.DOUBLE,
            width=width
        )
    )

    def show_help(self):
        """Display help information for normal mode with enhanced capabilities."""
        help_text = """
        ## Basic Commands
        
        - `quit` or `exit`: Exit the application
        - `clear`: Clear the screen (or use Ctrl+L)
        - `help`: Show this help menu
        - `switch`: Switch to security mode
        - `history`: History of previous commands
        
        ## Search & Analysis
        
        - `search <query>`: Perform a search for information
        - `grok <query>`: Apply deep analysis to a question or problem
        - `analyze <text>`: Perform in-depth analysis of provided text
        
        ## Session Management
        
        - `save`: Save the current session
        - `load <session_id>`: Load a previous session
        - `sessions`: List all saved sessions
        - `export <filename>`: Export conversation to file
        
        ## Customization
        
        - `theme <theme_name>`: Change color theme (default, dark, light)
        - `keyboard`: Show keyboard shortcuts
        
        ## Tips
        
        - Use Tab for command completion
        - Use ↑/↓ to navigate command history
        - For real-time data, include terms like "current", "latest", "today"
        - Complex questions are automatically analyzed deeply
        - Use Ctrl+G to toggle grok analysis mode
        """
        
        console.print(Panel(
            Markdown(help_text),
            title=f"[{self.current_theme['banner_color']}]Help Menu[/{self.current_theme['banner_color']}]",
            border_style=self.current_theme['banner_color'],
            box=box.ROUNDED
        ))

    def show_keyboard_shortcuts(self):
        """Display keyboard shortcuts."""
        shortcuts = Table(title="Keyboard Shortcuts", box=box.SIMPLE)
        shortcuts.add_column("Shortcut", style="cyan")
        shortcuts.add_column("Action", style="green")
        
        shortcuts.add_row("Tab", "Autocomplete commands")
        shortcuts.add_row("↑/↓", "Navigate command history")
        shortcuts.add_row("Ctrl+L", "Clear screen")
        shortcuts.add_row("Ctrl+S", "Save current session")
        shortcuts.add_row("Ctrl+G", "Toggle grok analysis mode")
        shortcuts.add_row("Ctrl+C", "Cancel current operation")
        
        console.print(shortcuts)
    
    def clear_screen(self):
        """Clear the terminal screen."""
        pt_clear()  # Clear prompt toolkit screen
        self.display_banner()
    
    def change_theme(self, theme_name: str) -> None:
        """Change the color theme."""
        if theme_name in self.available_themes:
            self.theme_name = theme_name
            self.current_theme = self.available_themes[theme_name]
            console.print(f"[success]Theme changed to {theme_name}[/success]")
            self.clear_screen()
        else:
            console.print(f"[error]Theme '{theme_name}' not found. Available themes: {', '.join(self.available_themes.keys())}[/error]")
    
    def save_current_session(self) -> None:
        """Save the current session."""
        self.current_session.save()
        console.print(f"[success]Session saved with ID: {self.current_session.session_id}[/success]")
    
    def load_session(self, session_id: str) -> None:
        """Load a previous session."""
        try:
            self.current_session = Session.load(session_id)
            console.print(f"[success]Loaded session: {session_id}[/success]")
            
            # Display loaded messages
            if self.current_session.messages:
                console.print(Panel(
                    "Session loaded successfully. Displaying previous messages:",
                    title=f"[{self.current_theme['highlight_color']}]Session History[/{self.current_theme['highlight_color']}]",
                    border_style=self.current_theme['highlight_color']
                ))
                
                for msg in self.current_session.messages:
                    if msg.sender == "user":
                        console.print(f"\n[{self.current_theme['user_color']}]You ({msg.timestamp.strftime('%H:%M:%S')})[/{self.current_theme['user_color']}]")
                        console.print(Panel(msg.content, border_style=self.current_theme['user_color']))
                    else:
                        # Check for special message types in metadata
                        metadata = getattr(msg, 'metadata', {})
                        is_realtime = False
                        is_grok = False
                        
                        if metadata and metadata.get('analysis'):
                            is_realtime = metadata['analysis'].get('is_realtime', False)
                            is_grok = metadata['analysis'].get('grok_analysis', {}).get('requires_grok', False)
                        
                        panel_style = self.current_theme['assistant_color']
                        title_prefix = "Assistant"
                        
                        if is_realtime and is_grok:
                            panel_style = "bold magenta"
                            title_prefix = "Assistant (Real-time + Deep Analysis)"
                        elif is_realtime:
                            panel_style = self.current_theme['realtime_color']
                            title_prefix = "Assistant (Real-time)"
                        elif is_grok:
                            panel_style = self.current_theme['grok_color']
                            title_prefix = "Assistant (Deep Analysis)"
                        
                        console.print(f"\n[{panel_style}]{title_prefix} ({msg.timestamp.strftime('%H:%M:%S')})[/{panel_style}]")
                        console.print(Panel(Markdown(msg.content), border_style=panel_style))
            else:
                console.print("[warning]Session loaded but contains no messages[/warning]")
                
        except FileNotFoundError:
            console.print(f"[error]Session not found: {session_id}[/error]")
        except Exception as e:
            console.print(f"[error]Error loading session: {str(e)}[/error]")
    
    def list_sessions(self) -> None:
        """List all saved sessions."""
        sessions = Session.list_sessions()
        
        if not sessions:
            console.print("[info]No saved sessions found.[/info]")
            return
        
        table = Table(title="Saved Sessions", box=box.SIMPLE)
        table.add_column("Session ID", style="cyan")
        table.add_column("Topic", style="yellow")
        table.add_column("Last Updated", style="green")
        table.add_column("Messages", style="magenta")
        
        for session in sessions:
            table.add_row(
                session["session_id"],
                session["topic"],
                session["last_updated"].strftime("%Y-%m-%d %H:%M:%S"),
                str(session["message_count"])
            )
        
        console.print(table)
        console.print("\n[info]To load a session, use: load <session_id>[/info]")
    
    def export_conversation(self, filename: str) -> None:
        """Export the current conversation to a file."""
        if not filename.endswith(('.txt', '.md')):
            filename += '.md'  # Default to markdown format
        
        try:
            with open(filename, 'w') as f:
                f.write(f"# Normal Assistant Conversation\n")
                f.write(f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for msg in self.current_session.messages:
                    sender = "You" if msg.sender == "user" else "Assistant"
                    
                    # Check for special message types in metadata
                    metadata = getattr(msg, 'metadata', {})
                    is_realtime = False
                    is_grok = False
                    
                    if metadata and metadata.get('analysis'):
                        is_realtime = metadata['analysis'].get('is_realtime', False)
                        is_grok = metadata['analysis'].get('grok_analysis', {}).get('requires_grok', False)
                    
                    if sender == "Assistant":
                        if is_realtime and is_grok:
                            sender = "Assistant (Real-time + Deep Analysis)"
                        elif is_realtime:
                            sender = "Assistant (Real-time)"
                        elif is_grok:
                            sender = "Assistant (Deep Analysis)"
                    
                    timestamp = msg.timestamp.strftime("%H:%M:%S")
                    f.write(f"## {sender} ({timestamp})\n\n")
                    f.write(f"{msg.content}\n\n")
            
            console.print(f"[success]Conversation exported to {filename}[/success]")
        except Exception as e:
            console.print(f"[error]Error exporting conversation: {str(e)}[/error]")
    
    def show_history(self):
        """Display command history."""
        try:
            history_file = os.path.join(os.path.expanduser("~"), ".normal_advisor", "history")
            if not os.path.exists(history_file):
                console.print("[info]No command history found.[/info]")
                return
                
            with open(history_file, 'r') as f:
                history_lines = f.readlines()
            
            if not history_lines:
                console.print("[info]Command history is empty.[/info]")
                return
                
            # Create a table for command history
            table = Table(title="Command History", box=box.SIMPLE)
            table.add_column("#", style="cyan")
            table.add_column("Command", style="green")
            
            # Display up to the last 20 commands
            for i, line in enumerate(history_lines[-20:], 1):
                table.add_row(str(i), line.strip())
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[error]Error displaying history: {str(e)}[/error]")

    def process_user_query(self, user_input: str) -> None:
        """Process a user query through both realtime and grok analysis, then merge results."""
        console.print("\n")
        
        # Initialize response panel with loading state
        response_panel = Panel(
            "Processing your query...",
            title=f"[{self.current_theme['assistant_color']}]Assistant[/{self.current_theme['assistant_color']}]",
            border_style=self.current_theme['assistant_color'],
            expand=False
        )
        console.print(response_panel)
        
        # Always process with both engines for every query
        with console.status("[realtime]Fetching real-time data...[/realtime]"):
            realtime_response = self.realtime_engine.process_realtime_query(user_input)
        
        with console.status("[grok]Performing deep analysis...[/grok]"):
            grok_prompt = f"Analyze this question thoroughly with deep thinking: {user_input}"
            grok_response = self.chat_engine.process_normal_query(grok_prompt)
        
        # Merge results by comparing them and selecting the best parts
        merged_response = self.merge_responses(realtime_response, grok_response, user_input)
        
        # Update panel with merged response
        response_panel.title = f"[bold magenta]Assistant (Enhanced Analysis)[/bold magenta]"
        response_panel.border_style = "bold magenta"
        response_panel.renderable = Markdown(merged_response)
        console.print(response_panel)
        
        # Add assistant message to session with metadata about both analyses
        metadata = {
            "analysis": {
                "is_realtime": True,
                "grok_analysis": {"requires_grok": True}
            },
            "processing": {
                "mode": "hybrid",
                "timestamp": datetime.now().isoformat()
            }
        }
        self.current_session.add_message("assistant", merged_response, metadata=metadata)
    
    def merge_responses(self, realtime_response: str, grok_response: str, query: str) -> str:
        """Intelligently merge realtime and grok responses for best results."""
        # First, analyze both responses for relevance and completeness
        relevance_analysis = self.analyze_response_relevance(realtime_response, grok_response, query)
        
        # If one response is significantly better, use it as primary
        if relevance_analysis['realtime_score'] > relevance_analysis['grok_score'] * 1.5:
            # Realtime is much better - use it as primary but incorporate insights from grok
            primary = realtime_response
            secondary = grok_response
            primary_type = "Real-time"
            secondary_type = "Deep Analysis"
        elif relevance_analysis['grok_score'] > relevance_analysis['realtime_score'] * 1.5:
            # Grok is much better - use it as primary but incorporate insights from realtime
            primary = grok_response
            secondary = realtime_response
            primary_type = "Deep Analysis"
            secondary_type = "Real-time"
        else:
            # Both are roughly equivalent - create a true hybrid
            return self.create_hybrid_response(realtime_response, grok_response)
        
        # Create a formatted response with clear sections
        merged = f"""## Primary Response ({primary_type})

{primary}

## Additional Insights ({secondary_type})

{self.extract_key_insights(secondary, primary)}
"""
        return merged
    
    def analyze_response_relevance(self, realtime_response: str, grok_response: str, query: str) -> Dict:
        """Analyze how relevant each response is to the query."""
        # Simple heuristic analysis - could be replaced with more sophisticated NLP
        # Here we just count keyword matches and response length as proxies for relevance
        
        query_keywords = set(re.findall(r'\w+', query.lower()))
        
        # Count keyword matches in each response
        realtime_matches = sum(1 for word in query_keywords if word.lower() in realtime_response.lower())
        grok_matches = sum(1 for word in query_keywords if word.lower() in grok_response.lower())
        
        # Consider response length (longer isn't always better, but is a factor)
        realtime_length_score = min(len(realtime_response) / 500, 3)  # Cap at 3 points
        grok_length_score = min(len(grok_response) / 500, 3)  # Cap at 3 points
        
        # Check for specific indicators of quality
        realtime_quality = 1.0
        grok_quality = 1.0
        
        # Real-time data often has specific indicators like dates, numbers, units
        if re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', realtime_response):  # Date formats
            realtime_quality += 0.5
        if re.search(r'\b\d+(\.\d+)?\s*(USD|EUR|GBP|JPY|$|€|£|¥)\b', realtime_response):  # Currency
            realtime_quality += 0.5
        
        # Grok analysis often has explanatory phrases and logical connectors
        if re.search(r'\b(because|therefore|thus|consequently|as a result)\b', grok_response, re.IGNORECASE):
            grok_quality += 0.5
        if re.search(r'\b(first|second|third|finally|moreover|furthermore)\b', grok_response, re.IGNORECASE):
            grok_quality += 0.5
        
        # Calculate final scores
        realtime_score = (realtime_matches * 1.5) + realtime_length_score + realtime_quality
        grok_score = (grok_matches * 1.5) + grok_length_score + grok_quality
        
        return {
            'realtime_score': realtime_score,
            'grok_score': grok_score,
            'realtime_matches': realtime_matches,
            'grok_matches': grok_matches
        }
    
    def create_hybrid_response(self, realtime_response: str, grok_response: str) -> str:
        """Create a hybrid response combining the best of realtime and grok analysis."""
        # Split responses into paragraphs
        realtime_paras = realtime_response.split('\n\n')
        grok_paras = grok_response.split('\n\n')
        
        # Start with an introduction that mentions both approaches
        hybrid = "# Combined Analysis\n\nThis response combines real-time data with deep analytical insights.\n\n"
        
        # Add a real-time data section
        hybrid += "## Current Information\n\n"
        # Take up to 3 paragraphs from realtime response, preferring those with data
        used_realtime = 0
        for para in realtime_paras:
            if used_realtime >= 3:
                break
            # Prioritize paragraphs with numbers, dates, measurements
            if re.search(r'\b\d+\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', para):
                hybrid += para + "\n\n"
                used_realtime += 1
        
        # If we didn't find enough data-rich paragraphs, add some regular ones
        if used_realtime < 2 and len(realtime_paras) > 0:
            for para in realtime_paras:
                if used_realtime >= 2:
                    break
                if para not in hybrid:
                    hybrid += para + "\n\n"
                    used_realtime += 1
        
        # Add a deep analysis section
        hybrid += "## Deep Analysis\n\n"
        # Take the most insightful paragraphs from grok response
        used_grok = 0
        # First, look for paragraphs with explanatory language
        for para in grok_paras:
            if used_grok >= 3:
                break
            if re.search(r'\b(because|therefore|thus|consequently|as a result|this means|this suggests|this indicates)\b', 
                         para, re.IGNORECASE):
                hybrid += para + "\n\n"
                used_grok += 1
        
        # Add some regular paragraphs if needed
        if used_grok < 2:
            for para in grok_paras:
                if used_grok >= 3:
                    break
                if para not in hybrid:
                    hybrid += para + "\n\n"
                    used_grok += 1
        
        # Add a conclusion
        hybrid += "## Summary\n\n"
        # Try to find a conclusion paragraph in either response
        conclusion = ""
        for para in reversed(grok_paras):
            if re.search(r'\b(in conclusion|to summarize|in summary|overall|ultimately)\b', para, re.IGNORECASE):
                conclusion = para
                break
        
        if not conclusion:
            # Create a simple generic conclusion
            conclusion = "The analysis combines both current data points and deeper insights to provide a comprehensive understanding of your query."
        
        hybrid += conclusion
        
        return hybrid
    
    def extract_key_insights(self, secondary_response: str, primary_response: str) -> str:
        """Extract key insights from secondary response that aren't in primary."""
        # Split into sentences
        secondary_sentences = re.split(r'(?<=[.!?])\s+', secondary_response)
        primary_content = primary_response.lower()
        
        # Look for sentences that contain unique information
        unique_insights = []
        
        for sentence in secondary_sentences:
            # Skip very short sentences
            if len(sentence) < 15:
                continue
                
            # Check if this sentence contains significant words not in primary
            words = re.findall(r'\b\w{5,}\b', sentence.lower())
            unique_words = [word for word in words if word not in primary_content and not word.startswith('http')]
            
            # If sentence has enough unique content, include it
            if len(unique_words) >= 2:
                unique_insights.append(sentence)
        
        # Limit to 3-5 key insights
        if len(unique_insights) > 5:
            unique_insights = unique_insights[:5]
        
        if not unique_insights:
            return "No significant additional insights found."
            
        return " ".join(unique_insights)

    def run(self):
        """Run the normal mode interface."""
        self.display_banner()
        console.print("\n[info]Welcome to Normal Mode. Type 'help' for available commands.[/info]\n")
        
        while True:
            try:
                # Display the enhanced multi-color prompt with current theme
                prompt_text = f"You:"
                user_input = self.prompt_session.prompt(
                    prompt_text,  # Display prompt inline
                    key_bindings=self.kb
                ).strip()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Add message to session
                self.current_session.add_message("user", user_input)
                
                # Process commands
                if user_input.lower() in ["quit", "exit"]:
                    console.print("[info]Exiting Normal Mode. Goodbye![/info]")
                    break
                
                elif user_input.lower() == "clear":
                    self.clear_screen()
                    
                elif user_input.lower() == "help":
                    self.show_help()
                    
                elif user_input.lower() == "switch":
                    console.print("[warning]Security mode not yet implemented.[/warning]")
                    
                elif user_input.lower().startswith("search "):
                    query = user_input[7:].strip()
                    console.print(f"[info]Searching for: {query}[/info]")
                    response, metadata = self.query_processor.process_query(query, mode="search")
                    
                    console.print(f"\n[{self.current_theme['assistant_color']}]Assistant[/{self.current_theme['assistant_color']}]")
                    console.print(Panel(Markdown(response), border_style=self.current_theme['assistant_color']))
                    
                    self.current_session.add_message("assistant", response, metadata=metadata)
                
                elif user_input.lower() == "save":
                    self.save_current_session()
                    
                elif user_input.lower().startswith("load "):
                    session_id = user_input[5:].strip()
                    self.load_session(session_id)
                    
                elif user_input.lower() == "sessions":
                    self.list_sessions()
                    
                elif user_input.lower().startswith("theme "):
                    theme_name = user_input[6:].strip()
                    self.change_theme(theme_name)
                    
                elif user_input.lower() == "keyboard":
                    self.show_keyboard_shortcuts()
                    
                elif user_input.lower().startswith("export "):
                    filename = user_input[7:].strip()
                    self.export_conversation(filename)
                
                elif user_input.lower() == "history":
                    self.show_history()
                
                # New grok analysis command
                elif user_input.lower().startswith("grok "):
                    query = user_input[5:].strip()
                    console.print(f"[grok]🧠 Performing deep analysis on: {query}[/grok]")
                    
                    with console.status("[grok]Thinking deeply...[/grok]"):
                        grok_prompt = f"Analyze this question with extreme thoroughness, showing all steps of reasoning and considering multiple perspectives: {query}"
                        response = self.chat_engine.process_normal_query(grok_prompt)
                    
                    console.print(f"\n[{self.current_theme['grok_color']}]Assistant (Deep Analysis)[/{self.current_theme['grok_color']}]")
                    console.print(Panel(Markdown(response), border_style=self.current_theme['grok_color']))
                    
                    metadata = {
                        "analysis": {
                            "is_realtime": False,
                            "grok_analysis": {"requires_grok": True}
                        }
                    }
                    self.current_session.add_message("assistant", response, metadata=metadata)
                
                # New analyze command for text analysis
                elif user_input.lower().startswith("analyze "):
                    text = user_input[8:].strip()
                    console.print(f"[grok]🔍 Analyzing text...[/grok]")
                    
                    with console.status("[grok]Performing analysis...[/grok]"):
                        analysis_prompt = f"Analyze this text thoroughly, identifying key themes, notable patterns, and important insights: {text}"
                        response = self.chat_engine.process_normal_query(analysis_prompt)
                    
                    console.print(f"\n[{self.current_theme['grok_color']}]Assistant (Text Analysis)[/{self.current_theme['grok_color']}]")
                    console.print(Panel(Markdown(response), border_style=self.current_theme['grok_color']))
                    
                    metadata = {
                        "analysis": {
                            "is_realtime": False,
                            "grok_analysis": {"requires_grok": True, "analysis_type": ["text_analysis"]}
                        }
                    }
                    self.current_session.add_message("assistant", response, metadata=metadata)
                
                else:
                    # Regular query - process through real-time engine and apply grok analysis as needed
                    response, metadata = self.query_processor.process_query(user_input)
                    
                    # Determine appropriate styling based on the analysis
                    panel_style = self.current_theme['assistant_color']
                    title_prefix = "Assistant"
                    
                    if metadata['analysis']['is_realtime'] and metadata['analysis']['grok_analysis']['requires_grok']:
                        panel_style = "bold magenta"
                        title_prefix = "Assistant (Enhanced Analysis)"
                    elif metadata['analysis']['is_realtime']:
                        panel_style = self.current_theme['realtime_color']
                        title_prefix = "Assistant (Real-time)"
                    elif metadata['analysis']['grok_analysis']['requires_grok']:
                        panel_style = self.current_theme['grok_color']
                        title_prefix = "Assistant (Deep Analysis)"
                    
                    console.print(f"\n[{panel_style}]{title_prefix}[/{panel_style}]")
                    console.print(Panel(Markdown(response), border_style=panel_style))
                    
                    self.current_session.add_message("assistant", response, metadata=metadata)
                
            except KeyboardInterrupt:
                console.print("\n[warning]Operation cancelled. Type 'exit' to quit.[/warning]")
            except Exception as e:
                console.print(f"[error]Error: {str(e)}[/error]")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    config = Config()
    normal_mode = NormalMode(config)
    normal_mode.run()
