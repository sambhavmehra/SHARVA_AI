import os
import sys
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional
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
})

console = Console(theme=custom_theme)

class Message:
    """Class to represent a chat message."""
    def __init__(self, sender: str, content: str, timestamp: datetime = None):
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary for serialization."""
        return {
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Create message from dictionary."""
        return cls(
            sender=data["sender"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


class Session:
    """Class to manage chat sessions."""
    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.messages: List[Message] = []
        self.session_dir = os.path.join(os.path.expanduser("~"), ".normal_advisor", "sessions")
    
    def add_message(self, sender: str, content: str) -> None:
        """Add a message to the session."""
        self.messages.append(Message(sender, content))
    
    def save(self) -> None:
        """Save the session to disk."""
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
        """Load a session from disk."""
        session = cls(session_id)
        session_file = os.path.join(session.session_dir, f"{session_id}.json")
        
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                data = json.load(f)
            
            session.messages = [Message.from_dict(msg) for msg in data["messages"]]
        
        return session
    
    @classmethod
    def list_sessions(cls) -> List[Dict]:
        """List all saved sessions."""
        session_dir = os.path.join(os.path.expanduser("~"), ".normal_advisor", "sessions")
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


class NormalMode:
    def __init__(self, config):
        self.config = config
        
        # Import here to avoid circular imports
        from chat_engine import ChatEngine
        self.chat_engine = ChatEngine(config)
        
        # Create command completer
        self.commands = [
            "quit", "exit", "clear", "help", "switch", "search", 
            "save", "load", "sessions", "theme", "keyboard", "export"
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
        
        # Initialize session
        self.current_session = Session()
        
        # Default theme settings
        self.theme_name = "default"
        self.available_themes = {
            "default": {
                "user_color": "bold green",
                "assistant_color": "bold blue",
                "banner_color": "blue",
                "highlight_color": "cyan"
            },
            "dark": {
                "user_color": "bold yellow",
                "assistant_color": "bold purple",
                "banner_color": "dark_blue",
                "highlight_color": "bright_cyan"
            },
            "light": {
                "user_color": "dark_green",
                "assistant_color": "dark_blue",
                "banner_color": "cyan",
                "highlight_color": "magenta"
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
            title="[bold blue]SHARVA AI | Normal Mode v4.0.1[/bold blue]",
            # Subtitle uses a different color (italic yellow)
            subtitle="[italic yellow]Smart Hacker's Assistant for Reconnaissance & Vulnerability Assessment[/italic yellow]",
            # Border is blue
            border_style="blue",
            box=box.DOUBLE,
            width=width
        )
    )



    
    def show_help(self):
        """Display help information for normal mode."""
        help_text = """
        ## Basic Commands
        
        - `quit` or `exit`: Exit the application
        - `clear`: Clear the screen (or use Ctrl+L)
        - `help`: Show this help menu
        - `switch`: Switch to security mode
        
        ## Search & Conversation
        
        - `search <query>`: Perform a search for information
        
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
        - Ask questions naturally as you would to a human assistant
        - For technical cybersecurity queries, switch to security mode
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
                    console.print(f"\n[{self.current_theme['assistant_color']}]Assistant ({msg.timestamp.strftime('%H:%M:%S')})[/{self.current_theme['assistant_color']}]")
                    console.print(Panel(Markdown(msg.content), border_style=self.current_theme['assistant_color']))
            
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
        table.add_column("Last Updated", style="green")
        table.add_column("Messages", style="yellow")
        
        for session in sessions:
            table.add_row(
                session["session_id"],
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
                    timestamp = msg.timestamp.strftime("%H:%M:%S")
                    f.write(f"## {sender} ({timestamp})\n\n")
                    f.write(f"{msg.content}\n\n")
            
            console.print(f"[success]Conversation exported to {filename}[/success]")
        except Exception as e:
            console.print(f"[error]Error exporting conversation: {str(e)}[/error]")
    
    def run(self):
        """Run the normal mode interface."""
        self.clear_screen()
        
        while True:
            try:
                # Get user input with command completion and history
                user_input = self.prompt_session.prompt(
                    f"You :",
                    key_bindings=self.kb,
                    mouse_support=True
                )
                
                # Add user message to session
                self.current_session.add_message("user", user_input)
                
                # Process commands
                if user_input.lower() in ['quit', 'exit']:
                    # Ask if user wants to save before quitting
                    if len(self.current_session.messages) > 1:
                        save_confirm = Prompt.ask(
                            "[warning]Save this session before quitting? (y/n)[/warning]",
                            choices=["y", "n"], default="y"
                        )
                        if save_confirm.lower() == 'y':
                            self.save_current_session()
                    
                    console.print("[warning]Goodbye![/warning]")
                    break
                    
                elif user_input.lower() == 'clear':
                    self.clear_screen()
                    
                elif user_input.lower() == 'help':
                    self.show_help()
                    
                elif user_input.lower() == 'keyboard':
                    self.show_keyboard_shortcuts()
                    
                elif user_input.lower() == 'switch':
                    console.print("[warning]Switching to security mode...[/warning]")
                    return "security"
                    
                elif user_input.lower().startswith('search '):
                    query = user_input[7:].strip()  # Remove 'search ' prefix
                    if query:
                        response_panel = Panel(
                            "",
                            title=f"[{self.current_theme['assistant_color']}]Assistant[/{self.current_theme['assistant_color']}]",
                            border_style=self.current_theme['assistant_color'],
                            expand=False
                        )
                        console.print("\n")
                        console.print(response_panel)
                        
                        with console.status(f"[success]Searching for information...[/success]"):
                            response = self.chat_engine.process_normal_query(query, use_search=True)
                        
                        # Update panel with response content
                        response_panel.renderable = Markdown(response)
                        console.print(response_panel)
                        
                        # Add assistant message to session
                        self.current_session.add_message("assistant", response)
                    else:
                        console.print("[error]Please provide a search query[/error]")
                
                elif user_input.lower() == 'save':
                    self.save_current_session()
                
                elif user_input.lower().startswith('load '):
                    session_id = user_input[5:].strip()
                    self.load_session(session_id)
                
                elif user_input.lower() == 'sessions':
                    self.list_sessions()
                
                elif user_input.lower().startswith('theme '):
                    theme_name = user_input[6:].strip()
                    self.change_theme(theme_name)
                
                elif user_input.lower().startswith('export '):
                    filename = user_input[7:].strip()
                    self.export_conversation(filename)
                    
                else:
                    # Regular query processing
                    response_panel = Panel(
                        "",
                        title=f"[{self.current_theme['assistant_color']}]Assistant[/{self.current_theme['assistant_color']}]",
                        border_style=self.current_theme['assistant_color'],
                        expand=False
                    )
                    console.print("\n")
                    console.print(response_panel)
                    
                    with console.status(f"[success]Thinking...[/success]"):
                        try:
                            response = self.chat_engine.process_normal_query(user_input)
                            # Update panel with response content
                            response_panel.renderable = Markdown(response)
                            console.print(response_panel)
                            
                            # Add assistant message to session
                            self.current_session.add_message("assistant", response)
                        except Exception as e:
                            error_message = f"Error processing query: {str(e)}"
                            response_panel.renderable = Text(error_message, style="bold red")
                            console.print(response_panel)
                    
            except KeyboardInterrupt:
                console.print("\n[warning]Use 'quit' to exit properly[/warning]")
            except Exception as e:
                console.print(f"[error]Error:[/error] {str(e)}")
                console.print("[info]Type [command]help[/command] for available commands[/info]")
        
        return "quit"