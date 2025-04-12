import os
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich import box
from rich.layout import Layout
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich.syntax import Syntax
from config import Config
from datetime import datetime
import random
import shutil
from hackbot import HackBot

console = Console()

class SecurityMode:
    """Enhanced cybersecurity-focused mode with advanced security tools for both defensive and offensive security."""

    def __init__(self, config):
        self.config = config
        self.session_start = datetime.now()
        self.query_count = 0
        self.security_level = "standard"
        self.security_mode = "defensive"  # Default to defensive mode
        self.auth_level = "standard"  # Authorization level
        self.session_id = f"SID-{random.randint(100000, 999999)}"
        self.hackbot = None

        # Import here to avoid circular imports
        from chat_engine import ChatEngine
        self.chat_engine = ChatEngine(config)

    def display_banner(self):
       """Display an enhanced security mode banner for SHARVA."""
       blink = "\033[5m"
       reset = "\033[0m"
       console.clear()

       width = shutil.get_terminal_size().columns

       banner_art = r"""
    ____       __  __       ___        ____     _    __       ___ 
  / ___/      / / / /      /   |      / __ \   | |  / /      /   |
  \__ \      / /_/ /      / /| |     / /_/ /   | | / /      / /| |
 ___/ /  _  / __  /   _  / ___ | _  / _, _/  _ | |/ /   _  / ___ |
/____/  (_)/_/ /_/   (_)/_/  |_|(_)/_/ |_|  (_)|___/   (_)/_/  |_|
       [""" + blink + " SECURITY MODE ACTIVATED " + blink + r"""]
       
 Developed by Sambhav Mehra
    """

    # Center each line manually
       centered_banner = "\n".join(line.center(width) for line in banner_art.strip("\n").splitlines())

       console.print(
           Panel(
               Text(centered_banner, style="bold red"),
               title="[bold green]SHARVA AI | Security Mode v4.0.1[/bold green]",
               subtitle="[italic yellow]Smart Hacker's Assistant for Reconnaissance & Vulnerability Assessment[/italic yellow]",
               border_style="bright_red",
               box=box.DOUBLE,
               width=width
        )
    )
    def _generate_hash(self):
        """Generate a fake hash for visual effect."""
        hash_chars = "0123456789abcdef"
        return ''.join(random.choice(hash_chars) for _ in range(32))
    
    def _generate_glitchy_text(self, text):
        """Generate glitchy text with some characters randomly styled."""
        result = ""
        glitch_chars = "!@#$%^&*()_+-=[]\\{}|;':\",./<>?"
        
        for char in text:
            if random.random() < 0.2:  # 20% chance for a character to be styled
                style = random.choice(["bold red", "bold cyan", "bold yellow", "reverse"])
                
                if random.random() < 0.3:
                  glitch_char = random.choice(glitch_chars)
                  result += f"[{style}]{glitch_char}[/{style}]"
            # Sometimes add blink
                elif random.random() < 0.4:
                  result += f"[{style}][blink]{char}[/blink][/{style}]"
                else:
                  result += f"[{style}]{char}[/{style}]"
            else:
             result += char
            return result
        
                
            

    def show_help(self):
        """Display enhanced security help with command categories."""
        # More aggressive looking help menu
        commands_table = Table(show_header=True, header_style="bold red", 
                             border_style="red", box=box.HEAVY)
        commands_table.add_column("[CMD]", style="cyan", justify="left")
        commands_table.add_column("[DESCRIPTION]", style="white")
        commands_table.add_column("[ACCESS LVL]", style="green", justify="center", width=12)

        # General commands
        commands_table.add_row("help", "Show this help menu", "ALL")
        commands_table.add_row("clear", "Clear the screen", "ALL")
        commands_table.add_row("stats", "Show session statistics", "ALL")
        commands_table.add_row("switch", "Switch to normal mode", "ALL")
        commands_table.add_row("quit/exit", "Terminate application", "ALL")
        
        # Configuration commands
        commands_table.add_row("set_level <level>", "Set security level (standard/advanced/expert)", "ALL")
        commands_table.add_row("set_mode <mode>", "Set security mode (defensive/offensive)", "ALL")
        commands_table.add_row("set_auth <level>", "Set auth level (standard/government/certified)", "ALL")
        
        # Defensive commands
        commands_table.add_row("vuln_analysis", "Analyze vulnerability scan data", "STD+")
        commands_table.add_row("static_code_analysis", "Perform static code analysis", "STD+")
        commands_table.add_row("threat_hunt <log_file>", "Search for threats in log files", "STD+")
        commands_table.add_row("malware_analysis <file>", "Analyze potential malware samples", "ADV+")
        
        # Offensive commands
        if self.security_mode == "offensive":
            commands_table.add_row("recon <target>", "Perform passive reconnaissance (OSINT)", "ADV+")
            commands_table.add_row("pentest_report <scope>", "Generate penetration testing report", "ADV+")
            commands_table.add_row("exploit_analysis <cve>", "Analyze exploit for CVE", "ADV+")
            commands_table.add_row("payload_gen <platform>", "Generate test payload for platform", "GOV+")
            commands_table.add_row("run_hackbot", "Run HackBot cybersecurity assistant", "GOV+")
            commands_table.add_row("", "[red]Requires offensive mode & government auth[/red]", "")

        # Topics based on current mode with more technical details
        if self.security_mode == "defensive":
            topics = Text.from_markup("""
            [bold red]> DEFENSIVE SECURITY MODULES:[/bold red]
            
            [bold cyan]1.[/bold cyan] [white]Vulnerability Assessment & Management[/white]
            [bold cyan]2.[/bold cyan] [white]Network Security Hardening Protocols[/white]
            [bold cyan]3.[/bold cyan] [white]Web Application Security (OWASP Top 10)[/white]
            [bold cyan]4.[/bold cyan] [white]Advanced Malware Analysis & Containment[/white]
            [bold cyan]5.[/bold cyan] [white]Incident Response Procedures (NIST)[/white]
            [bold cyan]6.[/bold cyan] [white]Security Tool Operation (Nmap, Burp Suite, etc.)[/white]
            [bold cyan]7.[/bold cyan] [white]Secure Coding Practices & Architecture[/white]
            [bold cyan]8.[/bold cyan] [white]Threat Intelligence Integration[/white]
            [bold cyan]9.[/bold cyan] [white]Security Monitoring & SIEM Configuration[/white]
            """)
        else:  # offensive mode
            topics = Text.from_markup("""
            [bold red]> OFFENSIVE SECURITY MODULES:[/bold red]
            
            [bold cyan]1.[/bold cyan] [white]Penetration Testing Methodologies (PTES/OSSTMM)[/white]
            [bold cyan]2.[/bold cyan] [white]Advanced OSINT Collection Techniques[/white]
            [bold cyan]3.[/bold cyan] [white]Network Infrastructure Attack Vectors[/white]
            [bold cyan]4.[/bold cyan] [white]Web Application Attack Surface Analysis[/white]
            [bold cyan]5.[/bold cyan] [white]Exploit Development Frameworks[/white]
            [bold cyan]6.[/bold cyan] [white]Social Engineering Attack Simulation[/white]
            [bold cyan]7.[/bold cyan] [white]Red Team Operation Planning[/white]
            [bold cyan]8.[/bold cyan] [white]Vulnerability Research Methodology[/white]
            [bold cyan]9.[/bold cyan] [white]CTF Technique Library & Reference[/white]
            """)

        # More aggressive looking ethical notice
        disclaimer = Text.from_markup("""
        [bold red]╔══════════════════════════════════════════════════╗[/bold red]
        [bold red]║[/bold red] [bold yellow]!!! OPERATIONAL SECURITY DIRECTIVE !!![/bold yellow]         [bold red]║[/bold red]
        [bold red]╚══════════════════════════════════════════════════╝[/bold red]
        
        [bold white]ALL SECURITY OPERATIONS MUST COMPLY WITH:[/bold white]
        
        [bold cyan]>[/bold cyan] [white]Explicit written authorization[/white]
        [bold cyan]>[/bold cyan] [white]Applicable legal frameworks[/white]
        [bold cyan]>[/bold cyan] [white]Defined scope limitations[/white]
        [bold cyan]>[/bold cyan] [white]Chain-of-custody documentation[/white]
        
        [bold red]VIOLATION OF OPERATIONAL SECURITY DIRECTIVES[/bold red]
        [bold red]MAY RESULT IN ACCESS TERMINATION & LEGAL ACTION[/bold red]
        """)

        layout = Layout()
        layout.split_column(
            Layout(Panel(commands_table, 
                     title="[bold red][ COMMAND INTERFACE ]",
                     subtitle="[ ACCESS LEVEL: " + self.auth_level.upper() + " ]",
                     border_style="red", 
                     box=box.HEAVY)),
            Layout().split_row(
                Layout(Panel(topics, 
                          title=f"[bold red][ {self.security_mode.upper()} MODULES ]",
                          border_style="blue", 
                          box=box.HEAVY_EDGE)),
                Layout(Panel(disclaimer, 
                          title="[bold yellow][ SECURITY DIRECTIVE ]",
                          border_style="yellow", 
                          box=box.HEAVY_EDGE))
            )
        )

        console.print(layout)

    def show_stats(self):
        """Display security session statistics."""
        session_duration = datetime.now() - self.session_start
        
        # Create a hacker-style frame for stats
        stats_frame_top = Text.from_markup("[bold cyan]╔════════════════ SECURITY SESSION METRICS ════════════════╗[/bold cyan]")
        stats_frame_bottom = Text.from_markup("[bold cyan]╚═══════════════════════════════════════════════════════╝[/bold cyan]")

        # Create matrix-like stats display
        stats = []
        stats.append(f"[bold green]>[/bold green] [bold white]SESSION_ID......:[/bold white] {self.session_id}")
        stats.append(f"[bold green]>[/bold green] [bold white]UPTIME.........:[/bold white] {str(session_duration).split('.')[0]}")
        stats.append(f"[bold green]>[/bold green] [bold white]QUERIES........:[/bold white] {self.query_count}")
        stats.append(f"[bold green]>[/bold green] [bold white]SECURITY.......:[/bold white] {self.security_level.upper()}")
        stats.append(f"[bold green]>[/bold green] [bold white]MODE...........:[/bold white] {self.security_mode.upper()}")
        stats.append(f"[bold green]>[/bold green] [bold white]AUTH_LEVEL.....:[/bold white] {self.auth_level.upper()}")
        stats.append(f"[bold green]>[/bold green] [bold white]MODEL..........:[/bold white] {self.config.default_model}")
        stats.append(f"[bold green]>[/bold green] [bold white]ANALYSES.......:[/bold white] {self.query_count // 2}")
        
        # Add fake memory usage and system load
        stats.append(f"[bold green]>[/bold green] [bold white]MEM_USAGE......:[/bold white] {random.randint(300, 500)} MB")
        stats.append(f"[bold green]>[/bold green] [bold white]SYS_LOAD.......:[/bold white] {random.randint(15, 85)}%")
        
        # Create a mini threat chart
        threat_data = [
            (random.randint(20, 100), "RED", "Critical"),
            (random.randint(30, 150), "YELLOW", "Warning"),
            (random.randint(100, 300), "GREEN", "Info")
        ]
        
        threat_chart = Table(show_header=False, box=None)
        threat_chart.add_column("Count", style="cyan", justify="right")
        threat_chart.add_column("Level", style="white")
        threat_chart.add_column("Type", style="white")
        
        for count, color, level in threat_data:
            threat_chart.add_row(f"{count}", f"[{color}]■[/{color}]", level)
        
        # Create the layout
        console.print(stats_frame_top)
        for stat in stats:
            console.print(Text.from_markup(f"[bold cyan]║[/bold cyan] {stat}"))
        
        console.print(Text.from_markup("[bold cyan]║[/bold cyan]"))
        console.print(Text.from_markup("[bold cyan]║[/bold cyan] [bold white]THREAT_ALERTS...:[/bold white]"))
        
        # Print threat chart with proper alignment
        for row in threat_chart.rows:
            formatted_row = "  ".join(str(cell) for cell in row)
            console.print(Text.from_markup(f"[bold cyan]║[/bold cyan]   {formatted_row}"))
            
        console.print(stats_frame_bottom)
        
        # Add a "system healthy" message
        console.print(Text.from_markup("\n[bold green][ SYSTEM STATUS: OPERATIONAL ][/bold green]"))

    def set_security_level(self, level):
        """Set the security analysis level."""
        valid_levels = ["standard", "advanced", "expert"]
        if level.lower() in valid_levels:
            self.security_level = level.lower()
            
            # Animated level change for visual effect
            with Progress(
                SpinnerColumn("dots", style="red"),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task(f"[red]Adjusting security level to {level.upper()}...", total=1)
                time.sleep(0.5)
                progress.update(task, advance=1)
            
            console.print(f"[bold green][[/bold green] Security level set to: [bold red]{self.security_level.upper()}[/bold red] [bold green]][/bold green]")
            
            if level.lower() == "expert":
                console.print(Text.from_markup("[bold yellow][blink]! WARNING: EXPERT MODE ENABLED ![/blink][/bold yellow]"))
        else:
            console.print(f"[bold red][!] ERROR:[/bold red] Invalid level. Choose from: {', '.join(valid_levels)}")

    def set_security_mode(self, mode):
        """Set the security mode (defensive/offensive)."""
        valid_modes = ["defensive", "offensive"]
        if mode.lower() in valid_modes:
            prev_mode = self.security_mode
            self.security_mode = mode.lower()
            
            # Animated mode switching
            with Progress(
                SpinnerColumn("dots", style="red"),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task(f"[red]Switching mode: {prev_mode.upper()} → {mode.upper()}...", total=1)
                time.sleep(0.5)
                progress.update(task, advance=1)
            
            console.print(f"[bold green][[/bold green] Security mode set to: [bold red]{self.security_mode.upper()}[/bold red] [bold green]][/bold green]")
            
            # Display authorization warning if switching to offensive mode
            if self.security_mode == "offensive":
                console.print(Panel(
                    "[bold yellow]AUTHORIZATION REQUIRED[/bold yellow]\n" +
                    "[bold white]Offensive security operations must be authorized under:[/bold white]\n" +
                    "- Formal penetration testing engagement\n" +
                    "- Red team authorization documents\n" +
                    "- Government security directive\n\n" +
                    "[bold red]All activities are cryptographically logged[/bold red]",
                    title="[bold red][ SECURITY ALERT ][/bold red]",
                    border_style="red", box=box.HEAVY_EDGE
                ))
        else:
            console.print(f"[bold red][!] ERROR:[/bold red] Invalid mode. Choose from: {', '.join(valid_modes)}")
    def authenticate(self):
        """Dramatic authentication sequence."""
        console.print("[bold red]SECURITY AUTHENTICATION REQUIRED[/bold red]")
    
        username = Prompt.ask("[bold green]>[/bold green] Username")
    
    # Dramatic password entry with fancy masking
        console.print("[bold green]>[/bold green] Password", end="")
        console.print(" ", end="")
    
        password = ""
        password_display = ""
        while True:
            key = self._get_key()
            if key == '\r' or key == '\n':
                console.print()
                break
            elif key == '\x7f' or key == '\x08':
                if len(password) > 0:
                    password = password[:-1]
                # Erase last character
                console.print("\b \b", end="", flush=True)
                password_display = password_display[:-1]
            elif key.isprintable():
                password += key
            
            # Show random character for dramatic effect
                mask_char = random.choice("*#$@")
                password_display += mask_char
                console.print(mask_char, end="", flush=True) 
                if random.random() < 0.2 and len(password) > 3:
                    scan_msg = "[bold yellow][scanning][/bold yellow]"
                    console.print(f" {scan_msg}", end="", flush=True)
                    time.sleep(0.2)
                # Erase the scanning message
                    console.print("\b" * (len(scan_msg) + 1) + " " * (len(scan_msg) + 1) + "\b" * (len(scan_msg) + 1), end="", flush=True)
            with Progress(
                SpinnerColumn("dots", style="yellow"),
                TextColumn("[yellow]Authenticating...[/yellow]"),
                transient=True
            ) as progress:
                task = progress.add_task("auth", total=1)
                time.sleep(2)
                progress.update(task, advance=1)
            console.print("[bold green]Authentication successful[/bold green]")
            return True 
    def _get_key(self):
        """Get a single keypress from the user."""
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1) 
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch                  
    def setup_tab_completion(self):
        """Set up tab completion for commands."""
        import readline
        commands = [
        'help', 'clear', 'stats', 'switch', 'exit', 'quit',
        'set_level standard', 'set_level advanced', 'set_level expert',
        'set_mode defensive', 'set_mode offensive',
        'set_auth standard', 'set_auth government', 'set_auth certified',
        'vuln_analysis', 'static_code_analysis', 'threat_hunt',
        'malware_analysis', 'recon', 'pentest_report',
        'exploit_analysis', 'payload_gen', 'show_threat_map'
    ]
        def completer(text, state):
            options = [cmd for cmd in commands if cmd.startswith(text)]
            if state < len(options):
                return options[state]
            else:
                return None
        readline.parse_and_bind("tab: complete")
        readline.set_completer(completer)
        
    def show_network_traffic(self):
        """Display an animated network traffic visualization."""
        console.print("\n[bold cyan]===== NETWORK TRAFFIC ANALYZER =====[/bold cyan]")
        console.print("[bold yellow]Monitoring live traffic...[/bold yellow]")
        services = ['HTTP', 'HTTPS', 'DNS', 'FTP', 'SSH', 'SMTP', 'RDP', 'SMB']
        traffic_data = {}
        for service in services:
            traffic_data[service] = {
                'value': random.randint(10, 100),
                'direction': random.choice(['in', 'out']),
                'status': random.choice(['normal', 'suspicious', 'blocked']),}
        for _ in range(20):
            os.system('clear') if os.name != 'nt' else os.system('cls')
            console.print("\n[bold cyan]===== LIVE NETWORK TRAFFIC =====[/bold cyan]")
            for service in services:
                traffic_data[service]['value'] += random.randint(-5, 5)
                traffic_data[service]['value'] = max(5, min(100, traffic_data[service]['value']))
                if random.random() < 0.1:
                    traffic_data[service]['status'] = random.choice(['normal', 'suspicious', 'blocked'])
                    status_color = {
                        'normal': 'green',
                        'suspicious': 'yellow',
                        'blocked': 'red'
                        }[traffic_data[service]['status']]
                direction = "▶" if traffic_data[service]['direction'] == 'out' else "◀"
                bar_length = traffic_data[service]['value'] // 2
                bar = "█" * bar_length
                console.print(f"[bold white]{service:5}[/bold white] [{status_color}]{direction} {bar} {traffic_data[service]['value']:3d} KB/s[/{status_color}]")
            console.print("\n[bold white]Press Ctrl+C to stop monitoring[/bold white]")
            time.sleep(0.3)
                              
    def set_auth_level(self, level):
        """Set the authorization level."""
        valid_levels = ["standard", "government", "certified"]
        if level.lower() in valid_levels:
            prev_level = self.auth_level
            self.auth_level = level.lower()
            
            # Authentication animation
            console.print("\n[bold yellow]Authentication verification in progress...[/bold yellow]")
            
            # Fake authentication process
            auth_steps = [
                "Verifying credentials...",
                "Checking authorization database...",
                "Validating security clearance...",
                "Updating access controls..."
            ]
            
            with Progress(
                SpinnerColumn("line", style="yellow"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=30, style="yellow", complete_style="green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                expand=False
            ) as progress:
                task = progress.add_task("[yellow]Authentication in progress...", total=100)
                
                for i, step in enumerate(auth_steps):
                    progress.update(task, description=f"[yellow]{step}", advance=25)
                    time.sleep(0.3)
            
            console.print(f"[bold green][[/bold green] Authorization level set to: [bold red]{self.auth_level.upper()}[/bold red] [bold green]][/bold green]")
            
            if level.lower() in ["government", "certified"]:
                # Cyberpunk-style clearance animation
                console.print(Text.from_markup("\n[bold green]┌──────────────────────────────────┐[/bold green]"))
                console.print(Text.from_markup("[bold green]│[/bold green] [bold white]ADVANCED CLEARANCE GRANTED[/bold white]      [bold green]│[/bold green]"))
                console.print(Text.from_markup("[bold green]└──────────────────────────────────┘[/bold green]"))
                console.print("[bold green]Advanced capabilities unlocked.[/bold green]")
        else:
            console.print(f"[bold red][!] ERROR:[/bold red] Invalid auth level. Choose from: {', '.join(valid_levels)}")

    def clear_screen(self):
        """Enhanced screen clearing with security animation."""
        # More dramatic screen clearing animation
        console.print("\n[bold red]Initiating secure terminal wipe...[/bold red]")
        
        with Progress(
            SpinnerColumn("dots12", style="red"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40, style="red", complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=False
        ) as progress:
            task = progress.add_task("[bold red]Clearing secure terminal...", total=100)
            
            clear_steps = [
                "Flushing memory buffers...",
                "Wiping command history...",
                "Clearing screen buffer...",
                "Resetting terminal state..."
            ]
            
            for i, step in enumerate(clear_steps):
                progress.update(task, description=f"[bold red]{step}", advance=25)
                time.sleep(0.2)
        
        # Actually clear the screen
        os.system('cls' if os.name == 'nt' else 'clear')
        self.display_banner()

    def process_security_query(self, query):
        """Process security query with enhanced feedback."""
        self.query_count += 1

        # Add contextual information about current mode
        mode_context = f"Current mode: {self.security_mode}. Security level: {self.security_level}. "
        mode_context += "Provide information suitable for professional security work with government authorization. "
        enhanced_query = mode_context + query

        # More dramatic progress display
        console.print(f"\n[bold red][[/bold red] [bold white]ANALYZING SECURITY QUERY[/bold white] [bold red]][/bold red]")
        
        with Progress(
            SpinnerColumn("dots", style="red"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50, style="red", complete_style="green"),
            expand=False
        ) as progress:
            # Create multiple steps for visual effect
            tasks = []
            steps = [
                f"Parsing query...",
                f"Processing in {self.security_mode.upper()} mode...",
                f"Applying security filters...",
                f"Generating response..."
            ]
            
            for step in steps:
                task = progress.add_task(f"[red]{step}", total=1)
                tasks.append(task)
            
            # Process the actual query while showing the animation
            response = self.chat_engine.process_security_query(enhanced_query)
            
            # Complete the progress bars
            for i, task in enumerate(tasks):
                time.sleep(0.3)
                progress.update(task, advance=1)

        # Hacker-style response header
        current_time = datetime.now().strftime("%H:%M:%S")
        console.print(f"\n[bold red]╔═══[{current_time}]═══[/bold red][bold cyan]SecurityBot[/bold cyan][bold red]═[{self.security_mode.upper()}]═════╗[/bold red]")
        console.print(Markdown(response))
        console.print(f"[bold red]╚════════════════════════════════════════════════════╝[/bold red]")

    def static_code_analysis(self):
        """Enhanced static code analysis workflow."""
        console.print(Text.from_markup("\n[bold red]==== STATIC CODE ANALYSIS ENGINE ====[/bold red]"))
        console.print(Text.from_markup("[bold cyan]Input required parameters:[/bold cyan]"))
        
        language = Prompt.ask("[bold green]>[/bold green] Programming Language")
        file_path = Prompt.ask("[bold green]>[/bold green] File Path")
        
        # Animated code analysis
        console.print("\n[bold yellow]Beginning static analysis...[/bold yellow]")
        
        with Progress(
            SpinnerColumn("dots", style="red"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50, style="red", complete_style="green"),
            expand=False
        ) as progress:
            tasks = []
            
            steps = [
                "Reading source code...",
                "Tokenizing and parsing...",
                "Building AST...",
                "Analyzing control flow...",
                "Checking for vulnerabilities...",
                "Generating report..."
            ]
            
            for step in steps:
                task = progress.add_task(f"[red]{step}", total=1)
                tasks.append(task)
            
            # Process the actual query
            result = self.chat_engine.static_code_analysis(language, file_path)
            
            # Complete the progress bars with slight delays
            for i, task in enumerate(tasks):
                time.sleep(0.3)
                progress.update(task, advance=1)

        # Display results in a more cyberpunk style
        console.print("\n[bold red]╔═══ VULNERABILITY SCAN REPORT ════════════════════╗[/bold red]")
        console.print(f"[bold red]║[/bold red] [bold cyan]Target:[/bold cyan] {os.path.basename(file_path)}")
        console.print(f"[bold red]║[/bold red] [bold cyan]Language:[/bold cyan] {language}")
        console.print(f"[bold red]║[/bold red] [bold cyan]Timestamp:[/bold cyan] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print("[bold red]╠═══════════════════════════════════════════════════╣[/bold red]")
        
        console.print(Syntax(result, "text", theme="monokai", line_numbers=True))
        console.print("[bold red]╚═══════════════════════════════════════════════════╝[/bold red]")

    def vulnerability_analysis(self):
        """Enhanced vulnerability analysis workflow."""
        console.print(Text.from_markup("\n[bold red]==== VULNERABILITY ANALYSIS ENGINE ====[/bold red]"))
        console.print(Text.from_markup("[bold cyan]Input required parameters:[/bold cyan]"))
        
        scan_type = Prompt.ask("[bold green]>[/bold green] Scan Type (nmap/nessus/burp/etc)")
        file_path = Prompt.ask("[bold green]>[/bold green] File Path")
        
        # Animated vulnerability analysis process
        console.print("\n[bold yellow]Beginning vulnerability analysis...[/bold yellow]")
        
        with Progress(
            SpinnerColumn("dots", style="red"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50, style="red", complete_style="green"),
            expand=False
        ) as progress:
            tasks = []
            
            steps = [
                f"Parsing {scan_type} data format...",
                "Extracting vulnerability data...",
                "Cross-referencing with threat database...",
                "Calculating risk scores...",
                "Prioritizing findings...",
                "Generating assessment..."
            ]
            
            for step in steps:
                task = progress.add_task(f"[red]{step}", total=1)
                tasks.append(task)
            
            # Process the vulnerability data
            result = self.chat_engine.analyze_vulnerabilities(scan_type, file_path)
            
            # Complete the progress bars with slight delays
            for i, task in enumerate(tasks):
                time.sleep(0.3)
                progress.update(task, advance=1)

        # Create a more dramatic table for critical findings
        table = Table(title="[bold red][ CRITICAL VULNERABILITY FINDINGS ]",
                     border_style="red", box=box.HEAVY_EDGE)
        table.add_column("ID", style="cyan", justify="left")
        table.add_column("Severity", style="red", justify="center")
        table.add_column("Category", style="yellow")
        table.add_column("Description", style="white")
        table.add_column("CVSS", style="green", justify="center")
        
        # Extract findings from result and populate table
        findings = result.strip().split('\n\n')
        for finding in findings[:5]:  # Show top 5 findings for display purposes
            parts = finding.split('\n')
            if len(parts) >= 4:
                id = parts[0].split(':')[1].strip() if ':' in parts[0] else "N/A"
                severity = parts[1].split(':')[1].strip() if ':' in parts[1] else "N/A"
                category = parts[2].split(':')[1].strip() if ':' in parts[2] else "N/A"
                description = parts[3].split(':')[1].strip() if ':' in parts[3] else "N/A"
                cvss = f"{random.uniform(3.0, 10.0):.1f}" if severity.lower() in ['high', 'critical'] else f"{random.uniform(1.0, 4.9):.1f}"
                
                table.add_row(id, severity, category, description, cvss)
        
        console.print(table)
        
        # Display additional analysis details
        console.print(Panel(Markdown(result), 
                          title="[bold red][ DETAILED ANALYSIS ]",
                          border_style="red", 
                          box=box.HEAVY_EDGE))

    def threat_hunt(self, log_file=None):
        """Threat hunting in log files."""
        if not log_file:
            log_file = Prompt.ask("[bold green]>[/bold green] Log File Path")
        
        console.print(Text.from_markup("\n[bold red]==== THREAT HUNTING ENGINE ====[/bold red]"))
        
        # Animated threat hunting process
        console.print("\n[bold yellow]Initiating threat hunting procedure...[/bold yellow]")
        
        with Progress(
            SpinnerColumn("dots", style="red"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50, style="red", complete_style="green"),
            expand=False
        ) as progress:
            tasks = []
            
            steps = [
                "Loading log data...",
                "Analyzing log patterns...",
                "Detecting anomalies...",
                "Correlating threat indicators...",
                "Identifying potential compromises...",
                "Generating threat report..."
            ]
            
            for step in steps:
                task = progress.add_task(f"[red]{step}", total=1)
                tasks.append(task)
            
            # Process the logs
            result = self.chat_engine.threat_hunt(log_file)
            
            # Complete the progress bars with slight delays
            for i, task in enumerate(tasks):
                time.sleep(0.3)
                progress.update(task, advance=1)

        # Create a timeline of suspicious events
        console.print("\n[bold red]╔═══ THREAT HUNTING RESULTS ════════════════════════╗[/bold red]")
        console.print(f"[bold red]║[/bold red] [bold cyan]Log Source:[/bold cyan] {os.path.basename(log_file)}")
        console.print(f"[bold red]║[/bold red] [bold cyan]Hunt Timestamp:[/bold cyan] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"[bold red]║[/bold red] [bold cyan]Analysis Mode:[/bold cyan] {self.security_level.upper()}")
        console.print("[bold red]╠═══════════════════════════════════════════════════╣[/bold red]")
        
        # Display results
        console.print(Markdown(result))
        console.print("[bold red]╚═══════════════════════════════════════════════════╝[/bold red]")

    def malware_analysis(self, file_path=None):
        """Analyze potential malware samples."""
        if not file_path:
            file_path = Prompt.ask("[bold green]>[/bold green] File Path")
        
        # Check for sufficient security level
        if self.security_level not in ["advanced", "expert"]:
            console.print("[bold red][!] ERROR:[/bold red] Insufficient security level. Advanced or Expert required.")
            console.print("[bold yellow]Use 'set_level advanced' or 'set_level expert' to enable this feature.[/bold yellow]")
            return
        
        console.print(Text.from_markup("\n[bold red]==== MALWARE ANALYSIS ENGINE ====[/bold red]"))
        console.print(Text.from_markup("[bold yellow][blink]! CAUTION: SANDBOXED ENVIRONMENT REQUIRED ![/blink][/bold yellow]"))
        
        # More dramatic animation for malware analysis
        console.print("\n[bold red]Initializing secure analysis environment...[/bold red]")
        
        with Progress(
            SpinnerColumn("dots", style="red"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50, style="red", complete_style="green"),
            expand=False
        ) as progress:
            tasks = []
            
            steps = [
                "Initializing sandbox...",
                "Scanning file for known signatures...",
                "Performing static analysis...",
                "Running dynamic analysis...",
                "Monitoring behavior...",
                "Extracting IOCs...",
                "Generating malware profile..."
            ]
            
            for step in steps:
                task = progress.add_task(f"[red]{step}", total=1)
                tasks.append(task)
            
            # Process the malware analysis
            result = self.chat_engine.analyze_malware(file_path)
            
            # Complete the progress bars with slightly longer delays
            for i, task in enumerate(tasks):
                time.sleep(0.4)  # Longer delay for dramatic effect
                progress.update(task, advance=1)

        # Display results with cyberpunk aesthetics
        console.print("\n[bold red]╔═══ MALWARE ANALYSIS REPORT ═══════════════════════╗[/bold red]")
        console.print(f"[bold red]║[/bold red] [bold cyan]Sample:[/bold cyan] {os.path.basename(file_path)}")
        console.print(f"[bold red]║[/bold red] [bold cyan]MD5:[/bold cyan] {self._generate_hash()[:32]}")
        console.print(f"[bold red]║[/bold red] [bold cyan]SHA256:[/bold cyan] {self._generate_hash()}")
        console.print("[bold red]╠═══════════════════════════════════════════════════╣[/bold red]")
        
        # Create a structured view of malware details
        console.print(Markdown(result))
        console.print("[bold red]╚═══════════════════════════════════════════════════╝[/bold red]")

    # Offensive security methods - only available in offensive mode
    def recon(self, target=None):
        """Perform passive reconnaissance (OSINT)."""
        if self.security_mode != "offensive":
            console.print("[bold red][!] ERROR:[/bold red] Command only available in offensive mode.")
            console.print("[bold yellow]Use 'set_mode offensive' to enable this feature.[/bold yellow]")
            return
            
        if not target:
            target = Prompt.ask("[bold green]>[/bold green] Target (domain/organization)")
        
        # Check for sufficient auth level
        if self.auth_level not in ["government", "certified"]:
            console.print("[bold red][!] ERROR:[/bold red] Insufficient authorization level.")
            console.print("[bold yellow]Use 'set_auth government' or 'set_auth certified' to enable this feature.[/bold yellow]")
            return
        
        console.print(Text.from_markup("\n[bold red]==== RECONNAISSANCE ENGINE ====[/bold red]"))
        console.print(Text.from_markup("[bold yellow]NOTICE: AUTHORIZED USE ONLY[/bold yellow]"))
        
        # Animation for reconnaissance
        console.print("\n[bold cyan]Gathering intelligence...[/bold cyan]")
        
        with Progress(
            SpinnerColumn("dots", style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50, style="cyan", complete_style="green"),
            expand=False
        ) as progress:
            tasks = []
            
            steps = [
                "Querying passive DNS...",
                "Gathering domain information...",
                "Analyzing social media presence...",
                "Searching public repositories...",
                "Checking breach databases...",
                "Collating intelligence..."
            ]
            
            for step in steps:
                task = progress.add_task(f"[cyan]{step}", total=1)
                tasks.append(task)
            
            # Process the recon query
            result = self.chat_engine.perform_recon(target)
            
            # Complete the progress bars
            for i, task in enumerate(tasks):
                time.sleep(0.3)
                progress.update(task, advance=1)

        # Display results
        console.print("\n[bold cyan]╔═══ RECONNAISSANCE REPORT ═══════════════════════════╗[/bold cyan]")
        console.print(f"[bold cyan]║[/bold cyan] [bold white]Target:[/bold white] {target}")
        console.print(f"[bold cyan]║[/bold cyan] [bold white]Scope:[/bold white] Passive OSINT Collection")
        console.print(f"[bold cyan]║[/bold cyan] [bold white]Auth Level:[/bold white] {self.auth_level.upper()}")
        console.print("[bold cyan]╠═══════════════════════════════════════════════════╣[/bold cyan]")
        
        console.print(Markdown(result))
        console.print("[bold cyan]╚═══════════════════════════════════════════════════╝[/bold cyan]")

    def pentest_report(self, scope=None):
        """Generate penetration testing report."""
        if self.security_mode != "offensive":
            console.print("[bold red][!] ERROR:[/bold red] Command only available in offensive mode.")
            console.print("[bold yellow]Use 'set_mode offensive' to enable this feature.[/bold yellow]")
            return
            
        if not scope:
            scope = Prompt.ask("[bold green]>[/bold green] Scope (web/network/mobile/etc)")
        
        # Check for sufficient auth level
        if self.auth_level not in ["government", "certified"]:
            console.print("[bold red][!] ERROR:[/bold red] Insufficient authorization level.")
            console.print("[bold yellow]Use 'set_auth government' or 'set_auth certified' to enable this feature.[/bold yellow]")
            return
        
        console.print(Text.from_markup("\n[bold red]==== PENTEST REPORT GENERATOR ====[/bold red]"))
        
        # Get additional details
        target = Prompt.ask("[bold green]>[/bold green] Target Organization/System")
        findings = Prompt.ask("[bold green]>[/bold green] Number of Findings", default="5")
        
        # Animation for report generation
        console.print("\n[bold cyan]Generating penetration test report...[/bold cyan]")
        
        with Progress(
            SpinnerColumn("dots", style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50, style="cyan", complete_style="green"),
            expand=False
        ) as progress:
            tasks = []
            
            steps = [
                "Compiling findings...",
                "Calculating risk scores...",
                "Preparing executive summary...",
                "Generating technical details...",
                "Adding remediation advice...",
                "Finalizing report..."
            ]
            
            for step in steps:
                task = progress.add_task(f"[cyan]{step}", total=1)
                tasks.append(task)
            
            # Generate the report
            result = self.chat_engine.generate_pentest_report(scope, target, int(findings))
            
            # Complete the progress bars
            for i, task in enumerate(tasks):
                time.sleep(0.4)
                progress.update(task, advance=1)

        # Display results
        console.print("\n[bold cyan]╔═══ PENETRATION TEST REPORT ═══════════════════════╗[/bold cyan]")
        console.print(f"[bold cyan]║[/bold cyan] [bold white]Target:[/bold white] {target}")
        console.print(f"[bold cyan]║[/bold cyan] [bold white]Scope:[/bold white] {scope.upper()} Assessment")
        console.print(f"[bold cyan]║[/bold cyan] [bold white]Findings:[/bold white] {findings}")
        console.print(f"[bold cyan]║[/bold cyan] [bold white]Date:[/bold white] {datetime.now().strftime('%Y-%m-%d')}")
        console.print("[bold cyan]╠═══════════════════════════════════════════════════╣[/bold cyan]")
        
        console.print(Markdown(result))
        console.print("[bold cyan]╚═══════════════════════════════════════════════════╝[/bold cyan]")

    def exploit_analysis(self, cve=None):
        """Analyze exploit for CVE."""
        if self.security_mode != "offensive":
            console.print("[bold red][!] ERROR:[/bold red] Command only available in offensive mode.")
            console.print("[bold yellow]Use 'set_mode offensive' to enable this feature.[/bold yellow]")
            return
            
        if not cve:
            cve = Prompt.ask("[bold green]>[/bold green] CVE ID (e.g., CVE-2021-44228)")
        
        # Check for sufficient security level
        if self.security_level not in ["advanced", "expert"]:
            console.print("[bold red][!] ERROR:[/bold red] Insufficient security level. Advanced or Expert required.")
            console.print("[bold yellow]Use 'set_level advanced' or 'set_level expert' to enable this feature.[/bold yellow]")
            return
        
        console.print(Text.from_markup("\n[bold red]==== EXPLOIT ANALYSIS ENGINE ====[/bold red]"))
        console.print(Text.from_markup("[bold yellow]NOTICE: AUTHORIZED USE ONLY[/bold yellow]"))
        
        # Animation for exploit analysis
        console.print("\n[bold cyan]Analyzing exploit capabilities...[/bold cyan]")
        
        with Progress(
            SpinnerColumn("dots", style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50, style="cyan", complete_style="green"),
            expand=False
        ) as progress:
            tasks = []
            
            steps = [
                "Querying vulnerability databases...",
                "Retrieving exploit details...",
                "Analyzing exploit techniques...",
                "Assessing exploit reliability...",
                "Identifying target systems...",
                "Compiling analysis report..."
            ]
            
            for step in steps:
                task = progress.add_task(f"[cyan]{step}", total=1)
                tasks.append(task)
            
            # Process the exploit analysis
            result = self.chat_engine.analyze_exploit(cve)
            
            # Complete the progress bars
            for i, task in enumerate(tasks):
                time.sleep(0.3)
                progress.update(task, advance=1)

        # Display results
        console.print("\n[bold cyan]╔═══ EXPLOIT ANALYSIS REPORT ════════════════════════╗[/bold cyan]")
        console.print(f"[bold cyan]║[/bold cyan] [bold white]CVE ID:[/bold white] {cve}")
        console.print(f"[bold cyan]║[/bold cyan] [bold white]Analysis Level:[/bold white] {self.security_level.upper()}")
        console.print(f"[bold cyan]║[/bold cyan] [bold white]Auth Level:[/bold white] {self.auth_level.upper()}")
        console.print("[bold cyan]╠═══════════════════════════════════════════════════╣[/bold cyan]")
        
        console.print(Markdown(result))
        console.print("[bold cyan]╚═══════════════════════════════════════════════════╝[/bold cyan]")

    def display_matrix_effect(self, duration=3):
        """Display a brief Matrix-style digital rain animation."""
        console_width = os.get_terminal_size().columns
        rain_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]\\{}|;':\",./<>?" 
        streams = []
        for _ in range(console_width // 2):
           streams.append({
            'pos': 0, 
            'speed': random.uniform(0.1, 0.5), 
            'length': random.randint(5, 15),
            'column': random.randint(0, console_width-1),
            'last_update': 0
        })
        start_time = time.time()
        while time.time() - start_time < duration:
            os.system('cls' if os.name == 'nt' else 'clear')
            display = [" " * console_width for _ in range(20)]
            current_time = time.time()
            for stream in streams:
                if current_time - stream['last_update'] > stream['speed']:
                     stream['pos'] += 1
                     stream['last_update'] = current_time
            for i in range(stream['length']):
                row = int(stream['pos'] - i)
                if 0 <= row < 20:
                    if i == 0:
                        char = random.choice(rain_chars)
                        display_row = list(display[row])
                        display_row[stream['column']] = char
                        display[row] = ''.join(display_row)
                    else:
                        char = random.choice(rain_chars)
                        # Replace character at position with dim character
                        display_row = list(display[row])
                        display_row[stream['column']] = char
                        display[row] = ''.join(display_row)
            for line in display:
                console.print(Text.from_markup(f"[green]{line}[/green]"))
            time.sleep(0.05)                                
             
    def play_terminal_bell(self):
        """Play terminal bell for notifications."""
        print("\a", end="", flush=True)
    def notification(self, message, level="info"):
        """Display a notification with appropriate styling and optional sound."""
        level_styles = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "critical": "red reverse" }
        style = level_styles.get(level, "white")
        console.print(f"\n[bold {style}][!] {message}[/bold {style}]") 
        if level in ["warning", "error", "critical"]:              
            self.play_terminal_bell()
            
    def payload_gen(self, platform=None):
        """Generate test payload for platform."""
        if self.security_mode != "offensive":
            console.print("[bold red][!] ERROR:[/bold red] Command only available in offensive mode.")
            console.print("[bold yellow]Use 'set_mode offensive' to enable this feature.[/bold yellow]")
            return
            
        if not platform:
            platform = Prompt.ask("[bold green]>[/bold green] Platform (windows/linux/web/android)")
        
        # Check for sufficient security level and auth level
        if self.auth_level != "government":
            console.print("[bold red][!] ERROR:[/bold red] Insufficient authorization level. Government clearance required.")
            console.print("[bold yellow]Use 'set_auth government' to enable this feature.[/bold yellow]")
            return
            
        console.print(Text.from_markup("\n[bold red]==== TEST PAYLOAD GENERATOR ====[/bold red]"))
        console.print(Text.from_markup("[bold yellow][blink]! RESTRICTED OPERATION - GOVERNMENT CLEARANCE REQUIRED ![/blink][/bold yellow]"))
        
        # Get additional information
        payload_type = Prompt.ask("[bold green]>[/bold green] Payload Type (reverse_shell/bind_shell/web/etc)")
        purpose = Prompt.ask("[bold green]>[/bold green] Test Purpose")
        
        # Confirm operation with dramatic warning
        console.print(Panel(
            "[bold yellow]PAYLOAD GENERATION WARNING[/bold yellow]\n" +
            "[bold white]This operation will generate test security payloads for:[/bold white]\n" +
            f"- Platform: {platform}\n" +
            f"- Type: {payload_type}\n" +
            f"- Purpose: {purpose}\n\n" +
            "[bold red]ALL OPERATIONS ARE LOGGED FOR COMPLIANCE[/bold red]",
            title="[bold red][ SECURITY DIRECTIVE ][/bold red]",
            border_style="red", box=box.HEAVY_EDGE
        ))
        
        confirm = Confirm.ask("[bold red]Confirm payload generation[/bold red]")
        if not confirm:
            console.print("[bold yellow]Operation aborted.[/bold yellow]")
            return
        
        # Animation for payload generation
        console.print("\n[bold red]Generating security test payload...[/bold red]")
        
        with Progress(
            SpinnerColumn("dots", style="red"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50, style="red", complete_style="green"),
            expand=False
        ) as progress:
            tasks = []
            
            steps = [
                "Setting up payload environment...",
                "Creating payload structure...",
                "Implementing functionality...",
                "Encoding payload...",
                "Signing with test certificate...",
                "Finalizing payload..."
            ]
            
            for step in steps:
                task = progress.add_task(f"[red]{step}", total=1)
                tasks.append(task)
            
            # Generate the payload
            result = self.chat_engine.generate_test_payload(platform, payload_type, purpose)
            
            # Complete the progress bars
            for i, task in enumerate(tasks):
                time.sleep(0.4)
                progress.update(task, advance=1)

        # Display results
        console.print("\n[bold red]╔═══ TEST PAYLOAD DETAILS ═══════════════════════════╗[/bold red]")
        console.print(f"[bold red]║[/bold red] [bold white]Platform:[/bold white] {platform.upper()}")
        console.print(f"[bold red]║[/bold red] [bold white]Type:[/bold white] {payload_type}")
        console.print(f"[bold red]║[/bold red] [bold white]Auth Level:[/bold white] GOVERNMENT")
        console.print(f"[bold red]║[/bold red] [bold white]Hash:[/bold white] {self._generate_hash()[:16]}")
        console.print("[bold red]╠═══════════════════════════════════════════════════╣[/bold red]")
        
        console.print(Syntax(result, "python", theme="monokai", line_numbers=True))
        console.print("[bold red]╚═══════════════════════════════════════════════════╝[/bold red]")
        console.print("\n[bold yellow]WARNING: This payload is for authorized security testing only.[/bold yellow]")
    def run_hackbot(self):
        """Run HackBot in offensive security mode."""
        if self.security_mode != "offensive":
            console.print("[bold red]Error: HackBot only available in offensive mode[/bold red]")
            console.print("[yellow]Switch to offensive mode first with: set_mode offensive[/yellow]")
            return
        if self.auth_level not in ["government", "certified"]:
            console.print("[bold red]Error: Insufficient authorization level[/bold red]")
            console.print("[yellow]Elevate auth level with 'set_auth government' or 'set_auth certified'[/yellow]")
            return  


        try:    
            if self.hackbot is None:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Initializing HackBot..."),
                   transient=True
                   ) as progress:
                    progress.add_task("init", total=None)
                    self.hackbot = HackBot(config_file="hackbot_config.json")  # Use a dedicated config
            console.print("\n[bold red]╔══════════════════════════════════════════════════╗[/bold red]")
            console.print("[bold red]║[/bold red] [bold white]HACKBOT RUNNING IN OFFENSIVE SECURITY MODE[/bold white] [bold red]║[/bold red]")
            console.print("[bold red]╚══════════════════════════════════════════════════╝[/bold red]")
            original_banner = self.hackbot.show_banner
            def offensive_banner():
                console.print("\n[bold red]OFFENSIVE SECURITY MODE ACTIVE[/bold red]")
                console.print("[yellow]All activities are logged and monitored[/yellow]\n")
                original_banner()
            self.hackbot.show_banner = offensive_banner
            self.hackbot.run()
            self.hackbot.show_banner = original_banner
            console.print("[bold green]HackBot session finished successfully.[/bold green]")    
        except KeyboardInterrupt:
            console.print("\n[bold yellow]HackBot session interrupted by user.[/bold yellow]")   
        except Exception as e:
             console.print(f"[bold red]Failed to run HackBot: {str(e)}[/bold red]")
             self.hackbot = None
        finally: 
            self.hackbot = None   
    def enable_offensive_capabilities(self):
        """Forcefully enable offensive mode with elevated auth and expert level."""
        self.set_security_mode("offensive")
        self.set_auth_level("government")
        self.set_security_level("expert")
        console.print("[bold green]✓ Offensive capabilities enabled successfully[/bold green]")
                        
    def run(self):
        """Main method to run the security mode."""
        self.display_banner()
       
        self.setup_tab_completion()
        
        while True:
            try:
                # More hacker-style prompt
                cmd = Prompt.ask(f"\n[bold red][{self.security_mode}@{self.config.username}][/bold red][bold yellow]$[/bold yellow]")
                
                if cmd.lower() in ['exit', 'quit']:
                    # Dramatic exit
                    console.print("\n[bold red]Terminating secure session...[/bold red]")
                    with Progress(
                        SpinnerColumn("dots", style="red"),
                        TextColumn("[progress.description]{task.description}"),
                        transient=True
                    ) as progress:
                        task = progress.add_task("[red]Wiping session data...", total=1)
                        time.sleep(1)
                        progress.update(task, advance=1)
                    
                    console.print(f"[bold green]Session terminated. Duration: {datetime.now() - self.session_start}[/bold green]")
                    break
                    
                elif cmd.lower() == 'help':
                    self.show_help()
                    
                elif cmd.lower() == 'clear':
                    self.clear_screen()
                    
                elif cmd.lower() == 'stats':
                    self.show_stats()
                    
                elif cmd.lower() == 'switch':
                    console.print("[bold green]Switching to normal chat mode...[/bold green]")
                    return 'normal'
                    
                elif cmd.lower().startswith('set_level '):
                    level = cmd.split(' ')[1]
                    self.set_security_level(level)
                    
                elif cmd.lower().startswith('set_mode '):
                    mode = cmd.split(' ')[1]
                    self.set_security_mode(mode)
                    
                elif cmd.lower().startswith('set_auth '):
                    auth = cmd.split(' ')[1]
                    self.set_auth_level(auth)
                    
                elif cmd.lower() == 'static_code_analysis':
                    self.static_code_analysis()
                    
                elif cmd.lower() == 'vuln_analysis':
                    self.vulnerability_analysis()
                    
                elif cmd.lower().startswith('threat_hunt'):
                    parts = cmd.split(' ', 1)
                    log_file = parts[1] if len(parts) > 1 else None
                    self.threat_hunt(log_file)
                    
                elif cmd.lower().startswith('malware_analysis'):
                    parts = cmd.split(' ', 1)
                    file_path = parts[1] if len(parts) > 1 else None
                    self.malware_analysis(file_path)
                    
                elif cmd.lower().startswith('recon'):
                    parts = cmd.split(' ', 1)
                    target = parts[1] if len(parts) > 1 else None
                    self.recon(target)
                    
                elif cmd.lower().startswith('pentest_report'):
                    parts = cmd.split(' ', 1)
                    scope = parts[1] if len(parts) > 1 else None
                    self.pentest_report(scope)
                    
                elif cmd.lower().startswith('exploit_analysis'):
                    parts = cmd.split(' ', 1)
                    cve = parts[1] if len(parts) > 1 else None
                    self.exploit_analysis(cve)
                    
                elif cmd.lower().startswith('payload_gen'):
                    parts = cmd.split(' ', 1)
                    platform = parts[1] if len(parts) > 1 else None
                    self.payload_gen(platform)
                
                elif cmd.lower() == 'run_hackbot':
                    if self.security_mode != "offensive":
                        console.print("[bold red]Error: Must be in offensive mode to run HackBot[/bold red]")
                        console.print("[yellow]Use 'set_mode offensive' first[/yellow]")
                    else:
                        if self.auth_level not in ["government", "certified"]:
                            console.print("[bold red]Error: Insufficient authorization level[/bold red]")
                            console.print("[yellow]Elevate auth level with 'set_auth government'[/yellow]")
                        else:
                            self.run_hackbot()
                    
                else:
                    # Process as security query
                    self.process_security_query(cmd)
                    
                    
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Operation aborted. Type 'exit' to quit.[/bold yellow]")
                
            except Exception as e:
                console.print(f"[bold red][!] ERROR:[/bold red] {str(e)}")
