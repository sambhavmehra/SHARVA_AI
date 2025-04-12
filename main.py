#!/usr/bin/env python3
import os
import sys
import argparse
import time
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text
from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from normal_mode import NormalMode
from security_mode import SecurityMode
from config import Config

console = Console()

def display_animated_banner():
    """Display an animated application banner."""
    frames = [
        r"""
   _____       __  __       ___        ____     _    __       ___ 
  / ___/      / / / /      /   |      / __ \   | |  / /      /   |
  \__ \      / /_/ /      / /| |     / /_/ /   | | / /      / /| |
 ___/ /  _  / __  /   _  / ___ | _  / _, _/  _ | |/ /   _  / ___ |
/____/  (_)/_/ /_/   (_)/_/  |_|(_)/_/ |_|  (_)|___/   (_)/_/  |_|
                                                                  
                                                     
        """,
        r"""   
   _____       __  __       ___        ____     _    __       ___ 
  / ___/      / / / /      /   |      / __ \   | |  / /      /   |
  \__ \      / /_/ /      / /| |     / /_/ /   | | / /      / /| |
 ___/ /  _  / __  /   _  / ___ | _  / _, _/  _ | |/ /   _  / ___ |
/____/  (_)/_/ /_/   (_)/_/  |_|(_)/_/ |_|  (_)|___/   (_)/_/  |_|
                                                                  
        """
    ]

    # Display loading animation
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Loading S.H.A.R.V.A. AI Assistant...[/bold green]"),
        transient=True,
    ) as progress:
        task = progress.add_task("Loading...", total=10)
        for i in range(10):
            progress.update(task, advance=1)
            time.sleep(0.1)

    # Animate ASCII logo
    for _ in range(3):
        for frame in frames:
            console.clear()
            console.print(Panel(
                Text(frame, style="bold cyan"),
                title="[bold green]S.H.A.R.V.A. AI Assistant by Sambhav Mehra[/bold green]",
                subtitle="[italic]Systematic High-performance Artificial Response Virtual Agent[/italic]",
                border_style="green",
                box=box.DOUBLE
            ))
            time.sleep(0.25)

    # System information panel
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", "1.2.0")
    table.add_row("Developer", "Sambhav Mehra")
    

    console.print(Panel(
        table,
        title="[bold blue]System Information[/bold blue]",
        border_style="blue",
        box=box.ROUNDED
    ))

    # Final message with full form
    console.print("\n[bold green]S.H.A.R.V.A. AI Assistant initialized successfully![/bold green]")
    console.print("[italic]Your intelligent companion for both general assistance and security expertise[/italic]\n")
    console.print("[bold yellow][/bold yellow]")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SHARVA Advanced AI Chat Assistant")
    parser.add_argument("--mode", choices=["normal", "security"],
                        help="Starting mode (normal or security)")
    parser.add_argument("--local", action="store_true", 
                        help="Force using local LLM models when available")
    parser.add_argument("--search", action="store_true", 
                        help="Enable search capabilities by default")
    parser.add_argument("--theme", choices=["dark", "light", "cyberpunk", "minimal"],
                        help="UI theme to use")
    return parser.parse_args()

def check_environment():
    """Check environment and dependencies with visual feedback."""
    config = Config()
    
    with console.status("[bold green]Checking system configuration...[/bold green]"):
        time.sleep(0.5)  # Simulate checking process
    
    # Check for API keys
    warnings = []
    critical_issues = []
    successes = []
    
    if config.groq_api_key:
        successes.append("✅ Groq API key detected - Online features enabled")
    else:
        warnings.append("⚠️ Groq API key not found - Online API features will be disabled")
    
    if config.runpod_endpoint_id and config.runpod_api_key:
        successes.append("✅ RunPod configuration detected - Local model features available")
    else:
        warnings.append("⚠️ RunPod configuration missing - Local model features will be limited")
    
    # Check for optional dependencies
    try:
        import requests
        successes.append("✅ Network libraries available - Web connectivity enabled")
    except ImportError:
        critical_issues.append("❌ Required network libraries missing - Install 'requests' package")
    
    # Display check results
    if successes:
        console.print(Panel("\n".join(successes), title="[bold green]✅ System Checks Passed[/bold green]", 
                           border_style="green", box=box.ROUNDED))
    
    if warnings:
        console.print(Panel("\n".join(warnings), title="[bold yellow]⚠️ Configuration Warnings[/bold yellow]", 
                           border_style="yellow", box=box.ROUNDED))
    
    if critical_issues:
        console.print(Panel("\n".join(critical_issues), title="[bold red]❌ Critical Issues[/bold red]", 
                           border_style="red", box=box.ROUNDED))
    
    return config

def display_mode_selection():
    """Display an interactive mode selection menu."""
    table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD)
    table.add_column("Mode", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Recommended For", style="yellow")
    
    table.add_row(
        "Normal", 
        "General-purpose assistant for everyday tasks", 
        "Regular queries, information, and assistance"
    )
    table.add_row(
        "Security", 
        "Specialized security-focused assistant", 
        "Cybersecurity, technical analysis, and secure operations"
    )
    
    console.print(Panel(
        table,
        title="[bold green]SHARVA Mode Selection[/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))
    
    mode = Prompt.ask(
        "[bold]Select operating mode[/bold]",
        choices=["normal", "security"],
        default="normal"
    )
    
    return mode

def main():
    """Main application entry point with enhanced UI."""
    args = parse_arguments()
    
    # Clear screen for a clean start
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Show animated banner
    display_animated_banner()
    
    config = check_environment()
    
    # Override config with command line arguments
    if args.local:
        config.force_local = True
        console.print("[bold blue]Local mode enforced by command line argument[/bold blue]")
    
    if args.search:
        config.enable_search = True
        console.print("[bold blue]Search capabilities enabled by default[/bold blue]")
    
    # Custom theme handling if implemented
    if args.theme:
        console.print(f"[bold blue]Applied '{args.theme}' theme to interface[/bold blue]")
    
    # Determine which mode to start in
    if args.mode:
        current_mode = args.mode.lower()
        console.print(f"[bold blue]Starting in {current_mode.capitalize()} mode as specified[/bold blue]")
    else:
        # Interactive mode selection
        current_mode = display_mode_selection()
    
    console.print(f"\n[bold green]SHARVA AI Assistant initialized in {current_mode.capitalize()} mode![/bold green]")
    console.print("[italic]Type 'help' for available commands at any time[/italic]\n")
    
    # Initialize the appropriate mode
    while True:
        try:
            if current_mode == "normal":
                mode = NormalMode(config)
                exit_code = mode.run()
            elif current_mode == "security":
                mode = SecurityMode(config)
                mode.enable_offensive_capabilities()  # ✅ Unlocks offensive mode, expert level, and government access
                exit_code = mode.run()
            else:
                console.print("[bold red]Invalid mode selected![/bold red]")
                exit_code = "exit"
            
            # Handle exit codes
            if exit_code == "exit" or exit_code == "quit":
                console.print(Panel(
                    "[bold]Thank you for using SHARVA AI Assistant![/bold]\n\n[italic]Developed by Sambhav Mehra[/italic]",
                    title="[bold green]Goodbye![/bold green]",
                    border_style="green",
                    box=box.ROUNDED
                ))
                break
            elif exit_code == "normal":
                current_mode = "normal"
                console.print("[bold green]Switching to Normal Mode...[/bold green]")
            elif exit_code == "security":
                current_mode = "security"
                console.print("[bold blue]Switching to Security Mode...[/bold blue]")
            else:
                console.print("[bold red]Unknown exit code. Exiting...[/bold red]")
                break
        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]Program interrupted. Use 'exit' to quit properly next time.[/bold yellow]")
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Program interrupted by user. Exiting...[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        console.print("[bold yellow]Please report this issue to the developer: Sambhav Mehra[/bold yellow]")
        sys.exit(1)