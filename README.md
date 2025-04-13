# SHARVA - AI Assistant

**S**ystematic **H**igh-performance **A**rtificial **R**esponse **V**irtual **A**gent

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-brightgreen.svg)](https://www.python.org/downloads/)

## Overview

SHARVA is an advanced AI assistant with dual functionality - a general-purpose mode for everyday tasks and a specialized security mode for cybersecurity professionals. It leverages large language models through Groq's API to provide intelligent, context-aware responses with real-time search capabilities.

![SHARVA Banner](https://via.placeholder.com/800x200?text=SHARVA+AI+Assistant)

## Features

### Core Capabilities
- **Dual Mode Operation**:
  - **Normal Mode**: General assistance for everyday queries
  - **Security Mode**: Specialized cybersecurity expertise
- **Real-time Search**: Integration with Google search for up-to-date information
- **Conversation Management**: Save, load, and export chat sessions
- **Customizable UI**: Theme selection and terminal-based UI

### Security Features
- **Code Analysis**: Static code security analysis
- **Vulnerability Assessment**: Analysis of scan results
- **Reconnaissance Engine**: Domain/IP intelligence gathering with:
  - DNS enumeration
  - WHOIS lookup
  - SSL certificate analysis
  - Port scanning (simulated)
  - Shodan and VirusTotal integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/sharva.git
cd sharva
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables by creating a `.env` file:
```
Username=Your Name
Assistantname=Sharva
GroqAPIKey=your_groq_api_key
ShodanAPIKey=your_shodan_api_key
VirusTotalAPIKey=your_virustotal_api_key
```

## Usage

### Starting the Application

```bash
python main.py
```

#### Command Line Options

```bash
python main.py --mode [normal|security] --local --search --theme [dark|light|cyberpunk|minimal]
```

- `--mode`: Start in normal or security mode
- `--local`: Force using local LLM models when available
- `--search`: Enable search capabilities by default
- `--theme`: Select UI theme

### Available Commands

#### General Commands
- `quit` or `exit`: Exit the application
- `clear`: Clear the screen (or use Ctrl+L)
- `help`: Show the help menu
- `switch`: Switch between normal and security modes

#### Search & Session Management
- `search <query>`: Perform a search for information
- `save`: Save the current session
- `load <session_id>`: Load a previous session
- `sessions`: List all saved sessions
- `export <filename>`: Export conversation to file

#### Customization
- `theme <theme_name>`: Change color theme (default, dark, light)
- `keyboard`: Show keyboard shortcuts

### Keyboard Shortcuts
- `Tab`: Autocomplete commands
- `↑/↓`: Navigate command history
- `Ctrl+L`: Clear screen
- `Ctrl+S`: Save current session
- `Ctrl+C`: Cancel current operation

## Components

### Main Classes

- `ChatEngine`: Core NLP processing with Groq API integration
- `NormalMode`: General-purpose assistant interface
- `SecurityMode`: Specialized security-focused interface
- `Session`: Conversation management

### Reconnaissance Engine

The built-in recon engine can gather information about domains and IP addresses including:

- DNS records and resolution
- WHOIS domain registration details
- SSL certificate information
- Network service enumeration
- Public databases (Shodan, VirusTotal)
- Repository mentions
- Breach database checks

## Requirements

- Python 3.7+
- Rich library for terminal UI
- Groq API access
- Additional API keys for advanced reconnaissance (optional):
  - Shodan
  - VirusTotal

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Developed by Sambhav Mehra
- Powered by Groq API for language processing
- Special thanks to all contributors and testers

---

**Note**: SHARVA is designed for ethical use only. The security features should only be used on systems and networks you have explicit permission to test.
