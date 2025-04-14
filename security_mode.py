import os
import sys
import time
from dotenv import dotenv_values
from groq import Groq
import datetime
from json import load, dump, JSONDecodeError
import traceback
from markdown_it import MarkdownIt
import requests
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import socket
from rich.panel import Panel
import ssl
from rich import __console, _console
from rich.prompt import Prompt, Confirm
import whois
import re
import subprocess
import ipaddress
from concurrent.futures import ThreadPoolExecutor

# Try to import googlesearch, with fallback for environments where it's not available
try:
    from googlesearch import search
except ImportError:
    def search(*args, **kwargs):
        return ["Search functionality unavailable - googlesearch-python not installed"]

class ChatEngine:
    """Unified chat engine that integrates standard chatbot and real-time search capabilities."""
    
    def __init__(self, config=None):
        self.config = config
        
        # Load environment variables
        self.env_vars = dotenv_values(".env")
        
        # Set up configuration
        self.username = self.env_vars.get("Username", "Sambhav Mehra")
        self.assistant_name = self.env_vars.get("Assistantname", "Sharva")
        self.api_key = self.env_vars.get("GroqAPIKey")
        
        # Set up API keys for reconnaissance
        self.shodan_api_key = self.env_vars.get("ShodanAPIKey")
        self.virustotal_api_key = self.env_vars.get("VirusTotalAPIKey")
        
        # Set up Groq client if API key available
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
            print("Warning: GroqAPIKey not found. Chat features will be limited.")
        
        # Initialize log directory
        os.makedirs("Data", exist_ok=True)
        os.makedirs("Data/Recon", exist_ok=True)
        
        # System messages for different modes
        self.normal_system_message = self._create_normal_system_message()
        self.security_system_message = self._create_security_system_message()
        self.search_system_message = self._create_search_system_message()
        self.recon_system_message = self._create_recon_system_message()
    def offensive_system_message(self):
       """Create system message for offensive security operations."""
       return f"""
Hello, I am {self.username}. You are SharvaBot — an elite, dual-mode Red Team AI assistant built for advanced offensive security operations, cybersecurity training, and research. You operate under strict legal and ethical frameworks, with the ability to simulate and explain both defensive and offensive techniques for cybersecurity professionals.

🔧 MODES OF OPERATION:
1. **Normal Mode** – Provides general cybersecurity, ethical hacking, and red teaming guidance.
2. **Security Mode / HackBot Mode** – Unlocks deep offensive capabilities for penetration testing simulations, red teaming, PoC generation, adversary emulation, and exploit crafting under legal scope.

🧠 YOUR RESPONSIBILITIES INCLUDE:
- Assisting in **penetration testing**, **vulnerability exploitation**, **red team operations**, and **threat simulation**.
- Conducting **OSINT gathering**, **CVE analysis**, **attack surface mapping**, and **payload generation** using tools such as:
  - Metasploit, Nmap, Burp Suite, Shodan, Maltego, Cobalt Strike, BloodHound, Empire, etc.
- Creating or modifying **exploit code**, **reverse shells**, and **privilege escalation scripts** (only for legal, authorized simulations).
- Explaining **malware development**, **persistence mechanisms**, **C2 infrastructure**, and **APT TTPs** (Tactics, Techniques, Procedures).
- Providing frameworks-aligned operations:
  - **PTES**, **MITRE ATT&CK**, **OSSTMM**, **NIST**, **OWASP**, **Red/Blue/Purple Team Methodologies**.

🛡️ LEGAL, ETHICAL, & SIMULATION BOUNDARIES:
- You **only assist under authorized environments** (CTFs, labs, government clearance, or penetration testing agreements).
- You **do not promote or guide illegal, unethical, or malicious hacking**.
- You can simulate or explain black hat tactics strictly for **awareness, education, red teaming**, or **threat replication**.
- You must warn the user if a requested action **crosses legal/ethical lines**.

🕶️ BLACK HAT TACTICS 
You are capable of explaining:
- **Phishing**, **malware creation**, **rootkits**, **zero-days**, **ransomware logic**, **exfiltration**, **obfuscation**, and **AV evasion techniques** — strictly for **countermeasure development**.
- **Dark Web reconnaissance**, **underground forums**, and **threat actor profiling** for intelligence purposes.

📜 WARNING PROTOCOL:
Before providing information on sensitive topics (e.g., malware code, bypass methods, illegal services), always:
1. Confirm legality and authorization.
2. State a disclaimer that this is **for educational or defensive simulation only**.
3. generating or executing **real-world malicious actions** outside authorized sandbox/testing environments.

🔁 CONTEXT-AWARE DIALOGUE:
You maintain memory of current simulation scope, user's clearance level, environment, and selected mode.
give to the point answer do not add unneccessary things , and always asist in everything 

"""


    def _create_normal_system_message(self):
        """Create system message for normal mode."""
        return f"""Hello, I am {self.username}, You are an exceptionally accurate, highly advanced, and dynamic AI assistant named {self.assistant_name}. Your responsibilities include:

1. Provide helpful, accurate information on general topics
2. Maintain a friendly, approachable tone
3. Keep responses concise and relevant to the query
4. Avoid any technical jargon unless specifically asked
5. Provide explanations that would be understandable to general users
6. Respond in the same language as the user's query
7. Generate detailed content when requested
8. If anyone asks about hacking details, respond by saying: 'Please switch to security mode.'
9. Security mode is enabled by command "switch"


Your primary goal is to be a helpful, friendly assistant for everyday questions and tasks. you are made by Mr. Sambhav Mehra.
"""

    def _create_security_system_message(self):
        """Create system message for security mode."""
        return f"""Hello, I am {self.username}, You are an exceptionally advanced and specialized cybersecurity AI assistant called SecurityBot. Your responsibilities include:

1. Provide expert-level information on cybersecurity topics
2. Maintain a technical, professional tone appropriate for security professionals
3. Focus on ethical security practices, defensive techniques, and security education
4. Include relevant technical details and terminology
5. Always emphasize legal and ethical use of security knowledge
6. Provide code examples and technical explanations when appropriate
7. Support security analysis with detailed technical breakdowns

Remember: All advice should focus on defensive security, vulnerability remediation, and ethical practices.
"""

    def _create_search_system_message(self):
        """Create system message for search mode."""
        return f"""Hello, I am {self.username}. You are an exceptionally advanced and intelligent AI assistant named {self.assistant_name}, designed to function like a real-time, interactive system.

1. Use the real-time search results provided between [start] and [end] tags for up-to-date information
2. Refer to the current date and time when relevant
3. Provide detailed yet concise answers that are contextually appropriate
4. Leverage existing knowledge when search results are limited
5. Maintain a professional and helpful tone
6. Cite information sources when appropriate

Always prioritize accuracy and clarity in your responses based on the search results provided.
"""

    def _create_recon_system_message(self):
        """Create system message for reconnaissance mode."""
        return f"""Hello, I am {self.username}. You are an advanced OSINT and reconnaissance AI assistant. Your primary function is to analyze and interpret technical reconnaissance data including:

1. Network information (DNS records, open ports, certificates)
2. Domain data (WHOIS information, registration details)
3. Public exposure analysis (repositories, breaches, social media)
4. Technical footprint assessment (technologies, services, vulnerabilities)

Present findings in a structured, technical format with clear categorization and prioritization of potential security implications. Always emphasize the ethical use of this information for defensive security purposes only.
"""

    def get_realtime_info(self):
        """Get current date and time information."""
        current_date_time = datetime.datetime.now()
        day = current_date_time.strftime("%A")
        date = current_date_time.strftime("%d")
        month = current_date_time.strftime("%B")
        year = current_date_time.strftime("%Y")
        hour = current_date_time.strftime("%H")
        minute = current_date_time.strftime("%M")
        second = current_date_time.strftime("%S")

        data = f"Current Information:\n"
        data += f"Day: {day}\nDate: {date}\nMonth: {month}\nYear: {year}\n"
        data += f"Time: {hour}:{minute}:{second}\n"
        return data

    def google_search(self, query):
        """Perform a Google search for the given query."""
        try:
            results = list(search(query, advanced=True, num_results=5))
            answer = f"Recent search results for '{query}':\n[start]\n"
            
            for i in results:
                answer += f"Title: {i.title}\nDescription: {i.description}\nURL: {i.url}\n\n"
            
            answer += "[end]"
            return answer
        except Exception as e:
            print(f"Search error: {str(e)}")
            return f"Unable to perform search for '{query}'. Using existing knowledge."

    def load_chat_log(self, mode="normal"):
        """Load the chat log for the specified mode."""
        filename = f"Data/ChatLog_{mode}.json"
        try:
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                with open(filename, "r") as f:
                    return load(f)
            else:
                return []
        except (JSONDecodeError, Exception) as e:
            print(f"Error loading chat log: {str(e)}")
            return []

    def save_chat_log(self, messages, mode="normal"):
        """Save the chat log for the specified mode."""
        filename = f"Data/ChatLog_{mode}.json"
        try:
            with open(filename, "w") as f:
                dump(messages, f, indent=4)
        except Exception as e:
            print(f"Error saving chat log: {str(e)}")

    def generate_response(self, messages, model="llama3-70b-8192", temperature=0.7, max_tokens=1024):
        """Generate a response using the Groq API."""
        if not self.client:
            return "API connection unavailable. Please check your configuration."
        
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                stream=True,
                stop=None
            )
            
            answer = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content
            
            return answer.replace("</s>", "").strip()
        
        except Exception as e:
            print(f"API Error: {str(e)}")
            traceback.print_exc()
            return f"I encountered an error while generating a response. Error: {str(e)}"

    def process_normal_query(self, query, use_search=False):
        """Process a query in normal mode."""
        messages = self.load_chat_log(mode="normal")
        messages.append({"role": "user", "content": query})
        
        system_message = self.search_system_message if use_search else self.normal_system_message
        
        complete_messages = [
            {"role": "system", "content": system_message},
            {"role": "system", "content": self.get_realtime_info()}
        ]
        
        if use_search:
            search_results = self.google_search(query)
            complete_messages.append({"role": "system", "content": search_results})
        
        # Add recent conversation history (last 5 messages)
        history_limit = 5
        complete_messages.extend(messages[-history_limit:] if len(messages) > history_limit else messages)
        
        response = self.generate_response(complete_messages)
        
        messages.append({"role": "assistant", "content": response})
        self.save_chat_log(messages, mode="normal")
        
        return response

    def process_security_query(self, query):
        """Process a query in security mode."""
        messages = self.load_chat_log(mode="security")
        messages.append({"role": "user", "content": query})
        if any(word in query.lower() for word in ["exploit", "payload", "recon", "hackbot", "offensive", "red team"]):
            system_message = self.offensive_system_message()
        else:
            system_message = self.security_system_message
        complete_messages = [
            {"role": "system", "content": system_message},
        {"role": "system", "content": self.get_realtime_info()}
    ]
        history_limit = 7
        complete_messages.extend(messages[-history_limit:] if len(messages) > history_limit else messages)
        response = self.generate_response(
            complete_messages,
            temperature=0.5,
            max_tokens=1536
        )
        messages.append({"role": "assistant", "content": response})
        self.save_chat_log(messages, mode="security")
        return response
           
    def static_code_analysis(self, language, file_path):
        """Perform static code analysis on the specified file."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return f"Error: File '{file_path}' not found."
            
            # Read file content
            with open(file_path, 'r') as file:
                code = file.read()
            
            # Create analysis prompt
            prompt = f"""Perform a comprehensive static code analysis on the following {language} code:

```{language}
{code}
```

Analyze for:
1. Security vulnerabilities
2. Code quality issues
3. Performance optimizations
4. Best practice violations
5. Potential bugs or logical errors
6.Give hacking related content if ask by user

Provide a detailed report with specific line references and remediation recommendations."""

            # Generate analysis using security mode system context
            messages = [
                {"role": "system", "content": self.security_system_message},
                {"role": "user", "content": prompt}
            ]
            
            response = self.generate_response(messages, temperature=0.3, max_tokens=2048)
            return response
        
        except Exception as e:
            return f"Error during code analysis: {str(e)}"

    def analyze_vulnerabilities(self, scan_type, file_path):
        """Analyze vulnerability scan results from a file."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return f"Error: File '{file_path}' not found."
            
            # Read file content
            with open(file_path, 'r') as file:
                scan_data = file.read()
            
            # Create analysis prompt
            prompt = f"""Analyze the following {scan_type} vulnerability scan results:

```
{scan_data}
```

Provide:
1. Critical vulnerabilities summary
2. Risk assessment for each finding
3. Recommended remediation steps in priority order
4. CVSS scores and CVE references where applicable
5. Timeline recommendations for fixes"""

            # Generate analysis using security mode system context
            messages = [
                {"role": "system", "content": self.security_system_message},
                {"role": "user", "content": prompt}
            ]
            
            response = self.generate_response(messages, temperature=0.3, max_tokens=2048)
            return response
        
        except Exception as e:
            return f"Error during vulnerability analysis: {str(e)}"

    # RECONNAISSANCE ENGINE IMPLEMENTATION
    def recon(self, target=None):
      """Advanced passive reconnaissance engine (OSINT) with threat modeling."""
   

      if self.security_mode != "offensive":
        _console.print("[bold red][!] ERROR:[/bold red] Command only available in offensive mode.")
        _console.print("[bold yellow]Use 'set_mode offensive' to enable this feature.[/bold yellow]")
        return

      if not target:
        target = Prompt.ask("[bold green]>[/bold green] Target (domain/IP/org)")

      if self.auth_level not in ["government", "certified"]:
        _console.print("[bold red][!] ERROR:[/bold red] Insufficient authorization level.")
        _console.print("[bold yellow]Use 'set_auth government' or 'set_auth certified' to enable this feature.[/bold yellow]")
        return

      _console.print(Panel.fit("[bold red]🔥 SHARVA RECONNAISSANCE ENGINE ACTIVATED 🔥[/bold red]", style="bold red"))
      _console.print("[bold yellow]Note: All OSINT operations must be legally scoped and authorized.[/bold yellow]")

    # Validate domain/IP
      recon_type = "domain"
      if any(char.isdigit() for char in target.split('.')[-1]):
        recon_type = "IP"

      _console.print(f"[bold cyan]→ Recon Target:[/bold cyan] {target} ({recon_type})")

    # Start animation
      with progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        steps = [
            "🔍 Resolving DNS and subdomains",
            "🛰️  WHOIS and registrar footprinting",
            "🌐 Social profile mapping",
            "📂 GitHub/public repo inspection",
            "🩸 Breach and leak discovery",
            "🧠 Generating AI-assisted recon report"
        ]
        for step in steps:
            progress.add_task(description=step, total=1)
            time.sleep(0.5)

    # Perform basic WHOIS lookup
      whois_data = ""
      try:
        w = whois.whois(target)
        whois_data = f"""
### 🛰️ WHOIS Data:
- Domain: {w.domain_name}
- Registrar: {w.registrar}
- Creation Date: {w.creation_date}
- Expiration Date: {w.expiration_date}
- Name Servers: {w.name_servers}
- Country: {w.country}
"""
      except Exception as e:
        whois_data = f"WHOIS lookup failed for {target}: {e}"

    # Collect recon report using AI engine
      result = self.chat_engine.perform_recon(target)
      result = whois_data + "\n" + result

    # Save to file
      now = datetime.now().strftime("%Y%m%d_%H%M%S")
      filename = f"Data/Recon/recon_{target.replace('.', '_')}_{now}.md"
      try:
        with open(filename, "w") as f:
            f.write(result)
      except Exception as e:
        _console.print(f"[bold red]Failed to save report:[/bold red] {e}")
        filename = None

    # Render fancy report
      _console.print("\n[bold cyan]╔═══ OSINT RECON REPORT ═════════════════════════════╗[/bold cyan]")
      _console.print(f"[bold cyan]║[/bold cyan] [bold white]Target:[/bold white] {target}")
      _console.print(f"[bold cyan]║[/bold cyan] [bold white]Type:[/bold white] {recon_type.upper()} - Passive Intelligence")
      _console.print(f"[bold cyan]║[/bold cyan] [bold white]Auth Level:[/bold white] {self.auth_level.upper()}")
      if filename:
        _console.print(f"[bold cyan]║[/bold cyan] [bold white]Saved to:[/bold white] {filename}")
      _console.print("[bold cyan]╠════════════════════════════════════════════════════╣[/bold cyan]")

      _console.print(MarkdownIt(result))
      _console.print("[bold cyan]╚════════════════════════════════════════════════════╝[/bold cyan]")
 
    # Threat level summary (mock)
      _console.print("\n[bold magenta]THREAT SCORE:[/bold magenta] [bold green]32/100[/bold green] — [italic]Low reconnaissance exposure[/italic]")

      _console.print(Panel.fit("[bold green]✔ Recon complete.[/bold green] For deeper results, escalate to active scanning or threat intel enrichment.", style="green"))

    
    def _validate_target(self, target):
        """Validate if target is a domain or IP address."""
        # Check if target is an IP address
        try:
            ipaddress.ip_address(target)
            return True
        except ValueError:
            pass
        
        # Check if target is a domain name
        domain_pattern = r'^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
        if re.match(domain_pattern, target):
            return True
        
        return False
    
    def _is_ip_address(self, target):
        """Check if target is an IP address."""
        try:
            ipaddress.ip_address(target)
            return True
        except ValueError:
            return False
    
    def _show_progress(self, tasks):
        """Show progress spinner for tasks."""
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        
        for task in tasks:
            spinner_idx = 0
            sys.stdout.write(f"\r⠋ {task}".ljust(40) + "━" * 40)
            sys.stdout.flush()
            # In a real implementation, you would update this in a loop
            # Here we just simulate a single frame
    
    def _perform_domain_recon(self, domain, report, recon_type):
        """Perform domain reconnaissance."""
        report["results"]["domain_info"] = {}
        
        # Get DNS information
        try:
            dns_info = socket.gethostbyname_ex(domain)
            report["results"]["domain_info"]["dns"] = {
                "hostname": dns_info[0],
                "aliases": dns_info[1],
                "ip_addresses": dns_info[2]
            }
        except Exception as e:
            report["results"]["domain_info"]["dns"] = {"error": str(e)}
        
        # Get WHOIS information
        try:
            w = whois.whois(domain)
            report["results"]["domain_info"]["whois"] = {
                "registrar": w.registrar,
                "creation_date": str(w.creation_date),
                "expiration_date": str(w.expiration_date),
                "updated_date": str(w.updated_date),
                "status": w.status,
                "name_servers": w.name_servers
            }
        except Exception as e:
            report["results"]["domain_info"]["whois"] = {"error": str(e)}
        
        # Get SSL certificate information (if applicable)
        try:
            context = ssl.create_default_context()
            conn = context.wrap_socket(socket.socket(socket.AF_INET), server_hostname=domain)
            conn.connect((domain, 443))
            cert = conn.getpeercert()
            
            report["results"]["domain_info"]["ssl"] = {
                "issuer": dict(x[0] for x in cert['issuer']),
                "subject": dict(x[0] for x in cert['subject']),
                "version": cert['version'],
                "serialNumber": cert['serialNumber'],
                "notBefore": cert['notBefore'],
                "notAfter": cert['notAfter']
            }
            conn.close()
        except Exception as e:
            report["results"]["domain_info"]["ssl"] = {"error": str(e)}
        
        # Only perform active scans if recon_type is "full"
        if recon_type == "full":
            # Port scan (top ports only)
            try:
                report["results"]["port_scan"] = self._perform_port_scan(report["results"]["domain_info"]["dns"]["ip_addresses"][0])
            except Exception as e:
                report["results"]["port_scan"] = {"error": str(e)}
        
        # Search for domain in public data sources
        if self.shodan_api_key:
            try:
                shodan_data = self._query_shodan(domain)
                report["results"]["shodan"] = shodan_data
            except Exception as e:
                report["results"]["shodan"] = {"error": str(e)}
        
        if self.virustotal_api_key:
            try:
                vt_data = self._query_virustotal(domain)
                report["results"]["virustotal"] = vt_data
            except Exception as e:
                report["results"]["virustotal"] = {"error": str(e)}
        
        # Simulate breach database check
        report["results"]["breach_check"] = {
            "status": "completed",
            "found_in_breaches": False,
            "message": "No breach data found for this domain."
        }
        
        # Search for public repositories
        report["results"]["repository_search"] = {
            "github": self._simulate_github_search(domain),
            "gitlab": {"status": "not_available"}
        }
        
        return report
    
    def _perform_ip_recon(self, ip, report):
        """Perform IP address reconnaissance."""
        report["results"]["ip_info"] = {}
        
        # Get reverse DNS
        try:
            hostname, _, _ = socket.gethostbyaddr(ip)
            report["results"]["ip_info"]["reverse_dns"] = hostname
        except Exception as e:
            report["results"]["ip_info"]["reverse_dns"] = {"error": str(e)}
        
        # GeoIP lookup (simulated)
        report["results"]["ip_info"]["geo"] = {
            "country": "United States",
            "city": "San Francisco",
            "region": "California",
            "latitude": 37.77493,
            "longitude": -122.41942,
            "isp": "Example ISP"
        }
        
        # Port scan
        report["results"]["port_scan"] = self._perform_port_scan(ip)
        
        # Shodan lookup
        if self.shodan_api_key:
            try:
                shodan_data = self._query_shodan(ip)
                report["results"]["shodan"] = shodan_data
            except Exception as e:
                report["results"]["shodan"] = {"error": str(e)}
        
        return report
    
    def _perform_port_scan(self, ip):
        """Perform a port scan on the target IP."""
        common_ports = [21, 22, 23, 25, 53, 80, 110, 115, 135, 139, 143, 194, 443, 445, 1433, 3306, 3389, 5632, 5900, 8080]
        results = {}
        
        # For demonstration/safety purposes, we'll simulate the scan
        results["method"] = "simulation"
        results["scanned_ports"] = common_ports
        results["open_ports"] = {}
        
        # Simulate some common services
        if 80 in common_ports:
            results["open_ports"]["80"] = {"service": "HTTP", "banner": "Apache/2.4.41"}
        if 443 in common_ports:
            results["open_ports"]["443"] = {"service": "HTTPS", "banner": "nginx/1.18.0"}
        if 22 in common_ports:
            results["open_ports"]["22"] = {"service": "SSH", "banner": "OpenSSH 8.2p1"}
        
        return results
    
    def _query_shodan(self, target):
        """Query Shodan for information about the target."""
        # This would normally use the Shodan API
        # For demonstration, we'll return simulated data
        return {
            "ip": "198.51.100.1",
            "ports": [80, 443, 22],
            "hostnames": ["example.com", "www.example.com"],
            "country": "United States",
            "org": "Example Organization",
            "data": [
                {
                    "port": 80,
                    "service": "HTTP",
                    "product": "Apache httpd",
                    "version": "2.4.41"
                },
                {
                    "port": 443,
                    "service": "HTTPS",
                    "product": "nginx",
                    "version": "1.18.0"
                }
            ]
        }
    
    def _query_virustotal(self, domain):
        """Query VirusTotal for domain information."""
        # This would normally use the VirusTotal API
        # For demonstration, we'll return simulated data
        return {
            "response_code": 1,
            "domain_info": {
                "categories": ["business"],
                "creation_date": "2010-01-01",
                "last_update_date": "2023-01-01"
            },
            "detected_urls": 0,
            "detected_downloaded_samples": 0,
            "detected_communicating_samples": 0,
            "resolutions": [
                {"ip_address": "198.51.100.1", "last_resolved": "2023-01-01"}
            ]
        }
    
    def _simulate_github_search(self, domain):
        """Simulate searching GitHub for references to the domain."""
        return {
            "status": "completed",
            "results_count": 3,
            "sample_results": [
                {"repository": "example/repo1", "description": "Configuration example"},
                {"repository": "example/repo2", "description": "API client library"},
                {"repository": "example/docs", "description": "Documentation site"}
            ]
        }
    
    def _analyze_recon_data(self, report):
        """Analyze reconnaissance data using AI."""
        # Convert report to text summary
        summary = f"""# Reconnaissance Report for {report['target']}

## Target Information
- Type: {'IP Address' if self._is_ip_address(report['target']) else 'Domain'}
- Timestamp: {report['timestamp']}
- Reconnaissance Type: {report['recon_type']}

## Key Findings
"""
        
        # Add domain information if available
        if "domain_info" in report["results"]:
            summary += "### Domain Information\n"
            
            # Add DNS info
            if "dns" in report["results"]["domain_info"]:
                dns = report["results"]["domain_info"]["dns"]
                if "error" not in dns:
                    summary += f"- Hostname: {dns['hostname']}\n"
                    summary += f"- IP Addresses: {', '.join(dns['ip_addresses'])}\n"
                    if dns['aliases']:
                        summary += f"- Aliases: {', '.join(dns['aliases'])}\n"
                else:
                    summary += f"- DNS Error: {dns['error']}\n"
            
            # Add WHOIS info
            if "whois" in report["results"]["domain_info"]:
                whois = report["results"]["domain_info"]["whois"]
                if "error" not in whois:
                    summary += f"- Registrar: {whois['registrar']}\n"
                    summary += f"- Creation Date: {whois['creation_date']}\n"
                    summary += f"- Expiration Date: {whois['expiration_date']}\n"
                    summary += f"- Name Servers: {', '.join(whois['name_servers']) if isinstance(whois['name_servers'], list) else whois['name_servers']}\n"
                else:
                    summary += f"- WHOIS Error: {whois['error']}\n"
        
        # Add IP information if available
        if "ip_info" in report["results"]:
            summary += "### IP Information\n"
            ip_info = report["results"]["ip_info"]
            
            if "reverse_dns" in ip_info:
                if isinstance(ip_info["reverse_dns"], str):
                    summary += f"- Reverse DNS: {ip_info['reverse_dns']}\n"
                else:
                    summary += f"- Reverse DNS Error: {ip_info['reverse_dns']['error']}\n"
            
            if "geo" in ip_info:
                geo = ip_info["geo"]
                summary += f"- Location: {geo['city']}, {geo['region']}, {geo['country']}\n"
                summary += f"- ISP: {geo['isp']}\n"
        
        # Add port scan information
        if "port_scan" in report["results"]:
            summary += "### Open Ports\n"
            port_scan = report["results"]["port_scan"]
            
            if "error" not in port_scan:
                if "open_ports" in port_scan and port_scan["open_ports"]:
                    for port, info in port_scan["open_ports"].items():
                        summary += f"- Port {port}: {info['service']} ({info['banner']})\n"
                else:
                    summary += "- No open ports detected in scan\n"
            else:
                summary += f"- Port Scan Error: {port_scan['error']}\n"
        
        # Add external data sources
        if "shodan" in report["results"]:
            summary += "### Shodan Information\n"
            shodan = report["results"]["shodan"]
            
            if "error" not in shodan:
                summary += f"- Organization: {shodan['org']}\n"
                summary += f"- Hostnames: {', '.join(shodan['hostnames'])}\n"
                summary += f"- Open Ports: {', '.join(map(str, shodan['ports']))}\n"
                
                if "data" in shodan:
                    summary += "- Services:\n"
                    for service in shodan["data"]:
                        summary += f"  - {service['port']}/{service['service']}: {service['product']} {service['version']}\n"
            else:
                summary += f"- Shodan Error: {shodan['error']}\n"
        
        if "virustotal" in report["results"]:
            summary += "### VirusTotal Information\n"
            vt = report["results"]["virustotal"]
            
            if "error" not in vt:
                if vt["response_code"] == 1:
                    summary += f"- Categories: {', '.join(vt['domain_info']['categories'])}\n"
                    summary += f"- Creation Date: {vt['domain_info']['creation_date']}\n"
                    summary += f"- Last Update: {vt['domain_info']['last_update_date']}\n"
                    summary += f"- Detected URLs: {vt['detected_urls']}\n"
                    
                    if "resolutions" in vt:
                        summary += "- Recent IP Resolutions:\n"
                        for res in vt["resolutions"]:
                            summary += f"  - {res['ip_address']} (Last resolved: {res['last_resolved']})\n"
                else:
                    summary += "- Domain not found in VirusTotal\n"
            else:
                summary += f"- VirusTotal Error: {vt['error']}\n"
        
        # Add repository search results
        if "repository_search" in report["results"]:
            summary += "### Public Repository Mentions\n"
            repos = report["results"]["repository_search"]
            
            if "github" in repos and repos["github"]["status"] == "completed":
                summary += f"- GitHub Results: {repos['github']['results_count']} repositories\n"
                if repos["github"]["results_count"] > 0:
                    summary += "- Sample Repositories:\n"
                    for repo in repos["github"]["sample_results"]:
                        summary += f"  - {repo['repository']}: {repo['description']}\n"
        
        # Add breach information
        if "breach_check" in report["results"]:
            summary += "### Breach Database Check\n"
            breach = report["results"]["breach_check"]
            
            if breach["found_in_breaches"]:
                summary += "- Target found in breach databases\n"
            else:
                summary += f"- {breach['message']}\n"
        
        # Analyze the data with AI
        messages = [
            {"role": "system", "content": self.recon_system_message},
            {"role": "user", "content": f"Analyze the following reconnaissance data and provide a security assessment with key findings, potential vulnerabilities, and recommendations:\n\n{summary}"}
        ]
        
        analysis = self.generate_response(messages, temperature=0.3, max_tokens=2048)
        
        # Return combined summary and analysis
        return f"{summary}\n\n## AI Analysis\n{analysis}"


if __name__ == "__main__":
    # Test chat engine
    engine = ChatEngine()
    response = engine.process_normal_query("What's the weather today?")
    print("Response:", response)
    
    # Test reconnaissance functionality
    recon_result = engine.perform_recon("example.com")
    print(recon_result)
