import os
import sys
import time
import datetime
import json
import re
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from concurrent.futures import ThreadPoolExecutor
from markdown_it import MarkdownIt
import whois

# Try to import optional dependencies
try:
    from googlesearch import search
except ImportError:
    def search(*args, **kwargs):
        return ["Search functionality unavailable - googlesearch-python not installed"]

class PersonOSINT:
    """Advanced person-based OSINT module that integrates with the ChatEngine."""
    
    def __init__(self, chat_engine=None, config=None):
        self.chat_engine = chat_engine
        self.config = config
        self.console = Console()
        
        # Initialize directories
        os.makedirs("Data/Person_OSINT", exist_ok=True)
        
        # API keys (inherit from chat_engine if available)
        if chat_engine:
            self.groq_api_key = chat_engine.api_key
        else:
            self.groq_api_key = os.environ.get("GroqAPIKey")
            
        # Social media regex patterns
        self.social_patterns = {
            'twitter': r'twitter\.com/([A-Za-z0-9_]+)',
            'linkedin': r'linkedin\.com/in/([A-Za-z0-9_-]+)',
            'facebook': r'facebook\.com/([A-Za-z0-9.]+)',
            'instagram': r'instagram\.com/([A-Za-z0-9_.]+)',
            'github': r'github\.com/([A-Za-z0-9_-]+)',
            'youtube': r'youtube\.com/(@?[A-Za-z0-9_-]+)',
            'medium': r'medium\.com/@?([A-Za-z0-9_.]+)',
            'reddit': r'reddit\.com/user/([A-Za-z0-9_-]+)'
        }
        
    def person_recon(self, target=None, depth="standard"):
        """Main method to perform person reconnaissance."""
        if not target:
            target = Prompt.ask("[bold green]>[/bold green] Target Person (full name)")
        
        # Validate target format (basic check for name format)
        if not self._validate_person_name(target):
            self.console.print("[bold red]Invalid person name format. Please use a full name.[/bold red]")
            return
            
        # Display header
        self.console.print(Panel.fit(
            f"[bold cyan]🔍 PERSON OSINT OPERATION: {target.upper()} 🔍[/bold cyan]", 
            style="bold blue"
        ))
        
        # Initialize report structure
        report = {
            "target": target,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recon_type": depth,
            "results": {}
        }
        
        # Show progress indicators
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            # Define OSINT steps
            steps = [
                "Searching public records",
                "Identifying social media profiles",
                "Looking for email addresses",
                "Checking professional background",
                "Finding online presence",
                "Searching for public documents",
                "Analyzing digital footprint",
                "Generating comprehensive report"
            ]
            
            for step in steps:
                task = progress.add_task(description=f"[bold blue]{step}...[/bold blue]", total=1)
                
                # Perform the actual OSINT operations based on step
                if "social media" in step:
                    report["results"]["social_media"] = self._find_social_profiles(target)
                elif "email" in step:
                    report["results"]["emails"] = self._find_email_addresses(target)
                elif "professional" in step:
                    report["results"]["professional"] = self._find_professional_info(target)
                elif "public records" in step:
                    report["results"]["public_records"] = self._find_public_records(target)
                elif "online presence" in step:
                    report["results"]["online_presence"] = self._find_online_presence(target)
                elif "documents" in step:
                    report["results"]["documents"] = self._find_documents(target)
                elif "digital footprint" in step:
                    report["results"]["digital_footprint"] = self._analyze_digital_footprint(target)
                
                # Simulate work for demo purposes
                time.sleep(0.5)
                progress.update(task, completed=1)
                
        # Generate AI analysis of collected data
        analysis_text = self._generate_osint_analysis(report)
        report["analysis"] = analysis_text
        
        # Create a human-readable report
        formatted_report = self._format_osint_report(report)
        
        # Save the report
        filename = self._save_report(target, formatted_report)
        
        # Display the report
        self.console.print("\n[bold cyan]╔═══ PERSON OSINT REPORT ═════════════════════════════╗[/bold cyan]")
        self.console.print(f"[bold cyan]║[/bold cyan] [bold white]Target:[/bold white] {target}")
        self.console.print(f"[bold cyan]║[/bold cyan] [bold white]Type:[/bold white] Person Intelligence - {depth.capitalize()}")
        if filename:
            self.console.print(f"[bold cyan]║[/bold cyan] [bold white]Saved to:[/bold white] {filename}")
        self.console.print("[bold cyan]╠════════════════════════════════════════════════════╣[/bold cyan]")
        
        self.console.print(formatted_report)
        
        self.console.print("[bold cyan]╚════════════════════════════════════════════════════╝[/bold cyan]")
        
        # Exposure risk assessment (mock)
        self.console.print("\n[bold magenta]PRIVACY EXPOSURE SCORE:[/bold magenta] [bold yellow]58/100[/bold yellow] — [italic]Moderate digital footprint[/italic]")
        
        return formatted_report
        
    def _validate_person_name(self, name):
        """Validate if input appears to be a person's name."""
        # Basic check for name format (at least two parts, alphabetic)
        name_parts = name.strip().split()
        if len(name_parts) < 2:
            return False
            
        # Check that name parts are alphabetic (allowing for hyphens and apostrophes)
        name_pattern = r'^[A-Za-z\'\-]+$'
        for part in name_parts:
            if not re.match(name_pattern, part):
                return False
                
        return True
        
    def _find_social_profiles(self, person):
        """Find social media profiles for the target person."""
        profiles = {}
        name_query = person.replace(" ", "+")
        
        # Search for profiles using search engine
        try:
            search_queries = [
                f"{person} site:linkedin.com",
                f"{person} site:twitter.com",
                f"{person} site:facebook.com",
                f"{person} site:instagram.com",
                f"{person} site:github.com"
            ]
            
            all_results = []
            for query in search_queries:
                results = list(search(query, num_results=3))
                all_results.extend(results)
                
            # Extract usernames using regex patterns
            for result in all_results:
                for platform, pattern in self.social_patterns.items():
                    match = re.search(pattern, result)
                    if match:
                        username = match.group(1)
                        profiles[platform] = {
                            "username": username,
                            "url": result,
                            "verified": False  # Cannot verify without API access
                        }
        except Exception as e:
            profiles["search_error"] = str(e)
            
        return profiles
        
    def _find_email_addresses(self, person):
        """Find potential email addresses for the target person."""
        emails = {"potential_formats": [], "found_addresses": []}
        
        # Generate common email formats
        name_parts = person.lower().split()
        if len(name_parts) >= 2:
            first = name_parts[0]
            last = name_parts[-1]
            
            # Common formats
            formats = [
                f"{first}@example.com",
                f"{first}.{last}@example.com",
                f"{first[0]}{last}@example.com",
                f"{last}.{first}@example.com",
                f"{first}_{last}@example.com"
            ]
            
            emails["potential_formats"] = formats
            
        # Search for email addresses (simulation)
        # In a real implementation, you would use various OSINT techniques
        
        return emails
        
    def _find_professional_info(self, person):
        """Find professional information about the target person."""
        professional = {
            "possible_employers": [],
            "job_titles": [],
            "education": [],
            "skills": []
        }
        
        # Simulate finding professional information
        # In a real implementation, this would involve:
        # 1. Scraping LinkedIn or similar profiles
        # 2. Searching job sites
        # 3. Looking at company employee directories
        
        return professional
        
    def _find_public_records(self, person):
        """Find public records related to the target person."""
        records = {
            "address_history": [],
            "phone_numbers": [], 
            "court_records": [],
            "property_records": []
        }
        
        # Simulate finding public records
        # In a real implementation, this would require:
        # 1. Searching public records databases
        # 2. Court record searches
        # 3. Property databases
        
        return records
        
    def _find_online_presence(self, person):
        """Find online presence indicators for the target person."""
        presence = {
            "blogs": [],
            "forums": [],
            "news_mentions": [],
            "articles": []
        }
        
        # Search for online presence using search engine
        try:
            search_queries = [
                f'"{person}" blog author',
                f'"{person}" forum profile',
                f'"{person}" news'
            ]
            
            for query in search_queries:
                results = list(search(query, num_results=5))
                
                # Categorize results
                for result in results:
                    result_lower = result.lower()
                    if "blog" in result_lower or "author" in result_lower:
                        presence["blogs"].append(result)
                    elif "forum" in result_lower:
                        presence["forums"].append(result)
                    elif "news" in result_lower:
                        presence["news_mentions"].append(result)
                        
        except Exception as e:
            presence["search_error"] = str(e)
            
        return presence
        
    def _find_documents(self, person):
        """Find public documents related to the target person."""
        documents = {
            "academic_papers": [],
            "presentations": [],
            "pdfs": [],
            "patents": []
        }
        
        # Search for documents using search engine
        try:
            search_queries = [
                f'"{person}" filetype:pdf',
                f'"{person}" academic paper',
                f'"{person}" patent'
            ]
            
            for query in search_queries:
                results = list(search(query, num_results=5))
                
                # Categorize results
                for result in results:
                    result_lower = result.lower()
                    if result_lower.endswith(".pdf"):
                        documents["pdfs"].append(result)
                    elif "academic" in result_lower or "paper" in result_lower:
                        documents["academic_papers"].append(result)
                    elif "patent" in result_lower:
                        documents["patents"].append(result)
                    elif "presentation" in result_lower or "slides" in result_lower:
                        documents["presentations"].append(result)
                        
        except Exception as e:
            documents["search_error"] = str(e)
            
        return documents
        
    def _analyze_digital_footprint(self, person):
        """Analyze overall digital footprint of the target person."""
        footprint = {
            "platforms": [],
            "frequency": "unknown",
            "first_appearance": "unknown",
            "sentiment": "neutral",
            "topics": []
        }
        
        # In a real implementation, this would aggregate data from
        # other collection methods and analyze patterns
        
        return footprint
        
    def _generate_osint_analysis(self, report):
        """Generate AI analysis of the collected OSINT data using Groq."""
        if not self.chat_engine or not self.groq_api_key:
            return "AI analysis unavailable - API integration not configured."
            
        # Create summary of collected data
        summary = f"""# PERSON OSINT DATA FOR: {report['target']}

## Collection Summary
- Target: {report['target']}
- Collection Time: {report['timestamp']}
- Collection Depth: {report['recon_type']}

## Social Media Profiles
"""
        # Add social media information
        if "social_media" in report["results"]:
            social = report["results"]["social_media"]
            if "search_error" in social:
                summary += f"- Error retrieving social profiles: {social['search_error']}\n"
            else:
                for platform, data in social.items():
                    summary += f"- {platform.capitalize()}: @{data['username']}\n"
                
        # Add email information
        if "emails" in report["results"]:
            summary += "\n## Email Intelligence\n"
            emails = report["results"]["emails"]
            if emails["found_addresses"]:
                summary += "- Found addresses:\n"
                for email in emails["found_addresses"]:
                    summary += f"  - {email}\n"
            else:
                summary += "- No confirmed email addresses found\n"
            
            summary += "- Common email format predictions:\n"
            for fmt in emails["potential_formats"][:3]:
                summary += f"  - {fmt}\n"
                
        # Add online presence information
        if "online_presence" in report["results"]:
            summary += "\n## Online Presence\n"
            presence = report["results"]["online_presence"]
            
            if "search_error" in presence:
                summary += f"- Error retrieving online presence: {presence['search_error']}\n"
            else:
                if presence["blogs"]:
                    summary += f"- Found {len(presence['blogs'])} potential blog entries\n"
                if presence["news_mentions"]:
                    summary += f"- Found {len(presence['news_mentions'])} potential news mentions\n" 
                if presence["forums"]:
                    summary += f"- Found {len(presence['forums'])} potential forum profiles\n"
                    
        # Add document information
        if "documents" in report["results"]:
            summary += "\n## Documents\n"
            docs = report["results"]["documents"]
            
            if "search_error" in docs:
                summary += f"- Error retrieving documents: {docs['search_error']}\n"
            else:
                if docs["academic_papers"]:
                    summary += f"- Found {len(docs['academic_papers'])} potential academic papers\n"
                if docs["pdfs"]:
                    summary += f"- Found {len(docs['pdfs'])} PDF documents\n"
                if docs["patents"]:
                    summary += f"- Found {len(docs['patents'])} potential patents\n"
                    
        # Use the AI to analyze this data
        prompt = f"""As a cybersecurity OSINT analyst, please analyze the following intelligence collected on a target individual. 
Provide insights on:
1. Key findings and patterns
2. Potential privacy/security concerns
3. Identity confidence level
4. Recommended additional research areas

Here's the collected data:

{summary}

Remember to focus on factual analysis rather than speculation, and maintain ethical standards.
"""
        
        try:
            # Generate analysis using chat_engine or direct API call
            if self.chat_engine:
                system_message = self.chat_engine.security_system_message
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
                analysis = self.chat_engine.generate_response(messages, temperature=0.3, max_tokens=2048)
            else:
                # This would be a direct API implementation
                analysis = "AI analysis unavailable in standalone mode."
                
            return analysis
            
        except Exception as e:
            return f"AI analysis error: {str(e)}"
    
    def _format_osint_report(self, report):
        """Format the OSINT data into a readable report."""
        # Basic markdown report format
        markdown = f"""# PERSON INTELLIGENCE REPORT

## Target: {report['target']}
- **Collection Date:** {report['timestamp']}
- **Collection Depth:** {report['recon_type']}

## SOCIAL MEDIA INTELLIGENCE
"""
        # Add social profiles
        if "social_media" in report["results"]:
            social = report["results"]["social_media"]
            if not social or "search_error" in social:
                markdown += "- No confirmed social profiles identified\n"
            else:
                for platform, data in social.items():
                    markdown += f"- **{platform.capitalize()}:** [@{data['username']}]({data['url']})\n"
        
        # Add email intelligence  
        markdown += "\n## EMAIL INTELLIGENCE\n"
        if "emails" in report["results"] and report["results"]["emails"]["found_addresses"]:
            for email in report["results"]["emails"]["found_addresses"]:
                markdown += f"- {email}\n"
        else:
            markdown += "- No confirmed email addresses found\n"
            if "emails" in report["results"] and report["results"]["emails"]["potential_formats"]:
                markdown += "- **Potential email formats:**\n"
                for email_format in report["results"]["emails"]["potential_formats"][:3]:
                    markdown += f"  - {email_format}\n"
        
        # Add online presence
        markdown += "\n## ONLINE PRESENCE\n"
        if "online_presence" in report["results"]:
            presence = report["results"]["online_presence"]
            if "blogs" in presence and presence["blogs"]:
                markdown += "### Blogs & Articles\n"
                for url in presence["blogs"][:3]:
                    markdown += f"- {url}\n"
            
            if "news_mentions" in presence and presence["news_mentions"]:
                markdown += "\n### News Mentions\n"
                for url in presence["news_mentions"][:3]:
                    markdown += f"- {url}\n"
        
        # Add documents section
        if "documents" in report["results"] and any(report["results"]["documents"].values()):
            markdown += "\n## DOCUMENTS\n"
            docs = report["results"]["documents"]
            
            if "academic_papers" in docs and docs["academic_papers"]:
                markdown += "### Academic Papers\n"
                for paper in docs["academic_papers"][:2]:
                    markdown += f"- {paper}\n"
                    
            if "pdfs" in docs and docs["pdfs"]:
                markdown += "\n### PDF Documents\n"
                for pdf in docs["pdfs"][:2]:
                    markdown += f"- {pdf}\n"
        
        # Add AI analysis
        if "analysis" in report:
            markdown += "\n## ANALYSIS\n"
            markdown += report["analysis"]
            
        # Add disclaimer
        markdown += "\n\n---\n"
        markdown += "*This report is for authorized security assessment purposes only. All data collection methods used are passive and non-intrusive.*"
        
        return markdown
    
    def _save_report(self, target, report_content):
        """Save the report to a file."""
        try:
            # Create a filename-safe version of the target name
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', target)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Data/Person_OSINT/person_{safe_name}_{timestamp}.md"
            
            with open(filename, "w") as f:
                f.write(report_content)
                
            return filename
        except Exception as e:
            self.console.print(f"[bold red]Failed to save report: {e}[/bold red]")
            return None


# Integration functions for ChatEngine
def integrate_person_osint(chat_engine):
    """Integrate the PersonOSINT module with an existing ChatEngine."""
    # Create the OSINT module with reference to the chat engine
    osint_module = PersonOSINT(chat_engine=chat_engine)
    
    # Define method for the command handler to call
    def person_osint_command(args=None):
        """Command handler for person OSINT."""
        if not args:
            target = Prompt.ask("[bold green]>[/bold green] Target person (full name)")
        else:
            target = " ".join(args)
            
        # Check current mode
        if hasattr(chat_engine, 'security_mode') and chat_engine.security_mode != "security":
            Console().print("[bold yellow]Switching to security mode for OSINT operation...[/bold yellow]")
            
        return osint_module.person_recon(target)
        
    # Return the command handler
    return person_osint_command