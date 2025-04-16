import os
import sys
import time
import datetime
import json
import re
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.layout import Layout
from rich.text import Text

# Try to import optional dependencies with better error handling
try:
    from googlesearch import search
except ImportError:
    def search(*args, **kwargs):
        return ["Search functionality unavailable - googlesearch-python not installed"]

class PersonOSINT:
    """Person-based OSINT module that integrates with the ChatEngine."""
    
    def __init__(self, chat_engine=None, config=None):
        self.chat_engine = chat_engine
        self.config = config
        self.console = Console()
        os.makedirs("Data/Person_OSINT", exist_ok=True)
        self.api_key = chat_engine.api_key if chat_engine else os.environ.get("GroqAPIKey")
        self.social_patterns = {
            'twitter': r'twitter\.com/([A-Za-z0-9_]+)',
            'linkedin': r'linkedin\.com/in/([A-Za-z0-9_-]+)',
            'facebook': r'facebook\.com/([A-Za-z0-9.]+)',
            'instagram': r'instagram\.com/([A-Za-z0-9_.]+)',
            'github': r'github\.com/([A-Za-z0-9_-]+)',
            'youtube': r'youtube\.com/(@?[A-Za-z0-9_-]+)',
            'medium': r'medium\.com/@?([A-Za-z0-9_.]+)',
            'reddit': r'reddit\.com/user/([A-Za-z0-9_-]+)',
            'tiktok': r'tiktok\.com/@([A-Za-z0-9_.]+)'
        }
        # Set search limits and timeouts for more reliable results
        self.search_timeout = 10
        self.search_limit = 5
        
        # Google dork patterns
        self.dorks = {
            "personal_info": [
                '"{target}" filetype:pdf OR filetype:doc OR filetype:docx',
                '"{target}" intitle:resume OR intitle:CV',
                '"{target}" intext:address OR intext:phone OR intext:email',
                '"{target}" intext:birthday OR intext:born',
                '"{target}" "contact information"'
            ],
            "education_work": [
                '"{target}" intitle:alumni OR intext:graduated',
                '"{target}" intext:university OR intext:college',
                '"{target}" intext:employee OR intext:staff',
                '"{target}" intext:worked OR intext:position'
            ],
            "financial": [
                '"{target}" intext:salary OR intext:income',
                '"{target}" intext:investment OR intext:property',
                '"{target}" site:linkedin.com intitle:profile'
            ],
            "technical": [
                '"{target}" intext:username OR intext:password',
                '"{target}" inurl:config OR inurl:backup OR inurl:admin',
                '"{target}" site:github.com',
                '"{target}" filetype:sql OR filetype:log OR filetype:env'
            ]
        }
        
        # Dark web search URLs (these would be placeholder URLs in a real implementation)
        self.dark_web_engines = {
            "tor_search": "http://torsearchruby77t.onion",  # Placeholder
            "not_evil": "http://notevilmcgdruol.onion",     # Placeholder
            "torch": "http://torchsearchrzg5jhj.onion"      # Placeholder
        }
    
    def person_recon(self, target=None, depth="standard"):
        """Main method to perform person reconnaissance."""
        if not target:
            target = Prompt.ask("[bold green]>[/bold green] Target Person (full name)")
        if not self._validate_person_name(target):
            self.console.print("[bold red]Invalid person name format. Please use a full name.[/bold red]")
            return

        # Create a more visually striking header with custom styling
        header_text = Text()
        header_text.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n", style="bold cyan")
        header_text.append("┃  ", style="bold cyan")
        header_text.append(f"🔍 PERSON OSINT OPERATION: {target.upper()}", style="bold white")
        header_text.append(" 🔍  ┃\n", style="bold cyan")
        header_text.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛", style="bold cyan")
        self.console.print(header_text)
        
        report = {
            "target": target,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recon_type": depth,
            "results": {}
        }

        # Define collection steps with more detailed progress tracking
        steps = [
            ("Identifying social media profiles", self._find_social_profiles),
            ("Looking for email addresses", self._find_email_addresses),
            ("Finding online presence", self._find_online_presence),
            ("Searching for public documents", self._find_documents),
            ("Running Google dork queries", self._run_google_dorks),
            ("Searching dark web mentions", self._search_dark_web),
            ("Generating comprehensive report", None)
        ]

        # Execute collection steps with improved visual progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            for step_desc, step_func in steps:
                task = progress.add_task(description=f"[bold blue]{step_desc}...[/bold blue]", total=1)
                
                try:
                    if step_func:
                        result_key = step_desc.lower().split(" ")[1].replace("ing", "").strip()
                        if result_key == "social":
                            result_key = "social_media"
                        elif result_key == "google":
                            result_key = "dork_results"
                        elif result_key == "dark":
                            result_key = "dark_web"
                        report["results"][result_key] = step_func(target)
                except Exception as e:
                    self.console.print(f"[bold yellow]Warning: Error in {step_desc}: {str(e)}[/bold yellow]")
                    report["results"][f"error_{result_key}"] = str(e)
                
                time.sleep(0.5)
                progress.update(task, completed=1)

        # Generate analysis and format report
        report["analysis"] = self._generate_osint_analysis(report)
        formatted_report = self._format_osint_report(report)
        filename = self._save_report(target, formatted_report)

        # Display report summary with improved visual presentation
        self._display_report_summary(target, depth, filename, report)
        
        return formatted_report
    
    def _validate_person_name(self, name):
        """Validate if input appears to be a person's name with improved validation."""
        name_parts = name.strip().split()
        if len(name_parts) < 2:
            return False
            
        # Check that name parts are alphabetic (allowing for common name characters)
        name_pattern = r'^[A-Za-z\'\-\.]+$'
        for part in name_parts:
            if not re.match(name_pattern, part):
                return False
                
        return True
        
    def _find_social_profiles(self, person):
        """Find social media profiles for the target person with improved accuracy."""
        profiles = {}
        
        try:
            # More targeted search queries with name variations
            name_parts = person.split()
            first_name = name_parts[0]
            last_name = name_parts[-1]
            
            search_queries = [
                f'"{person}" site:linkedin.com',
                f'"{person}" site:twitter.com OR site:x.com',
                f'"{first_name} {last_name}" site:facebook.com',
                f'"{first_name} {last_name}" site:instagram.com',
                f'"{first_name} {last_name}" site:github.com'
            ]
            
            all_results = []
            for query in search_queries:
                try:
                    results = list(search(query, num_results=self.search_limit, timeout=self.search_timeout))
                    all_results.extend(results)
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Search failed for '{query}': {str(e)}[/yellow]")
                
            # Extract usernames using regex patterns and verify potential matches
            for result in all_results:
                for platform, pattern in self.social_patterns.items():
                    match = re.search(pattern, result)
                    if match:
                        username = match.group(1)
                        # Add confidence score based on name match
                        confidence = "High" if person.lower().replace(" ", "") in result.lower() else "Medium"
                        
                        profiles[platform] = {
                            "username": username,
                            "url": result,
                            "confidence": confidence
                        }
        except Exception as e:
            profiles["search_error"] = str(e)
            
        return profiles
        
    def _find_email_addresses(self, person):
        """Find potential email addresses for the target person with improved accuracy."""
        emails = {"potential_formats": [], "found_addresses": []}
        
        # Generate common email formats with domain research
        name_parts = person.lower().split()
        if len(name_parts) >= 2:
            first = name_parts[0]
            last = name_parts[-1]
            
            # Common formats with more realistic domains
            common_domains = ["gmail.com", "outlook.com", "yahoo.com", "protonmail.com"]
            professional_domains = self._find_potential_work_domains(person)
            
            all_domains = common_domains + professional_domains
            
            # Generate formats
            for domain in all_domains[:5]:  # Limit to 5 domains
                emails["potential_formats"].append(f"{first}@{domain}")
                emails["potential_formats"].append(f"{first}.{last}@{domain}")
                emails["potential_formats"].append(f"{first[0]}{last}@{domain}")
                emails["potential_formats"].append(f"{last}.{first}@{domain}")
            
        # Search for public email addresses
        try:
            query = f'"{person}" email OR contact OR "reach me at" OR "contact info"'
            results = list(search(query, num_results=5, timeout=self.search_timeout))
            
            # Extract email patterns
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            for result in results:
                # In a real app, would need to fetch and scan page content
                # This is a placeholder for the email extraction logic
                pass
                
        except Exception as e:
            emails["search_error"] = str(e)
            
        return emails
    
    def _find_potential_work_domains(self, person):
        """Find potential work domains for the target."""
        try:
            query = f'"{person}" company OR organization OR employer'
            results = list(search(query, num_results=3, timeout=self.search_timeout))
            
            # Extract potential company domains
            domains = []
            company_pattern = r'at ([A-Za-z0-9\s]+)'
            domain_pattern = r'([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            
            for result in results:
                domain_match = re.search(domain_pattern, result)
                if domain_match and "google" not in domain_match.group(1):
                    domains.append(domain_match.group(1))
            
            return domains[:3]  # Return top 3 domains
        except:
            return []
        
    def _find_online_presence(self, person):
        """Find online presence indicators with improved categorization."""
        presence = {
            "blogs": [],
            "forums": [],
            "news_mentions": [],
            "publications": [],
            "social_activities": []
        }
        
        # Enhanced search for online presence
        try:
            search_queries = [
                f'"{person}" blog OR author OR writer',
                f'"{person}" forum OR community OR discussion',
                f'"{person}" news OR article OR mentioned',
                f'"{person}" publication OR research OR paper'
            ]
            
            for query in search_queries:
                try:
                    results = list(search(query, num_results=self.search_limit, timeout=self.search_timeout))
                    
                    # Improved categorization with confidence scoring
                    for result in results:
                        result_lower = result.lower()
                        # Use weighted scoring to determine best category
                        scores = {
                            "blogs": sum(1 for term in ["blog", "author", "post", "write"] if term in result_lower),
                            "forums": sum(1 for term in ["forum", "community", "thread", "discussion"] if term in result_lower),
                            "news_mentions": sum(1 for term in ["news", "press", "article", "announced"] if term in result_lower),
                            "publications": sum(1 for term in ["publication", "research", "paper", "journal"] if term in result_lower),
                            "social_activities": sum(1 for term in ["profile", "social", "network", "connected"] if term in result_lower)
                        }
                        
                        # Find category with highest score
                        best_category = max(scores, key=scores.get)
                        if scores[best_category] > 0:
                            if result not in presence[best_category]:
                                presence[best_category].append(result)
                except Exception as e:
                    continue
                    
        except Exception as e:
            presence["search_error"] = str(e)
            
        return presence
        
    def _find_documents(self, person):
        """Find public documents with improved document classification."""
        documents = {
            "academic_papers": [],
            "presentations": [],
            "pdfs": [],
            "patents": [],
            "reports": []
        }
        
        # Enhanced document search
        try:
            search_queries = [
                f'"{person}" filetype:pdf',
                f'"{person}" research OR paper OR publication',
                f'"{person}" patent OR intellectual property',
                f'"{person}" presentation OR slides OR "slide deck"',
                f'"{person}" report OR whitepaper OR "case study"'
            ]
            
            for query in search_queries:
                try:
                    results = list(search(query, num_results=self.search_limit, timeout=self.search_timeout))
                    
                    # Improved categorization with file extension detection
                    for result in results:
                        result_lower = result.lower()
                        if result_lower.endswith(".pdf"):
                            if result not in documents["pdfs"]:
                                documents["pdfs"].append(result)
                        elif result_lower.endswith((".ppt", ".pptx", ".key")) or "slides" in result_lower:
                            if result not in documents["presentations"]:
                                documents["presentations"].append(result)
                        elif "patent" in result_lower or "ip." in result_lower:
                            if result not in documents["patents"]:
                                documents["patents"].append(result)
                        elif any(term in result_lower for term in ["research", "paper", "journal", "conference"]):
                            if result not in documents["academic_papers"]:
                                documents["academic_papers"].append(result)
                        elif any(term in result_lower for term in ["report", "whitepaper", "case study"]):
                            if result not in documents["reports"]:
                                documents["reports"].append(result)
                except Exception:
                    continue
                        
        except Exception as e:
            documents["search_error"] = str(e)
            
        return documents
        
    def _run_google_dorks(self, person):
        """Execute Google dork queries for targeted OSINT."""
        dork_results = {
            "personal_information": [],
            "education_work_history": [],
            "financial_traces": [],
            "technical_footprints": []
        }
        
        try:
            # Execute each dork category
            for category, dork_list in self.dorks.items():
                result_key = {
                    "personal_info": "personal_information",
                    "education_work": "education_work_history",
                    "financial": "financial_traces",
                    "technical": "technical_footprints"
                }.get(category)
                
                for dork in dork_list:
                    query = dork.replace("{target}", person)
                    try:
                        results = list(search(query, num_results=3, timeout=self.search_timeout))
                        for result in results:
                            if result not in dork_results[result_key]:
                                dork_results[result_key].append(result)
                    except Exception as e:
                        continue
                        
        except Exception as e:
            dork_results["dork_error"] = str(e)
            
        return dork_results
        
    def _search_dark_web(self, person):
        """Search for mentions on dark web (simulation - in a real tool this would use Tor)."""
        dark_web_results = {
            "breached_credentials": [],
            "forum_mentions": [],
            "marketplace_listings": [],
            "search_engines": []
        }
        
        # This is a simulation - in a real tool this would use Tor
        # We'll generate simulated results for demonstration purposes
        try:
            self.console.print("[yellow]Note: Dark web search is simulated for this demo[/yellow]")
            
            # Simulate search results with randomization for demo purposes
            for engine_name in self.dark_web_engines:
                dark_web_results["search_engines"].append({
                    "engine": engine_name,
                    "query_status": "simulated",
                    "search_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
        except Exception as e:
            dark_web_results["search_error"] = str(e)
            
        return dark_web_results
        
    def _generate_osint_analysis(self, report):
        """Generate analysis with improved content organization and insights."""
        if not self.chat_engine or not self.api_key:
            return "AI analysis unavailable - API integration not configured."
            
        # Create enhanced summary of collected data
        summary = f"""# PERSON OSINT DATA FOR: {report['target']}

## Collection Summary
- Target: {report['target']}
- Collection Time: {report['timestamp']}
- Collection Depth: {report['recon_type']}

## Social Media Profiles
"""
        # Add social media information with confidence levels
        if "social_media" in report["results"]:
            social = report["results"]["social_media"]
            if "search_error" in social:
                summary += f"- Error retrieving social profiles: {social['search_error']}\n"
            else:
                for platform, data in social.items():
                    confidence = data.get('confidence', 'Unknown')
                    summary += f"- {platform.capitalize()}: @{data['username']} (Confidence: {confidence})\n"
                
        # Add email information with improved organization
        if "emails" in report["results"]:
            summary += "\n## Email Intelligence\n"
            emails = report["results"]["emails"]
            if emails.get("found_addresses"):
                summary += "- Found addresses:\n"
                for email in emails["found_addresses"]:
                    summary += f"  - {email}\n"
            else:
                summary += "- No confirmed email addresses found\n"
            
            if emails.get("potential_formats"):
                summary += "- Common email format predictions (sorted by likelihood):\n"
                for fmt in emails["potential_formats"][:5]:
                    summary += f"  - {fmt}\n"
                
        # Add Google dork results
        if "dork_results" in report["results"]:
            summary += "\n## Google Dork Intelligence\n"
            dorks = report["results"]["dork_results"]
            
            for category, items in dorks.items():
                if items and len(items) > 0:
                    summary += f"- {category.replace('_', ' ').title()}: {len(items)} results found\n"
                    
        # Add dark web results
        if "dark_web" in report["results"]:
            summary += "\n## Dark Web Intelligence\n"
            dark_web = report["results"]["dark_web"]
            
            if "search_engines" in dark_web:
                engines = dark_web["search_engines"]
                summary += f"- Searched {len(engines)} dark web engines\n"
                
            for category in ["breached_credentials", "forum_mentions", "marketplace_listings"]:
                if category in dark_web and dark_web[category]:
                    summary += f"- {category.replace('_', ' ').title()}: {len(dark_web[category])} results found\n"
                
        # Add online presence with enhanced categorization
        if "online_presence" in report["results"]:
            summary += "\n## Online Presence\n"
            presence = report["results"]["online_presence"]
            
            if "search_error" in presence:
                summary += f"- Error retrieving online presence: {presence['search_error']}\n"
            else:
                for category, items in presence.items():
                    if items:
                        summary += f"- Found {len(items)} potential {category.replace('_', ' ')}\n"
                    
        # Add document information with improved details
        if "documents" in report["results"]:
            summary += "\n## Documents\n"
            docs = report["results"]["documents"]
            
            if "search_error" in docs:
                summary += f"- Error retrieving documents: {docs['search_error']}\n"
            else:
                for doc_type, items in docs.items():
                    if items:
                        summary += f"- Found {len(items)} {doc_type.replace('_', ' ')}\n"
                    
        # Use the AI to analyze this data with improved prompting
        prompt = f"""As a cybersecurity OSINT analyst, analyze the following intelligence on {report['target']}. 
Provide a structured analysis covering:

1. Digital Footprint Assessment
   - Extent and visibility of online presence
   - Platform diversity and activity patterns

2. Identity Verification & Risk Analysis
   - Consistency across platforms and information sources
   - Primary privacy/security exposure points
   - Identity confidence level (High/Medium/Low)

3. Key Intelligence Findings
   - Professional background indicators
   - Content publishing patterns
   - Network relationships (if evident)
   - Potential security exposures from Google dork findings
   - Dark web presence analysis

4. OSINT Collection Recommendations
   - Specific targeted search areas for follow-up
   - Alternative intelligence sources worth exploring

Here's the collected data:

{summary}

Provide a factual, ethical analysis focused on defensive security applications.
"""
        
        try:
            # Generate analysis using chat_engine with improved error handling
            if self.chat_engine:
                system_message = self.chat_engine.security_system_message
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
                analysis = self.chat_engine.generate_response(messages, temperature=0.3, max_tokens=1024)
            else:
                analysis = "AI analysis unavailable in standalone mode."
                
            return analysis
            
        except Exception as e:
            return f"AI analysis error: {str(e)}"

    def _format_osint_report(self, report):
        """Format the OSINT data into a readable report with improved markdown structure."""
        markdown = f"""# PERSON INTELLIGENCE REPORT

## Target: {report['target']}
- **Collection Date:** {report['timestamp']}
- **Collection Depth:** {report['recon_type']}

## SOCIAL MEDIA INTELLIGENCE
"""
        social = report["results"].get("social_media", {})
        if not social or "search_error" in social:
            markdown += "- No confirmed social profiles identified\n"
        else:
            markdown += "| Platform | Username | Confidence | URL |\n"
            markdown += "|----------|----------|------------|-----|\n"
            for platform, data in social.items():
                if isinstance(data, dict):  # Skip any error entries
                    confidence = data.get('confidence', 'Unknown')
                    markdown += f"| {platform.capitalize()} | @{data['username']} | {confidence} | [Link]({data['url']}) |\n"

        markdown += "\n## EMAIL INTELLIGENCE\n"
        emails = report["results"].get("emails", {})
        if emails.get("found_addresses"):
            markdown += "### Confirmed Addresses\n"
            for email in emails["found_addresses"]:
                markdown += f"- `{email}`\n"
        else:
            markdown += "- No confirmed email addresses found\n"
            if emails.get("potential_formats"):
                markdown += "\n### Potential Email Formats\n"
                markdown += "*These formats are predictions based on common patterns:*\n\n"
                for fmt in emails["potential_formats"][:5]:
                    markdown += f"- `{fmt}`\n"

        # Add Google dork results section
        markdown += "\n## GOOGLE DORK INTELLIGENCE\n"
        dorks = report["results"].get("dork_results", {})
        
        if "dork_error" in dorks:
            markdown += f"- Error running Google dorks: {dorks['dork_error']}\n"
        else:
            dork_categories = {
                "personal_information": "### Personal Information Exposure",
                "education_work_history": "### Education & Work History",
                "financial_traces": "### Financial Information",
                "technical_footprints": "### Technical Footprints"
            }
            
            for category, header in dork_categories.items():
                if category in dorks and dorks[category]:
                    markdown += f"{header}\n"
                    for url in dorks[category][:5]:
                        markdown += f"- {url}\n"
                    markdown += "\n"

        # Add Dark Web section
        markdown += "## DARK WEB INTELLIGENCE\n"
        dark_web = report["results"].get("dark_web", {})
        
        if "search_error" in dark_web:
            markdown += f"- Error searching dark web: {dark_web['search_error']}\n"
        else:
            engines = dark_web.get("search_engines", [])
            if engines:
                markdown += "### Search Engines Queried\n"
                for engine in engines:
                    markdown += f"- {engine['engine']} (Status: {engine['query_status']})\n"
                
            markdown += "\n### Breach & Exposure Status\n"
            if not dark_web.get("breached_credentials") and not dark_web.get("forum_mentions"):
                markdown += "- No confirmed exposures identified in dark web sources\n"
            else:
                # If we had real results, they would be formatted here
                markdown += "- Dark web search results would be displayed here in a real implementation\n"

        markdown += "\n## ONLINE PRESENCE\n"
        presence = report["results"].get("online_presence", {})
        
        # Create online presence sections with better organization
        presence_categories = {
            "blogs": "### Blogs & Written Content",
            "news_mentions": "### News & Media Mentions",
            "publications": "### Publications & Research",
            "forums": "### Forum & Community Activity",
            "social_activities": "### Other Social Activities"
        }
        
        for category, header in presence_categories.items():
            if category in presence and presence[category]:
                markdown += f"{header}\n"
                for url in presence[category][:3]:
                    markdown += f"- {url}\n"
                markdown += "\n"

        markdown += "## DOCUMENT INTELLIGENCE\n"
        documents = report["results"].get("documents", {})
        
        # Create document sections with better organization
        doc_categories = {
            "academic_papers": "### Academic & Research Papers",
            "presentations": "### Presentations & Slides",
            "pdfs": "### PDF Documents",
            "patents": "### Patents & IP",
            "reports": "### Reports & Whitepapers"
        }
        
        for category, header in doc_categories.items():
            if category in documents and documents[category]:
                markdown += f"{header}\n"
                for doc in documents[category][:3]:
                    markdown += f"- {doc}\n"
                markdown += "\n"

        if "analysis" in report:
            markdown += "## ANALYSIS\n"
            markdown += report["analysis"]

        markdown += "\n\n---\n"
        markdown += "*This report is for authorized security assessment purposes only. All data collection methods used are passive and non-intrusive.*\n"
        markdown += f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"

        return markdown
    
    def _save_report(self, target, report_content):
        """Save the report to a file with proper error handling."""
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
    
    def _display_report_summary(self, target, depth, filename, report):
        """Display a formatted summary of the report with improved visuals."""
        # Create a custom layout for more impressive visualization
        layout = Layout()
        layout.split_column(
            Layout(name="header"),
            Layout(name="body"),
            Layout(name="footer")
        )
        
        # Header with dynamic ASCII art
        header = Text()
        header.append("\n")
        header.append("╔══════════════════════════════════════════════════════════════════╗\n", style="bold cyan")
        header.append("║                                                                  ║\n", style="bold cyan")
        header.append("║  ", style="bold cyan")
        header.append(f" 🔍 PERSON OSINT REPORT: {target.upper()}", style="bold white")
        header.append(" ".rjust(46 - len(target)), style="bold cyan")
        header.append("║\n", style="bold cyan")
        header.append("║                                                                  ║\n", style="bold cyan")
        header.append("╚══════════════════════════════════════════════════════════════════╝", style="bold cyan")
        
        # Create results summary table
        summary_table = Table(title="Intelligence Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Intelligence Type", style="cyan", width=25)
        summary_table.add_column("Results", style="green")
        
        # Add data to the table
        social_count = len(report["results"].get("social_media", {}))
        if "search_error" in report["results"].get("social_media", {}):
            social_count = 0
            
        email_count = len(report["results"].get("emails", {}).get("found_addresses", []))
        
        # Count dork results
        dork_count = 0
        for category, results in report["results"].get("dork_results", {}).items():
            if isinstance(results, list):
                dork_count += len(results)
                
        # Count online presence
        presence_count = 0
        for category, results in report["results"].get("online_presence", {}).items():
            if isinstance(results, list):
                presence_count += len(results)
                
        # Count documents
        document_count = 0
        for category, results in report["results"].get("documents", {}).items():
            if isinstance(results, list):
                document_count += len(results)
                
        summary_table.add_row("Social Media Profiles", f"{social_count} identified")
        summary_table.add_row("Email Addresses", f"{email_count} confirmed")
        summary_table.add_row("Google Dork Results", f"{dork_count} findings")
        summary_table.add_row("Online Presence", f"{presence_count} sources")
        summary_table.add_row("Document Intelligence", f"{document_count} documents")
        
        # Create a section for notable findings
        notable_panel = Panel(
            Text("Full report saved to: " + (filename or "Error saving file"), style="bold"),
            title="Report Location",
            border_style="green"
        )
        
        # Assign components to layout
        layout["header"].update(header)
        layout["body"].update(summary_table)
        layout["footer"].update(notable_panel)
        
        # Display the layout
        self.console.print(layout)
        
        # Add option to view in console
        choice = Prompt.ask(
            "[bold green]>[/bold green] View full report in console?", 
            choices=["y", "n"], 
            default="n"
        )
        
        if choice.lower() == "y":
            self.console.print("\n")
            self.console.rule("[bold]FULL REPORT[/bold]")
            self.console.print(report["analysis"])
            self.console.rule()

    def list_saved_reports(self):
        """List all saved OSINT reports with improved organization."""
        try:
            reports_dir = "Data/Person_OSINT"
            reports = [f for f in os.listdir(reports_dir) if f.endswith(".md")]
            
            if not reports:
                self.console.print("[yellow]No saved reports found.[/yellow]")
                return
                
            # Create a table to display reports
            table = Table(title="Saved OSINT Reports", show_header=True, header_style="bold magenta")
            table.add_column("ID", style="cyan", width=5)
            table.add_column("Target", style="green")
            table.add_column("Date", style="blue")
            table.add_column("Filename", style="dim")
            
            for i, report_file in enumerate(reports, 1):
                # Extract info from filename
                parts = report_file.replace("person_", "").replace(".md", "").split("_")
                if len(parts) >= 3:
                    name = parts[0].replace("_", " ")
                    date_str = f"{parts[-2]} {parts[-1]}"
                    try:
                        date_str = datetime.datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                    
                    table.add_row(str(i), name, date_str, report_file)
                else:
                    table.add_row(str(i), "Unknown", "Unknown", report_file)
                    
            self.console.print(table)
            
            # Allow viewing a report
            choice = Prompt.ask(
                "\n[bold green]>[/bold green] Enter report ID to view (or 'q' to quit)",
                default="q"
            )
            
            if choice.lower() != "q" and choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(reports):
                    with open(os.path.join(reports_dir, reports[idx]), "r") as f:
                        content = f.read()
                        self.console.print(Panel(
                            content,
                            title=f"Report: {reports[idx]}",
                            border_style="green",
                            width=100
                        ))
                        
        except Exception as e:
            self.console.print(f"[bold red]Error listing reports: {str(e)}[/bold red]")
    
    def compare_osint_reports(self, report1=None, report2=None):
        """Compare two OSINT reports to identify commonalities and differences."""
        try:
            reports_dir = "Data/Person_OSINT"
            reports = [f for f in os.listdir(reports_dir) if f.endswith(".md")]
            
            if not reports:
                self.console.print("[yellow]No saved reports found for comparison.[/yellow]")
                return
                
            if not report1:
                # Display reports for selection
                self.list_saved_reports()
                
                # Get first report selection
                choice1 = Prompt.ask(
                    "\n[bold green]>[/bold green] Enter first report ID to compare",
                    default="1"
                )
                
                if choice1.isdigit() and 0 < int(choice1) <= len(reports):
                    report1 = reports[int(choice1) - 1]
                else:
                    self.console.print("[yellow]Invalid report selection.[/yellow]")
                    return
                    
                # Get second report selection
                choice2 = Prompt.ask(
                    "\n[bold green]>[/bold green] Enter second report ID to compare",
                    default="2"
                )
                
                if choice2.isdigit() and 0 < int(choice2) <= len(reports):
                    report2 = reports[int(choice2) - 1]
                else:
                    self.console.print("[yellow]Invalid report selection.[/yellow]")
                    return
            
            # Load reports
            with open(os.path.join(reports_dir, report1), "r") as f:
                content1 = f.read()
                
            with open(os.path.join(reports_dir, report2), "r") as f:
                content2 = f.read()
                
            # Extract target names
            target1 = re.search(r"Target: (.*)", content1)
            target1 = target1.group(1) if target1 else "Unknown"
            
            target2 = re.search(r"Target: (.*)", content2)
            target2 = target2.group(1) if target2 else "Unknown"
            
            # Generate comparison using AI
            if self.chat_engine:
                prompt = f"""Compare the following two OSINT intelligence reports and identify:
1. Key similarities in digital footprint
2. Notable differences in online presence
3. Shared platforms or communities
4. Common security exposure points
5. Intelligence gaps present in both reports

REPORT 1: {target1}
{content1[:1500]}...

REPORT 2: {target2}
{content2[:1500]}...

Provide a factual, concise comparison highlighting the most significant findings.
"""
                system_message = self.chat_engine.security_system_message
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
                comparison = self.chat_engine.generate_response(messages, temperature=0.3, max_tokens=1024)
                
                # Display comparison
                self.console.print(Panel(
                    comparison,
                    title=f"OSINT Comparison: {target1} vs {target2}",
                    border_style="cyan",
                    width=100
                ))
            else:
                self.console.print("[yellow]AI comparison unavailable - API integration not configured.[/yellow]")
                
        except Exception as e:
            self.console.print(f"[bold red]Error comparing reports: {str(e)}[/bold red]")

    def search_person_data(self, query=None):
        """Search across all collected person OSINT data."""
        try:
            reports_dir = "Data/Person_OSINT"
            reports = [f for f in os.listdir(reports_dir) if f.endswith(".md")]
            
            if not reports:
                self.console.print("[yellow]No saved reports found to search.[/yellow]")
                return
                
            if not query:
                query = Prompt.ask("[bold green]>[/bold green] Enter search query")
                
            # Search across all reports
            matches = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task(description="Searching across OSINT reports...", total=len(reports))
                
                for report_file in reports:
                    with open(os.path.join(reports_dir, report_file), "r") as f:
                        content = f.read()
                        
                        # Extract target name
                        target = re.search(r"Target: (.*)", content)
                        target = target.group(1) if target else "Unknown"
                        
                        # Find matches
                        if query.lower() in content.lower():
                            # Find context for the match
                            matches.append({
                                "file": report_file,
                                "target": target,
                                "context": self._extract_match_context(content, query)
                            })
                            
                    progress.update(task, advance=1)
            
            # Display results
            if matches:
                table = Table(title=f"Search Results for '{query}'", show_header=True, header_style="bold magenta")
                table.add_column("Target", style="cyan")
                table.add_column("Context", style="green")
                table.add_column("File", style="dim")
                
                for match in matches:
                    table.add_row(match["target"], match["context"], match["file"])
                    
                self.console.print(table)
            else:
                self.console.print(f"[yellow]No matches found for '{query}'[/yellow]")
                
        except Exception as e:
            self.console.print(f"[bold red]Error searching reports: {str(e)}[/bold red]")
    
    def _extract_match_context(self, content, query, context_chars=50):
        """Extract context around search matches."""
        query_lower = query.lower()
        content_lower = content.lower()
        
        matches = []
        start_pos = 0
        
        while True:
            pos = content_lower.find(query_lower, start_pos)
            if pos == -1:
                break
                
            # Get context
            context_start = max(0, pos - context_chars)
            context_end = min(len(content), pos + len(query) + context_chars)
            
            context = content[context_start:context_end]
            if context_start > 0:
                context = "..." + context
            if context_end < len(content):
                context += "..."
                
            matches.append(context)
            start_pos = pos + len(query)
            
            # Limit to first 3 matches
            if len(matches) >= 3:
                break
                
        if matches:
            return matches[0]  # Return first match context
        return ""


class SimpleOSINTCLI:
    """Simple CLI interface for the Person OSINT module."""
    
    def __init__(self):
        self.console = Console()
        self.osint = PersonOSINT()
        
    def display_menu(self):
        """Display the main menu."""
        self.console.print("\n[bold cyan]===== PERSON OSINT TOOLKIT =====\n")
        
        menu = Table(show_header=False, box=None)
        menu.add_column(style="green", width=5)
        menu.add_column(style="white")
        
        menu.add_row("1", "Run Person OSINT Collection")
        menu.add_row("2", "List Saved Reports")
        menu.add_row("3", "Search Across Reports")
        menu.add_row("4", "Compare Two Reports")
        menu.add_row("q", "Quit")
        
        self.console.print(menu)
        return Prompt.ask("\n[bold green]>[/bold green] Select an option", choices=["1", "2", "3", "4", "q"])
        
    def run(self):
        """Run the CLI interface."""
        while True:
            choice = self.display_menu()
            
            if choice == "1":
                target = Prompt.ask("[bold green]>[/bold green] Target Person (full name)")
                depth = Prompt.ask(
                    "[bold green]>[/bold green] Collection Depth", 
                    choices=["basic", "standard", "deep"], 
                    default="standard"
                )
                self.osint.person_recon(target, depth)
                
            elif choice == "2":
                self.osint.list_saved_reports()
                
            elif choice == "3":
                self.osint.search_person_data()
                
            elif choice == "4":
                self.osint.compare_osint_reports()
                
            elif choice == "q":
                self.console.print("[yellow]Exiting Person OSINT Toolkit...[/yellow]")
                break
                
            # Pause before showing menu again
            input("\nPress Enter to continue...")
            
def integrate_person_osint(chat_engine=None, config=None):
    """Create and return an instance of PersonOSINT integrated with the chat engine."""
    from osint import PersonOSINT
    return PersonOSINT(chat_engine=chat_engine, config=config)            


if __name__ == "__main__":
    try:
        # Display app intro
        console = Console()
        console.print("\n[bold cyan]╔══════════════════════════════════════════════╗")
        console.print("[bold cyan]║                                              ║")
        console.print("[bold cyan]║  PERSON OSINT MODULE v1.0                    ║")
        console.print("[bold cyan]║  Intelligence Collection & Analysis Tool     ║")
        console.print("[bold cyan]║                                              ║")
        console.print("[bold cyan]╚══════════════════════════════════════════════╝\n")
        
        # If run directly, use simple CLI
        cli = SimpleOSINTCLI()
        cli.run()
        
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)
        
    except Exception as e:
        console = Console()
        console.print(f"\n[bold red]An error occurred: {str(e)}[/bold red]")
        sys.exit(1)
