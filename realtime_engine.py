import re
import os
import json
import requests
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from urllib.parse import quote_plus
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from googlesearch import search
from chat_engine import ChatEngine

console = Console()

class RealTimeEngine:
    """Enhanced engine to identify and process real-time queries using targeted website scraping,
    with additional capabilities for cybersecurity and ethical hacking information."""

    def __init__(self, chat_engine: ChatEngine):
        self.chat_engine = chat_engine
        self.groq_api_key = os.environ.get("GroqAPIKey")
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        
        # Enhanced real-time query patterns
        self.realtime_patterns = [
            # Weather patterns
            r"(current|today|now|real[ -]?time)\s+(weather|forecast|temperature)",
            r"what('s| is)\s+(the\s+)?weather\s+(in|for|at)",
            
            # News patterns
            r"latest\s+(news|updates|headlines)",
            r"(recent|current|today|breaking)\s+news",
            r"what('s| is)\s+happening\s+(in|with)",
            
            # Financial patterns
            r"(stock|share)\s+price\s+(of|for)",
            r"(current|latest)\s+(stock|market|exchange)\s+(price|value|rate)",
            r"(crypto|bitcoin|ethereum|dogecoin)",
            
            # Time patterns
            r"(current|exact)\s+time\s+(in|for|at)",
            r"what\s+time\s+is\s+it\s+(in|at)",
            
            # Sports patterns
            r"(live|current|latest)\s+(score|match|game|result)",
            r"cricket\s+(score|match|update|result)",
            r"(ipl|nba|nfl|fifa|premier league|world cup)",
            
            # General real-time patterns
            r"(right now|currently|as of now|at this moment)",
            r"(today's|today|this morning|this evening|tonight)",
            r"ongoing",
            r"happening now",
            
            # Cybersecurity patterns
            r"(latest|recent|current|ongoing)\s+(cyber|security)\s+(threat|attack|breach|vulnerability|exploit)",
            r"(zero-day|0day|cve|exploit|vulnerability)\s+(latest|new|current)",
            r"(latest|recent)\s+(ransomware|malware|phishing|ddos)\s+(attack|campaign|threat)",
            r"(current|active)\s+(security|threat)\s+(advisory|alert|bulletin)",
            r"(latest|new)\s+cve(\s+|-)?\d+",
            r"(security|cyber|hack)\s+(news|update|alert)",
            r"(threat|vulnerability|exploit|patch)\s+(intelligence|report|update)",
            
            # Ethical hacking patterns
            r"(penetration|pen)\s+testing\s+(tool|technique|methodology)",
            r"(recent|new|latest)\s+(hack|breach|exploit|attack|vulnerability)",
            r"(security|vulnerability)\s+assessment",
            r"(bug|vulnerability)\s+bounty\s+(program|update|news)",
            r"(current|best|top)\s+(cybersecurity|infosec|hacking)\s+(tool|framework|practice|methodology)",
            r"(updated|latest)\s+(kali|parrot|metasploit|burp|wireshark)",
            r"(ethical|white hat)\s+hacking"
            
            # New hacking patterns
            r"hacking\s*(techniques|methods|tools|news)?",
            r"(latest|recent|current)\s+hacking"
        ]
        
        # Specialized websites for different categories
        self.specialized_sites = {
            "weather": [
                "https://weather.com/",
                "https://www.accuweather.com/",
                "https://www.wunderground.com/"
            ],
            "news": [
                "https://news.google.com/",
                "https://www.bbc.com/news",
                "https://www.reuters.com/",
                "https://apnews.com/"
            ],
            "stocks": [
                "https://finance.yahoo.com/",
                "https://www.marketwatch.com/",
                "https://www.investing.com/"
            ],
            "crypto": [
                "https://coinmarketcap.com/",
                "https://www.coingecko.com/",
                "https://cryptonews.com/"
            ],
            "cricket": [
                "https://www.cricbuzz.com/",
                "https://www.espncricinfo.com/",
                "https://www.icc-cricket.com/"
            ],
            "football": [
                "https://www.goal.com/",
                "https://www.espn.com/soccer/",
                "https://www.bbc.com/sport/football"
            ],
            "general_sports": [
                "https://www.espn.com/",
                "https://sports.yahoo.com/",
                "https://www.skysports.com/"
            ],
            # New cybersecurity sites
            "cybersecurity": [
                "https://thehackernews.com/",
                "https://www.bleepingcomputer.com/",
                "https://www.darkreading.com/",
                "https://www.securityweek.com/",
                "https://krebsonsecurity.com/"
            ],
            "vulnerabilities": [
                "https://nvd.nist.gov/vuln/search",
                "https://cve.mitre.org/",
                "https://www.cvedetails.com/",
                "https://www.zerodayinitiative.com/advisories/published/"
            ],
            "threat_intelligence": [
                "https://exchange.xforce.ibmcloud.com/",
                "https://attack.mitre.org/",
                "https://otx.alienvault.com/"
            ],
            "ethical_hacking": [
                "https://www.exploit-db.com/",
                "https://portswigger.net/daily-swig",
                "https://www.hackerone.com/vulnerability-and-security-testing-blog",
                "https://hackaday.com/"
            ],
            "security_tools": [
                "https://www.kali.org/blog/",
                "https://www.offensive-security.com/blog/",
                "https://nmap.org/",
                "https://www.metasploit.com/"
            ],
            # Add this to the specialized_sites dictionary in __init__
            "hacking": [
    "https://www.hackerone.com/",
    "https://www.offensive-security.com/blog/",
    "https://threatpost.com/",
    "https://www.hackread.com/",
    "https://www.kitploit.com/"
],
        }

    def is_realtime_query(self, query: str) -> bool:
        """Determine if the query requires real-time data with enhanced pattern matching."""
        query = query.lower().strip()
        
        category = self.identify_query_category(query)
        if category == "hacking":
          return True
        # Check against patterns
        for pattern in self.realtime_patterns:
            if re.search(pattern, query):
                return True
                
        # Check for time-sensitive keywords
        time_keywords = ["now", "today", "current", "latest", "live", "recent", "update", "hacking"]
        query_words = query.split()
        
        for keyword in time_keywords:
            if keyword in query_words:
                return True
                
        return False
    
    def identify_query_category(self, query: str) -> str:
        """Identify the category of the real-time query."""
        query = query.lower()
        if any(word in query for word in ["hack", "hacking", "hacked", "hacker", "bypassing", "jailbreak", 
                                    "root access", "privilege escalation", "backdoor"]):
              return "hacking"
        
        # Check for cybersecurity related queries
        if any(word in query for word in ["cyber", "hack", "security", "threat", "vulnerability", "malware", 
                                          "ransomware", "cve", "exploit", "breach", "phishing", ]):
            # Further categorize cybersecurity queries
            if any(word in query for word in ["cve", "vulnerability", "exploit", "patch", "zero-day", "0day"]):
                return "vulnerabilities"
            elif any(word in query for word in ["threat", "intelligence", "apt", "actor", "campaign", "ioc"]):
                return "threat_intelligence"
            elif any(word in query for word in ["ethical", "penetration", "pen test", "bug bounty", "white hat"]):
                return "ethical_hacking"
            elif any(word in query for word in ["tool", "kali", "metasploit", "burp", "nmap", "wireshark"]):
                return "security_tools"
            else:
                return "cybersecurity"
        
        # Check for weather related queries
        if any(word in query for word in ["weather", "temperature", "forecast", "rain", "sunny"]):
            return "weather"
            
        # Check for cricket related queries
        if any(word in query for word in ["cricket", "ipl", "t20", "test match", "odi"]):
            return "cricket"
            
        # Check for football/soccer related queries
        if any(word in query for word in ["football", "soccer", "premier league", "fifa", "uefa", "la liga"]):
            return "football"
            
        # Check for financial related queries
        if any(word in query for word in ["stock", "share", "market", "nasdaq", "dow", "s&p", "nyse"]):
            return "stocks"
            
        # Check for cryptocurrency related queries
        if any(word in query for word in ["crypto", "bitcoin", "ethereum", "blockchain", "coin"]):
            return "crypto"
            
        # Check for news related queries
        if any(word in query for word in ["news", "headlines", "breaking", "announcement"]):
            return "news"
            
        # Default to news if no specific category is identified
        return "news"

    def process_realtime_query(self, query: str, mode: str = "normal", category: str = None) -> str:
      """Process a real-time query using targeted website scraping based on query category.
     
    Args:
        query: The user's query text
        mode: Processing mode ("normal" or "augmented")
        category: Optional pre-defined category override (e.g., "ethical_hacking")
        
    Returns:
        Formatted response with real-time information
    """
      if not self.is_realtime_query(query):
        return "This is not a real-time query. Processing as a general query."

      try:
        # Add real-time context
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Identify query category if not provided
        if not category:
            category = self.identify_query_category(query)
        console.print(f"[info]Identified query category: {category}[/info]")
        
        # For cybersecurity queries, check for potential sensitive requests
        if category in ["cybersecurity", "vulnerabilities", "threat_intelligence", 
                        "ethical_hacking", "security_tools", "hacking"]:  # Added "hacking" category
            if self._is_sensitive_request(query):
                return self._handle_sensitive_request(query, mode)
        
        # Get content from specialized websites
        targeted_content = self._fetch_from_specialized_sites(query, category)
        
        # Get general search results as backup
        search_results = self._perform_search(query)
        search_content = self._format_search_results(search_results)
        
        # Combine specialized and general content
        combined_content = f"{targeted_content}\n\n{search_content}"
        
        # Prepare messages for the LLM
        system_message = f"""You are an expert assistant specializing in real-time {category} information.
        You have been provided with the latest data from specialized websites and search results about: "{query}".
        Extract the most relevant, accurate, and up-to-date information from these sources.
        Always provide attribution for specific facts and data points.
        If the sources don't contain relevant or recent information, be honest about it.
        Current time: {current_time}
        
        For cybersecurity information, focus on awareness, defense, and prevention rather than attack methodologies.
        Present only factual, educational information about vulnerabilities, threats, and security tools.
        When discussing ethical hacking concepts, emphasize the importance of proper authorization and legal compliance."""
        
        # First try to use Groq API if available
        if self.groq_api_key:
            response = self._query_groq(system_message, combined_content, query)
            if response:
                # Save to appropriate chat log
                self._save_to_chat_log(query, response, mode)
                return response
        
        # Fallback to ChatEngine
        messages = [
            {"role": "system", "content": system_message},
            {"role": "system", "content": combined_content},
            {"role": "user", "content": query}
        ]

        # Generate response
        response = self.chat_engine.generate_response(
            messages,
            model="llama3-70b-8192",  # Adjust this to your preferred model
            temperature=0.3,  # Lower temperature for factual responses
            max_tokens=1024
        )

        # Save to appropriate chat log
        self._save_to_chat_log(query, response, mode)
        return response

      except Exception as e:
        console.print(f"[error]Error processing real-time query: {str(e)}[/error]")
        return f"I encountered an issue while processing your real-time query. Please try again or rephrase your question. Technical detail: {str(e)}"
    
    def _is_sensitive_request(self, query: str) -> bool:
        """Identify potentially sensitive or malicious cybersecurity requests."""
        query = query.lower()
        
        # List of sensitive keywords that might indicate malicious intent
        malicious_keywords = [
            "hack into", "how to hack", "bypass security", "crack password", 
            "steal credentials", "bypass authentication", "ddos attack how to",
            "hack website", "hack account", "compromise system", "illegal hack",
            "destroy data", "bypass firewall", "hack wifi password"
        ]
        
        for keyword in malicious_keywords:
            if keyword in query:
                return True
                
        return False
    
    def _handle_sensitive_request(self, query: str, mode: str) -> str:
        """Handle potentially sensitive cybersecurity requests ethically."""
        # Generate an educational response that redirects to ethical alternatives
        system_message = """You are an expert cybersecurity educator focused on ethical practices.
        You've received a query that appears to be requesting potentially sensitive or harmful cybersecurity information.
        Provide an educational response that:
        1. Acknowledges the interest in the cybersecurity topic
        2. Redirects to ethical alternatives and educational resources
        3. Explains the importance of authorized testing and legal compliance
        4. Suggests constructive ways to learn about the topic through proper channels
        Do not provide specific techniques or instructions that could be used maliciously."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
        
        # Generate response using ChatEngine
        response = self.chat_engine.generate_response(
            messages,
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=1024
        )
        
        # Save to appropriate chat log
        self._save_to_chat_log(query, response, mode)
        return response

    def _fetch_from_specialized_sites(self, query: str, category: str) -> str:
        """Fetch information from specialized websites based on query category."""
        targeted_content = f"SPECIALIZED {category.upper()} INFORMATION:\n\n"
        
        # Get relevant sites for the category
        sites = self.specialized_sites.get(category, self.specialized_sites["news"])
        
        # Query terms
        query_terms = query.split()
        
        # Try each site until we get good content
        successful_fetches = 0
        
        for site in sites:
            try:
                # Modify site URL to include search if needed
                search_url = self._create_search_url(site, query)
                
                # Fetch content
                headers = {"User-Agent": self.user_agent}
                response = requests.get(search_url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    # Parse HTML
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Extract relevant content based on category
                    if category == "cricket":
                        content = self._extract_cricket_content(soup, query_terms)
                    elif category == "weather":
                        content = self._extract_weather_content(soup, query_terms)
                    elif category == "news":
                        content = self._extract_news_content(soup, query_terms)
                    elif category in ["stocks", "crypto"]:
                        content = self._extract_financial_content(soup, query_terms)
                    elif category == "vulnerabilities":
                        content = self._extract_vulnerability_content(soup, query_terms)
                    elif category == "cybersecurity":
                        content = self._extract_cybersecurity_content(soup, query_terms)
                    elif category == "threat_intelligence":
                        content = self._extract_threat_intel_content(soup, query_terms)
                    elif category == "ethical_hacking":
                        content = self._extract_ethical_hacking_content(soup, query_terms)
                    elif category == "security_tools":
                        content = self._extract_security_tools_content(soup, query_terms)
                    else:
                        content = self._extract_general_content(soup, query_terms)
                    
                    if content:
                        targeted_content += f"From {site}:\n{content}\n\n"
                        successful_fetches += 1
                        
                    # If we have content from at least 2 sites, that's enough
                    if successful_fetches >= 2:
                        break
                        
            except Exception as e:
                console.print(f"[warning]Error fetching from {site}: {str(e)}[/warning]")
                continue
        
        if successful_fetches == 0:
            targeted_content += "Could not retrieve specialized information from websites. Falling back to search results.\n"
            
        return targeted_content

    def _create_search_url(self, base_url: str, query: str) -> str:
        """Create appropriate search URL based on the site."""
        domain = base_url.split("//")[1].split("/")[0]
        
        # Site-specific search URL formats
        if "cricbuzz.com" in domain:
            return f"https://www.cricbuzz.com/search?q={quote_plus(query)}"
        elif "espncricinfo.com" in domain:
            return f"https://www.espncricinfo.com/search?query={quote_plus(query)}"
        elif "weather.com" in domain:
            # Extract location from query
            location_match = re.search(r"(?:in|at|for)\s+([a-zA-Z\s]+)(?:\s|$)", query)
            location = location_match.group(1) if location_match else query.split()[-1]
            return f"https://weather.com/weather/today/l/{quote_plus(location)}"
        elif "finance.yahoo.com" in domain:
            # Extract ticker/company from query
            company_match = re.search(r"(?:of|for)\s+([a-zA-Z\s]+)(?:\s|$)", query)
            company = company_match.group(1) if company_match else query.split()[-1]
            return f"https://finance.yahoo.com/quote/{quote_plus(company)}"
        elif "news.google.com" in domain:
            return f"https://news.google.com/search?q={quote_plus(query)}"
        elif "bbc.com" in domain:
            return f"https://www.bbc.co.uk/search?q={quote_plus(query)}"
        elif "thehackernews.com" in domain:
            return f"https://thehackernews.com/search?q={quote_plus(query)}"
        elif "bleepingcomputer.com" in domain:
            return f"https://www.bleepingcomputer.com/search/?q={quote_plus(query)}"
        elif "nvd.nist.gov" in domain:
            return f"https://nvd.nist.gov/vuln/search/results?form_type=Basic&results_type=overview&query={quote_plus(query)}"
        elif "cve.mitre.org" in domain:
            return f"https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword={quote_plus(query)}"
        elif "exploit-db.com" in domain:
            return f"https://www.exploit-db.com/search?q={quote_plus(query)}"
        elif "portswigger.net" in domain:
            return f"https://portswigger.net/daily-swig/search?q={quote_plus(query)}"
        elif "kali.org" in domain:
            return f"https://www.kali.org/search/?cx=partner-pub-2701916500055892%3A5450279765&cof=FORID%3A10&ie=UTF-8&q={quote_plus(query)}"
        
        # Default: just return the base URL
        return base_url

    def _extract_vulnerability_content(self, soup: BeautifulSoup, query_terms: List[str]) -> str:
        """Extract vulnerability specific content from the page."""
        content = ""
        
        # Look for CVE entries
        cve_elements = soup.select(".vulnDetail, .cveListRow, .searchResultsCell, .vuln-detail, .vulnerability-item")
        if cve_elements:
            for element in cve_elements[:3]:  # Limit to first 3
                # Extract CVE ID
                cve_id = element.select_one(".vulnDetailsTitleLink, .cveDetailsTitleLink, .cveID, .cve-id")
                if cve_id:
                    content += f"ID: {cve_id.get_text(strip=True)}\n"
                
                # Extract severity
                severity = element.select_one(".severityDetail, .cvss, .severity-level")
                if severity:
                    content += f"Severity: {severity.get_text(strip=True)}\n"
                
                # Extract description
                description = element.select_one(".vulnDetailsSummary, .cveDetailsSummary, .description, .vuln-summary")
                if description:
                    desc_text = description.get_text(strip=True)
                    content += f"Description: {desc_text[:300]}...\n" if len(desc_text) > 300 else f"Description: {desc_text}\n"
                
                # Extract published date
                published = element.select_one(".vulnDetailsDate, .publishedDate, .date-published")
                if published:
                    content += f"Published: {published.get_text(strip=True)}\n"
                
                content += "\n"
        
        # If no specific elements found, get general information
        if not content:
            content = self._extract_general_content(soup, query_terms)
            
        return content

    def _extract_cybersecurity_content(self, soup: BeautifulSoup, query_terms: List[str]) -> str:
        """Extract cybersecurity news specific content from the page."""
        content = ""
        
        # Look for article headlines and content
        articles = soup.select(".blog-post, .article, .news-item, .post-content, .entry-content")
        relevant_articles = []
        
        # If specific article elements found
        if articles:
            for article in articles[:5]:
                article_title = article.select_one("h1, h2, .post-title, .entry-title, .headline")
                article_text = ""
                
                if article_title:
                    article_title_text = article_title.get_text(strip=True)
                    
                    # Check if title contains any query terms
                    if any(term.lower() in article_title_text.lower() for term in query_terms):
                        article_text += f"Title: {article_title_text}\n"
                        
                        # Extract date if available
                        date_elem = article.select_one(".post-date, .date, .published-date, .time")
                        if date_elem:
                            article_text += f"Date: {date_elem.get_text(strip=True)}\n"
                        
                        # Extract summary or first paragraph
                        summary = article.select_one("p, .post-summary, .entry-summary, .article-summary")
                        if summary:
                            summary_text = summary.get_text(strip=True)
                            article_text += f"Summary: {summary_text[:300]}...\n" if len(summary_text) > 300 else f"Summary: {summary_text}\n"
                        
                        # Add to relevant articles if we have content
                        if article_text:
                            relevant_articles.append(article_text)
        
        # If we found relevant articles, format them
        for idx, article_text in enumerate(relevant_articles[:3]):
            content += f"Article {idx+1}:\n{article_text}\n"
        
        # If no specific elements found, get general information
        if not content:
            content = self._extract_general_content(soup, query_terms)
            
        return content

    def _extract_threat_intel_content(self, soup: BeautifulSoup, query_terms: List[str]) -> str:
        """Extract threat intelligence specific content from the page."""
        content = ""
        
        # Look for threat reports or IOC data
        threat_elements = soup.select(".threat-report, .intel-item, .attack-pattern, .indicator, .threat-actor")
        if threat_elements:
            for element in threat_elements[:3]:
                # Extract threat name/ID
                threat_id = element.select_one(".threat-id, .attack-id, .pattern-id, .indicator-id")
                if threat_id:
                    content += f"ID: {threat_id.get_text(strip=True)}\n"
                
                # Extract threat type
                threat_type = element.select_one(".threat-type, .attack-type, .indicator-type")
                if threat_type:
                    content += f"Type: {threat_type.get_text(strip=True)}\n"
                
                # Extract description
                description = element.select_one(".threat-description, .attack-description, .indicator-description")
                if description:
                    desc_text = description.get_text(strip=True)
                    content += f"Description: {desc_text[:300]}...\n" if len(desc_text) > 300 else f"Description: {desc_text}\n"
                
                # Extract additional info
                additional = element.select_one(".threat-details, .attack-details, .indicator-details")
                if additional:
                    add_text = additional.get_text(strip=True)
                    content += f"Details: {add_text[:200]}...\n" if len(add_text) > 200 else f"Details: {add_text}\n"
                
                content += "\n"
        
        # If no specific elements found, get general information
        if not content:
            content = self._extract_general_content(soup, query_terms)
            
        return content

    def _extract_ethical_hacking_content(self, soup: BeautifulSoup, query_terms: List[str]) -> str:
        """Extract ethical hacking specific content from the page."""
        content = ""
        
        # Look for exploit details or blog posts
        exploit_elements = soup.select(".exploit, .exploit-details, .vulnerability-details, .bug-bounty-report, .security-research")
        if exploit_elements:
            for element in exploit_elements[:3]:
                # Extract title/name
                title = element.select_one(".exploit-title, .vuln-title, .report-title, .research-title")
                if title:
                    content += f"Title: {title.get_text(strip=True)}\n"
                
                # Extract disclosure date
                date = element.select_one(".exploit-date, .disclosure-date, .report-date, .research-date")
                if date:
                    content += f"Date: {date.get_text(strip=True)}\n"
                
                # Extract details
                details = element.select_one(".exploit-details, .vuln-details, .report-details, .research-details")
                if details:
                    det_text = details.get_text(strip=True)
                    content += f"Details: {det_text[:300]}...\n" if len(det_text) > 300 else f"Details: {det_text}\n"
                
                # Extract CVSS or severity if available
                severity = element.select_one(".exploit-severity, .vuln-severity, .cvss")
                if severity:
                    content += f"Severity: {severity.get_text(strip=True)}\n"
                
                content += "\n"
        
        # If no specific elements found, try blog posts or articles
        if not content:
            blog_elements = soup.select(".blog-post, .article, .research-paper")
            for element in blog_elements[:3]:
                title = element.select_one("h1, h2, .post-title, .article-title")
                if title:
                    title_text = title.get_text(strip=True)
                    if any(term.lower() in title_text.lower() for term in query_terms):
                        content += f"Title: {title_text}\n"
                        
                        summary = element.select_one("p, .post-summary, .article-summary")
                        if summary:
                            sum_text = summary.get_text(strip=True)
                            content += f"Summary: {sum_text[:300]}...\n" if len(sum_text) > 300 else f"Summary: {sum_text}\n"
                        
                        content += "\n"
        
        # If still no content, get general information
        if not content:
            content = self._extract_general_content(soup, query_terms)
            
        return content

    def _extract_security_tools_content(self, soup: BeautifulSoup, query_terms: List[str]) -> str:
        """Extract security tools specific content from the page."""
        content = ""
        
        # Look for tool information
        tool_elements = soup.select(".tool-details, .tool-info, .software-details, .release-notes, .update-info")
        if tool_elements:
            for element in tool_elements[:3]:
                # Extract tool name
                tool_name = element.select_one(".tool-name, .software-name, .release-name")
                if tool_name:
                    content += f"Tool: {tool_name.get_text(strip=True)}\n"
                
                # Extract version
                version = element.select_one(".tool-version, .software-version, .release-version")
                if version:
                    content += f"Version: {version.get_text(strip=True)}\n"
                
                # Extract description
                description = element.select_one(".tool-description, .software-description, .release-description")
                if description:
                    desc_text = description.get_text(strip=True)
                    content += f"Description: {desc_text[:300]}...\n" if len(desc_text) > 300 else f"Description: {desc_text}\n"
                
                # Extract features or changes
                features = element.select_one(".tool-features, .software-features, .release-features, .changes")
                if features:
                    feat_text = features.get_text(strip=True)
                    content += f"Features: {feat_text[:200]}...\n" if len(feat_text) > 200 else f"Features: {feat_text}\n"
                
                content += "\n"
        
        # If no specific elements found, try blog posts or documentation
        if not content:
            doc_elements = soup.select(".documentation, .docs, .guide, .tutorial")
            for element in doc_elements[:3]:
                title = element.select_one("h1, h2, .section-title")
                if title:
                    title_text = title.get_text(strip=True)
                    if any(term.lower() in title_text.lower() for term in query_terms):
                        content += f"Title: {title_text}\n"
                        
                        summary = element.select_one("p, .section-content")
                        if summary:
                            sum_text = summary.get_text(strip=True)
                            content += f"Content: {sum_text[:300]}...\n" if len(sum_text) > 300 else f"Content: {sum_text}\n"
                        
                        content += "\n"
        
        # If still no content, get general information
        if not content:
            content = self._extract_general_content(soup, query_terms)
            
        return content

    def _extract_cricket_content(self, soup: BeautifulSoup, query_terms: List[str]) -> str:
        """Extract cricket specific content from the page."""
        content = ""
        
        # Look for live scores
        live_scores = soup.select(".cb-scr-wll-chvrn, .match-score, .score-detail, .cb-col-scores")
        if live_scores:
            for score in live_scores[:2]:  # Limit to first 2
                score_text = score.get_text(strip=True)
                content += f"Score: {score_text}\n"
                
        # Look for match status
        match_status = soup.select_one(".cb-text-complete, .match-status, .status-text")
        if match_status:
            content += f"Status: {match_status.get_text(strip=True)}\n"
            
        # Look for team names
        teams = soup.select(".cb-team-nm, .team-name, .cscore_team")
        if teams and len(teams) >= 2:
            content += f"Teams: {teams[0].get_text(strip=True)} vs {teams[1].get_text(strip=True)}\n"
            
        # If specific cricket elements not found, get general content
        if not content:
            content = self._extract_general_content(soup, query_terms)
            
        return content

    def _extract_weather_content(self, soup: BeautifulSoup, query_terms: List[str]) -> str:
        """Extract weather specific content from the page."""
        content = ""
        
        # Look for current temperature
        temp = soup.select_one(".CurrentConditions--tempValue--MHmYY, .temp, .current-temp, .today-temp")
        if temp:
            content += f"Temperature: {temp.get_text(strip=True)}\n"
            
        # Look for weather condition
        condition = soup.select_one(".CurrentConditions--phraseValue--mZC_p, .condition, .weather-phrase")
        if condition:
            content += f"Condition: {condition.get_text(strip=True)}\n"
            
        # Look for location
        location = soup.select_one(".CurrentConditions--location--2_osB, .location, .loc-name")
        if location:
            content += f"Location: {location.get_text(strip=True)}\n"
            
        # Look for additional details
        details = soup.select(".WeatherDetailsListItem--wxData--kK35q, .detail-item, .weather-detail")
        if details:
            for detail in details[:4]:  # Limit to first 4 details
                detail_text = detail.get_text(strip=True)
                content += f"{detail_text}\n"
                
        # If specific weather elements not found, get general content
        if not content:
            content = self._extract_general_content(soup, query_terms)
            
        return content

    def _extract_news_content(self, soup: BeautifulSoup, query_terms: List[str]) -> str:
        """Extract news specific content from the page."""
        content = ""
        
        # Look for news headlines
        headlines = soup.select("h3.ipQwMb, .story-heading, .headline, .title, article h3")
        
        if headlines:
            content += "Headlines:\n"
            headline_count = 0
            
            for headline in headlines:
                headline_text = headline.get_text(strip=True)
                
                # Check if headline contains any query terms or if we're just getting all headlines
                if not query_terms or any(term.lower() in headline_text.lower() for term in query_terms):
                    content += f"- {headline_text}\n"
                    headline_count += 1
                    
                    # Try to get article date
                    date_elem = headline.find_next(".time, .date, .timestamp")
                    if date_elem:
                        content += f"  Published: {date_elem.get_text(strip=True)}\n"
                        
                    # Try to get article snippet
                    snippet = headline.find_next("p, .snippet, .summary, .desc")
                    if snippet:
                        snippet_text = snippet.get_text(strip=True)
                        content += f"  {snippet_text[:150]}...\n" if len(snippet_text) > 150 else f"  {snippet_text}\n"
                        
                    content += "\n"
                    
                    # Limit to 5 headlines max
                    if headline_count >= 5:
                        break
                        
        # If no headlines found, get general content
        if not content:
            content = self._extract_general_content(soup, query_terms)
            
        return content

    def _extract_financial_content(self, soup: BeautifulSoup, query_terms: List[str]) -> str:
        """Extract financial (stocks/crypto) specific content from the page."""
        content = ""
        
        # Look for ticker/symbol
        ticker = soup.select_one(".D(ib) h1, .symbol, .ticker, .coin-symbol")
        if ticker:
            content += f"Symbol: {ticker.get_text(strip=True)}\n"
            
        # Look for current price
        price = soup.select_one(".Fw\\(b\\).Fz\\(36px\\), .price, .coin-price, .last-price")
        if price:
            content += f"Price: {price.get_text(strip=True)}\n"
            
        # Look for price change
        change = soup.select_one(".Fw\\(500\\).Pstart\\(8px\\).Fz\\(24px\\), .change, .price-change")
        if change:
            content += f"Change: {change.get_text(strip=True)}\n"
            
        # Look for market cap
        mcap = soup.select_one(".market-cap, .cap, .coin-cap")
        if mcap:
            content += f"Market Cap: {mcap.get_text(strip=True)}\n"
            
        # Look for additional details
        details = soup.select("td.Ta\\(end\\).Fw\\(600\\).Lh\\(14px\\), .data-row, .detail-row")
        if details:
            detail_count = 0
            for detail in details:
                detail_text = detail.get_text(strip=True)
                content += f"{detail_text}\n"
                detail_count += 1
                
                # Limit to 8 details max
                if detail_count >= 8:
                    break
                    
        # If specific financial elements not found, get general content
        if not content:
            content = self._extract_general_content(soup, query_terms)
            
        return content

    def _extract_general_content(self, soup: BeautifulSoup, query_terms: List[str]) -> str:
        """Extract general content that might be relevant to the query."""
        content = ""
        
        # Look for header content
        headers = soup.select("h1, h2, h3")
        relevant_headers = []
        
        for header in headers:
            header_text = header.get_text(strip=True)
            
            # Check if header contains any query terms
            if any(term.lower() in header_text.lower() for term in query_terms):
                # Get the next paragraph if available
                next_p = header.find_next("p")
                if next_p:
                    p_text = next_p.get_text(strip=True)
                    relevant_headers.append(f"Section: {header_text}\nContent: {p_text}\n")
        
        # Take the first 3 relevant headers
        for header_content in relevant_headers[:3]:
            content += header_content + "\n"
            
        # If no relevant headers found, look for paragraphs containing query terms
        if not content:
            paragraphs = soup.select("p")
            relevant_paragraphs = []
            
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                
                # Check if paragraph contains any query terms
                if any(term.lower() in p_text.lower() for term in query_terms):
                    # Limit paragraph length
                    p_text = p_text[:300] + "..." if len(p_text) > 300 else p_text
                    relevant_paragraphs.append(p_text)
            
            # Take the first 3 relevant paragraphs
            for p_content in relevant_paragraphs[:3]:
                content += f"{p_content}\n\n"
                
        # If still no content, grab the first few paragraphs
        if not content:
            paragraphs = soup.select("p")
            for p in paragraphs[:3]:
                p_text = p.get_text(strip=True)
                if p_text and len(p_text) > 20:  # Only include substantial paragraphs
                    content += f"{p_text}\n\n"
                    
        return content

    def _perform_search(self, query: str) -> List[Dict[str, str]]:
        """Perform a web search using the googlesearch package."""
        try:
            search_results = []
            
            # Perform search
            for url in search(query, num_results=5, advanced=True):
                result = {
                    "title": url.title or url.url,
                    "url": url.url,
                    "description": url.description or "No description available"
                }
                search_results.append(result)
                
            return search_results
            
        except Exception as e:
            console.print(f"[warning]Error during search: {str(e)}[/warning]")
            return []

    def _format_search_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results into text for the LLM."""
        if not results:
            return "No search results available."
            
        content = "SEARCH RESULTS:\n\n"
        
        for i, result in enumerate(results):
            content += f"Result {i+1}:\nTitle: {result['title']}\nURL: {result['url']}\nDescription: {result['description']}\n\n"
            
        return content

    def _query_groq(self, system_message: str, content: str, query: str) -> Optional[str]:
        """Query the Groq API for faster responses if available."""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama3-70b-8192",  # Use Mixtral for real-time queries
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "system", "content": content},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.2,
                "max_tokens": 1500
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            else:
                console.print(f"[warning]Groq API error: {response.status_code} - {response.text}[/warning]")
                return None
                
        except Exception as e:
            console.print(f"[warning]Error using Groq API: {str(e)}[/warning]")
            return None

    def _save_to_chat_log(self, query: str, response: str, mode: str) -> None:
        """Save the query and response to the appropriate chat log."""
        try:
            # Different log files for normal vs augmented mode
            log_file = "realtime_log.jsonl" if mode == "normal" else "augmented_realtime_log.jsonl"
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "mode": mode
            }
            
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            console.print(f"[warning]Error saving to chat log: {str(e)}[/warning]")