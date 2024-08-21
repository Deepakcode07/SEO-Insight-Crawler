import os
import requests
from bs4 import BeautifulSoup   #for crawling
from urllib.parse import urljoin, urlparse
from pydantic import BaseModel, ConfigDict     #for data validation
from crewai import Agent, Task, Process, Crew   #crew ai works
from langchain_google_genai import ChatGoogleGenerativeAI    # to generate responses

# Load your API keys from environment variables or set them directly
gemini_api_key = os.environ.get("GOOGLE_API_KEY", "Replace with your google api key")

# Define a Pydantic base model with the new configuration
class MyBaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )

# Load the Gemini model using the Google API key
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.1, google_api_key=gemini_api_key
)

# Function to extract all internal links from a website
def get_internal_links(url, base_url):
    internal_links = set()
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href.startswith(('http://', 'https://')):  # Handle relative links
                href = urljoin(base_url, href)
            if urlparse(href).netloc == urlparse(base_url).netloc:  # Ensure it's the same domain
                internal_links.add(href)
    except Exception as e:
        print(f"Failed to process {url}: {e}")
    return internal_links

# Additional SEO checks to evaluate on-page SEO factors
def perform_seo_checks(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        seo_issues = []

        # Check for meta title and description
        title = soup.title.string if soup.title else None
        meta_description = soup.find('meta', attrs={'name': 'description'})

        if not title or len(title) < 10 or len(title) > 60:
            seo_issues.append("Meta title is missing or not optimized. Suggested length: 10-60 characters.")
        
        if not meta_description or len(meta_description['content']) < 50 or len(meta_description['content']) > 160:
            seo_issues.append("Meta description is missing or not optimized. Suggested length: 50-160 characters.")

        # Check headers (H1, H2, etc.)
        headers = soup.find_all(['h1', 'h2', 'h3'])
        if len(headers) == 0:
            seo_issues.append("No header tags found. Consider adding H1, H2, and H3 tags for better content structure.")
        else:
            for header in headers:
                if len(header.get_text()) > 70:
                    seo_issues.append(f"Header '{header.get_text()[:50]}...' is too long. Suggested length: <70 characters.")

        # Check for image alt attributes
        images = soup.find_all('img')
        for img in images:
            if not img.get('alt'):
                seo_issues.append("Some images are missing alt attributes. Add descriptive alt text for better accessibility and SEO.")

        # Check for broken links
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']
            if not href.startswith(('http://', 'https://')):  # Handle relative links
                href = urljoin(url, href)
            try:
                link_response = requests.head(href)
                if link_response.status_code >= 400:
                    seo_issues.append(f"Broken link found: {href}")
            except requests.exceptions.RequestException:
                seo_issues.append(f"Broken link found: {href}")

        # Check for keyword density
        content_text = soup.get_text().lower()
        keyword_density = {}
        for word in content_text.split():
            if word.isalpha():  # Simple keyword density check, can be expanded
                keyword_density[word] = keyword_density.get(word, 0) + 1

        # Add more checks as necessary

        return seo_issues

    except Exception as e:
        print(f"Failed to perform SEO checks on {url}: {e}")
        return []

# Define the SEO expert agent
seo_expert = Agent(
    role="SEO Expert",
    goal="Crawl the target website and its internal links to suggest the most effective keywords and provide suggestions for improving SEO.",
    backstory="""You are a highly skilled SEO professional with a deep understanding of search engine algorithms, keyword research, 
        and on-page optimization techniques. Your expertise lies in analyzing website content, crawling through web pages and internal links, 
        and identifying high-impact keywords that can significantly boost a website's visibility in search engine results. Your suggestions 
        are backed by advanced SEO techniques and tools, ensuring that the keywords recommended are not only relevant but also competitive 
        and effective in driving organic traffic. Additionally, you provide content replacements for elements that might be decreasing SEO rankings.""",
    verbose=True,
    allow_delegation=True,
    llm=llm_gemini
)

# Target website URL for SEO expert to crawl
target_website_url = "paste any website link here"  # Replace with the actual target website URL

# Get all internal links from the target website
all_links = get_internal_links(target_website_url, target_website_url)

# Perform SEO checks on the main URL and all internal links
all_seo_issues = {}
for link in all_links:
    seo_issues = perform_seo_checks(link)
    if seo_issues:
        all_seo_issues[link] = seo_issues

# Task description with all links included and SEO issues to address
task_description = f"""Crawl the target website {target_website_url} and all its internal links to analyze content and identify high-impact keywords 
    that can improve search engine rankings. Additionally, identify elements that might be decreasing SEO rankings and provide content replacements.
    Write a detailed report listing the most effective keywords, actionable recommendations for SEO improvements, and content replacement suggestions.
    The following internal links were crawled:
    {', '.join(all_links)}
"""

# Define the task for the SEO expert
task = Task(
    description=task_description,
    agent=seo_expert,
    expected_output="A detailed SEO report with at least 10 bullet points on high-impact keywords, actionable SEO recommendations, and content replacement suggestions."
)

# Define the crew and the process (only includes the SEO expert for this task)
crew = Crew(
    agents=[seo_expert],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
)

# Run the process
result = crew.kickoff()

print("######################")
print(result)