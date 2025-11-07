import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict
import json

class SHLCatalogCrawler:
    def __init__(self):
        self.base_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def crawl_catalog(self) -> List[Dict]:
        """Crawl all individual test solutions from SHL catalog"""
        assessments = []
        
        # Get main catalog page
        response = requests.get(self.base_url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all assessment links (excluding pre-packaged solutions)
        # Note: You'll need to identify the correct CSS selectors by inspecting the page
        assessment_links = soup.find_all('a', class_='assessment-link')  # Adjust selector
        
        for link in assessment_links:
            try:
                assessment_url = link.get('href')
                if not assessment_url.startswith('http'):
                    assessment_url = f"https://www.shl.com{assessment_url}"
                
                # Crawl individual assessment page
                assessment_data = self.crawl_assessment_page(assessment_url)
                if assessment_data:
                    assessments.append(assessment_data)
                
                time.sleep(1)  # Be respectful to the server
            except Exception as e:
                print(f"Error crawling {link}: {e}")
                
        return assessments
    
    def crawl_assessment_page(self, url: str) -> Dict:
        """Extract detailed information from individual assessment page"""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract assessment details (adjust selectors based on actual HTML)
            name = soup.find('h1').text.strip() if soup.find('h1') else "Unknown"
            
            # Extract description
            description_div = soup.find('div', class_='description')
            description = description_div.text.strip() if description_div else ""
            
            # Extract duration
            duration = self.extract_duration(soup)
            
            # Extract test type
            test_type = self.extract_test_type(soup)
            
            # Extract skills/competencies
            skills = self.extract_skills(soup)
            
            # Extract all text content for embeddings
            full_text = soup.get_text(separator=' ', strip=True)
            
            return {
                'name': name,
                'url': url,
                'description': description,
                'duration': duration,
                'test_type': test_type,
                'skills': skills,
                'full_text': full_text
            }
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return None
    
    def extract_duration(self, soup) -> int:
        """Extract test duration in minutes"""
        # Implement based on actual HTML structure
        return 0
    
    def extract_test_type(self, soup) -> str:
        """Extract test type (K, P, etc.)"""
        # Implement based on actual HTML structure
        return ""
    
    def extract_skills(self, soup) -> List[str]:
        """Extract assessed skills/competencies"""
        # Implement based on actual HTML structure
        return []
    
    def save_to_csv(self, assessments: List[Dict], filename: str):
        """Save crawled data to CSV"""
        df = pd.DataFrame(assessments)
        df.to_csv(filename, index=False)
        print(f"Saved {len(assessments)} assessments to {filename}")

if __name__ == "__main__":
    crawler = SHLCatalogCrawler()
    assessments = crawler.crawl_catalog()
    crawler.save_to_csv(assessments, '../data/raw/shl_assessments.csv')
