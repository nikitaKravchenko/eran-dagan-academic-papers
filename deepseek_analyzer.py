import json
import logging
from typing import Dict

import ollama


class DeepSeekAnalyzer:
    """Uses DeepSeek-R1 for intelligent author and affiliation analysis"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        logging.info(f"Initializing DeepSeek analyzer with model: {model_name}")
        logging.info(f"DeepSeek analyzer initialized: {model_name}")

    def test_connection(self) -> bool:
        """Test if DeepSeek model is available"""
        logging.info("Testing connection to DeepSeek R1...")
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'Hello! Just testing connection.'}],
                options={'num_predict': 10}
            )
            logging.info("DeepSeek R1 is connected and ready!")
            logging.info("DeepSeek connection test successful")
            return True
        except Exception as e:
            logging.warning(f"DeepSeek connection failed: {e}")
            logging.warning("   Please make sure DeepSeek R1 is installed: ollama pull deepseek-r1")
            logging.error(f"DeepSeek connection test failed: {e}")
            return False

    def analyze_authors_and_affiliation(self, title: str, authors: str, abstract: str = "") -> Dict:
        """Use DeepSeek to identify Israeli authors and their universities"""
        logging.info(f"""
            PAPER DETAILS:
                Title: {title}
                Authors: {authors}
            """)
        prompt = f"""You are an expert in identifying Israeli researchers and institutions. Analyze this AI/ML research paper.

    PAPER DETAILS:
    Title: {title}
    Authors: {authors}
    Abstract: {abstract[:800]}{"..." if len(abstract) > 800 else ""}

    ISRAELI RESEARCH INDICATORS:
        ️ Universities: Hebrew University of Jerusalem, Technion-Israel Institute of Technology, Tel Aviv University, Weizmann Institute of Science, Ben-Gurion University, Bar-Ilan University, University of Haifa, Open University of Israel, Ariel University, Academic College of Tel Aviv-Yafo

         Companies: Google Israel, Microsoft Israel, Intel Israel, IBM Israel, NVIDIA Israel, Meta Israel, Apple Israel, Amazon Israel, Mobileye, Check Point, Wix, Monday.com, Rafael, Elbit Systems

         Research Centers: Allen Institute for AI (some Israeli researchers), Israeli AI labs, defense research institutes

         Email domains: .ac.il, technion.ac.il, tau.ac.il, weizmann.ac.il, bgu.ac.il, huji.ac.il, etc.

     ISRAELI NAME PATTERNS: 
    - Hebrew/Jewish names: Or, Gil, Tal, Roi, Nir, Yael, Michal, Dror, Omer, Itai, Kfir
    - Names ending in -nik, -sky, -man, -berg, -stein are often Jewish
    - Be more generous with Hebrew first names like "Or" combined with Jewish surnames

     Locations: Israel, Jerusalem, Tel Aviv, Haifa, Beer Sheva, Herzliya, Ramat Gan, Petah Tikva, Rehovot

    ANALYSIS INSTRUCTIONS:
    1. Look for INSTITUTIONAL affiliations first (most reliable)
    2. Check for .il email domains
    3. Look for Israeli company affiliations
    4. Consider Hebrew first names (Or, Gil, Tal, etc.) with ANY surname as potential Israeli
    5. Be more generous - lower confidence threshold to 0.5 if name patterns suggest Israeli origin
    6. If you find clear Hebrew names like "Or Patashnik" or "Kfir Aberman", mark as Israeli even without explicit affiliation

    IMPORTANT: Respond ONLY with valid JSON. Do not include any reasoning or explanatory text outside the JSON.

    REQUIRED JSON RESPONSE:
    {{
        "israeli_authors": [
            {{
                "name": "Author Full Name",
                "university": "Unknown", 
                "confidence": 0.85,
                "reasoning": "Brief explanation"
            }}
        ],
        "is_israeli_research": true,
        "overall_confidence": 0.8
    }}

    If no clear Israeli connection found, return empty israeli_authors array and false for is_israeli_research."""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': 800
                }
            )

            content = response['message']['content'].strip()

            # Handle DeepSeek-R1 reasoning tokens
            # Remove <think> blocks if present
            if '<think>' in content:
                # Find the end of the thinking block
                think_end = content.find('</think>')
                if think_end != -1:
                    content = content[think_end + 8:].strip()

            # Remove any markdown code blocks
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # Find JSON boundaries more robustly
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]

                logging.info(json_str)

                result = json.loads(json_str)

                # Validate and fix result structure
                if not isinstance(result.get('israeli_authors'), list):
                    result['israeli_authors'] = []
                if 'is_israeli_research' not in result:
                    result['is_israeli_research'] = len(result['israeli_authors']) > 0
                if 'overall_confidence' not in result:
                    result['overall_confidence'] = 0.0

                # Validate each author entry
                for author in result['israeli_authors']:
                    if 'name' not in author:
                        author['name'] = 'Unknown'
                    if 'university' not in author:
                        author['university'] = 'Unknown'
                    if 'confidence' not in author:
                        author['confidence'] = 0.5
                    if 'reasoning' not in author:
                        author['reasoning'] = 'Name pattern analysis'

                return result
            else:
                logging.warning(f"No valid JSON found in DeepSeek response: {content[:200]}...")
                return self._empty_result()

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            logging.error(f"Cleaned JSON string: {json_str if 'json_str' in locals() else 'Not available'}")
            logging.error(f"Raw response: {content[:500]}")
            return self._empty_result()
        except Exception as e:
            logging.error(f"Error analyzing with DeepSeek: {e}")
            return self._empty_result()

    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            "israeli_authors": [],
            "is_israeli_research": False,
            "overall_confidence": 0.0
        }
