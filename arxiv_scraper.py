import requests
import feedparser
import pandas as pd
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import logging
from urllib.parse import quote

from config import Config
from db_manager import DatabaseManager
from deepseek_analyzer import DeepSeekAnalyzer
from vector_index import VectorIndex


class ArxivScraper:
    """ArXiv scraper with monthly data collection"""

    def __init__(self, config: Config):
        # Setup logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('arxiv_scraper.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info("=" * 60)
        self.logger.info("ArXiv Israeli GenAI Researchers Scraper v3.0-MONTHLY")
        self.logger.info("Systematic collection by periods")
        self.logger.info("=" * 60)

        self.config = config
        self.db = DatabaseManager(config.db_path)
        self.vector_index = VectorIndex(config.vector_model_name)
        self.deepseek_analyzer = DeepSeekAnalyzer(config.deepseek_model)

        self.base_url = "http://export.arxiv.org/api/query"
        self.batch_size = 2000  # maximum allowed
        self.api_wait = 3  # delay between requests (API recommends no more than once per 3 sec)

        # Extended list of categories
        self.categories_query = (
            "cat:cs.CV OR cat:cs.LG OR cat:cs.AI OR cat:cs.CL OR cat:cs.NE OR "
            "cat:stat.ML OR cat:cs.GR OR cat:cs.MM OR cat:cs.HC OR cat:cs.SD"
        )
        self.csv_file_name = f"israeli_genai_researchers_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

        # Create output directory
        config.output_dir.mkdir(exist_ok=True)
        self.logger.info(f"Output directory ready: {config.output_dir}")

        self.logger.info("ArXiv scraper v3.0-MONTHLY initialized successfully!")
        self.logger.info("=" * 60)

    def fetch_papers_by_month(self, year: int, month: int) -> List[Dict]:
        """
        Downloads all papers for a specific month
        """
        start_date = f"{year}{month:02d}010000"
        end_date = f"{year}{month:02d}312359"

        date_filter = f"submittedDate:[{start_date} TO {end_date}]"
        full_query = f"({self.categories_query}) AND {date_filter}"

        self.logger.info(f"Collecting papers for {year}-{month:02d}")
        self.logger.debug(f"Query: {full_query}")

        start = 0
        all_papers = []
        batch_count = 0

        while True:
            batch_count += 1
            url = f"{self.base_url}?search_query={quote(full_query)}&start={start}&max_results={self.batch_size}&sortBy=submittedDate&sortOrder=descending"

            self.logger.debug(f"Fetching batch {batch_count} (records {start}-{start + self.batch_size})")

            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                # Use feedparser for better parsing
                feed = feedparser.parse(response.text)

                if not feed.entries:
                    self.logger.info(f"No more records for {year}-{month:02d}")
                    break

                batch_papers = []
                for entry in feed.entries:
                    paper = self._parse_feedparser_entry(entry)
                    if paper:
                        batch_papers.append(paper)

                self.logger.debug(f"Retrieved {len(batch_papers)} papers in batch {batch_count}")
                all_papers.extend(batch_papers)

                # If we got fewer than batch_size, probably reached the end
                if len(feed.entries) < self.batch_size:
                    self.logger.debug("Got fewer papers than requested - end of period")
                    break

                start += self.batch_size

                # Delay between requests
                time.sleep(self.api_wait)

            except Exception as e:
                self.logger.error(f"Error in batch {batch_count}: {e}")
                time.sleep(10)  # Longer delay after error
                continue

        self.logger.info(f"Completed collection for {year}-{month:02d}: {len(all_papers)} papers")
        return all_papers

    def _parse_feedparser_entry(self, entry) -> Optional[Dict]:
        """Parse individual paper from feedparser"""
        try:
            # ArXiv ID
            arxiv_id = entry.get("id", "").split('/')[-1] if entry.get("id") else ""

            # Basic information
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            published = entry.get("published", "")
            updated = entry.get("updated", "")

            # Authors
            authors = []
            if hasattr(entry, 'authors'):
                authors = [author.name for author in entry.authors]

            # Categories
            categories = []
            if hasattr(entry, 'tags'):
                categories = [tag.term for tag in entry.tags]

            return {
                'arxiv_id': arxiv_id,
                'title': title,
                'abstract': summary,
                'authors': authors,
                'published': published,
                'updated': updated,
                'categories': categories,
                'authors_str': '; '.join(authors)
            }

        except Exception as e:
            self.logger.error(f"Error parsing feedparser entry: {e}")
            return None

    def collect_and_process_by_date_range(self, start_year: int = 2020, end_year: int = None) -> List[Dict]:
        """
        Collects papers by months and processes them immediately after each month
        """
        if end_year is None:
            end_year = datetime.now().year

        self.logger.info("=" * 70)
        self.logger.info("PAPER COLLECTION AND PROCESSING BY PERIODS")
        self.logger.info("=" * 70)
        self.logger.info(f"Period: {start_year} - {end_year}")
        self.logger.info(f"Categories: {self.categories_query}")
        self.logger.info("Processing starts immediately after each month collection")
        self.logger.info("=" * 70)

        all_results = []
        all_collected_papers = []
        total_months = (end_year - start_year + 1) * 12
        current_month = 0

        for year in range(start_year, end_year + 1):
            month_range = range(1, 13)

            # For current year limit to current month
            if year == datetime.now().year:
                month_range = range(1, datetime.now().month + 1)

            for month in month_range:
                current_month += 1
                self.logger.info(
                    f"PROCESSING MONTH {current_month}/{total_months if year != datetime.now().year else current_month}")

                try:
                    # Step 1: Collect papers for this month
                    month_papers = self.fetch_papers_by_month(year, month)

                    if month_papers:
                        # Remove duplicates within this month's collection
                        existing_ids = {paper['arxiv_id'] for paper in all_collected_papers}
                        unique_papers = []

                        for paper in month_papers:
                            if paper['arxiv_id'] not in existing_ids:
                                unique_papers.append(paper)
                                existing_ids.add(paper['arxiv_id'])

                        duplicates = len(month_papers) - len(unique_papers)
                        self.logger.info(f"Month collection - Unique: {len(unique_papers)}, duplicates: {duplicates}")

                        all_collected_papers.extend(unique_papers)

                        # Step 2: Process this month's papers immediately
                        if unique_papers:
                            self.logger.info(
                                f"Starting processing of {len(unique_papers)} papers from {year}-{month:02d}")
                            month_results = self.process_papers(unique_papers)

                            if month_results:
                                all_results.extend(month_results)
                                self.logger.info(
                                    f"Month processing complete - Found {len(month_results)} Israeli researchers")

                                # Save incremental results after each month
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                month_filename = f"results_month_{year}_{month:02d}_{timestamp}.csv"
                                self.save_to_csv(month_results, month_filename)
                            else:
                                self.logger.info("Month processing complete - No Israeli researchers found")

                        self.logger.info(f"Total collected papers so far: {len(all_collected_papers):,}")
                        self.logger.info(f"Total researchers found so far: {len(all_results)}")

                        # Save checkpoint of all collected papers every 3 months
                        if current_month % 3 == 0:
                            self.save_incremental_papers(all_collected_papers, f"checkpoint_{year}_{month:02d}")

                except Exception as e:
                    self.logger.error(f"Error processing {year}-{month:02d}: {e}")

                # Pause between months
                if current_month < total_months:
                    self.logger.debug("Pausing 5s before next month...")
                    time.sleep(5)

        self.logger.info("=" * 70)
        self.logger.info("PERIOD COLLECTION AND PROCESSING COMPLETED!")
        self.logger.info(f"Total unique papers collected: {len(all_collected_papers):,}")
        self.logger.info(f"Total Israeli researchers found: {len(all_results)}")
        self.logger.info("=" * 70)

        return all_results

    def save_incremental_papers(self, papers: List[Dict], checkpoint_name: str):
        """Save intermediate collection results"""
        if not papers:
            return

        filename = f"papers_checkpoint_{checkpoint_name}.csv"
        filepath = self.config.output_dir / filename

        df = pd.DataFrame(papers)
        df.to_csv(filepath, index=False, encoding='utf-8')
        self.logger.info(f"Checkpoint saved: {len(papers):,} papers")

    def improved_genai_filter(self, title: str, abstract: str) -> bool:
        """Improved filter for GenAI relevance"""
        text = (title + " " + abstract).lower()

        # Check explicit GenAI keywords
        for keyword in self.config.gen_ai_keywords:
            if keyword in text:
                return True

        # Additional patterns for implicit GenAI relevance
        genai_patterns = [
            r'neural.*network.*generat',
            r'deep.*learn.*generat',
            r'artificial.*intelligen.*generat',
            r'machine.*learn.*generat',
            r'automat.*generat',
            r'synth.*data',
            r'learn.*represent',
            r'unsuper.*learn',
            r'transformer.*model',
            r'attention.*mechanism',
            r'diffusion.*model',
            r'variational.*autoencoder'
        ]

        for pattern in genai_patterns:
            if re.search(pattern, text):
                return True

        return False

    def process_papers(self, papers: List[Dict]) -> List[Dict]:
        results = []
        processed_count = 0
        genai_relevant_count = 0
        israeli_papers_count = 0
        keyword_pass_count = 0
        vector_pass_count = 0

        self.logger.info("STARTING PAPER PROCESSING")
        self.logger.info("=" * 50)
        self.logger.info(f"Total papers to process: {len(papers)}")
        self.logger.info(f"Relevance threshold: {self.config.relevance_threshold}")
        self.logger.info("=" * 50)

        for i, paper in enumerate(papers):
            if (i + 1) % 100 == 0:
                self.logger.debug(f"[{i + 1}/{len(papers)}] Processing checkpoint...")

            # Skip if already processed
            if self.db.is_paper_processed(paper['arxiv_id']):
                continue

            processed_count += 1

            # Step 1: GenAI relevance filter
            if not self.improved_genai_filter(paper['title'], paper['abstract']):
                self.db.save_processed_paper(
                    paper['arxiv_id'], paper['title'],
                    paper['authors_str'], False, similarity_score=0.0
                )
                continue

            keyword_pass_count += 1

            # Step 2: Vector relevance check
            combined_text = f"{paper['title']} {paper['abstract']}"
            is_relevant, similarity_score = self.vector_index.is_relevant_to_genai(
                combined_text, self.config.relevance_threshold
            )

            if not is_relevant:
                self.db.save_processed_paper(
                    paper['arxiv_id'], paper['title'],
                    paper['authors_str'], False, similarity_score=similarity_score
                )
                continue

            genai_relevant_count += 1
            vector_pass_count += 1

            # Step 3: DeepSeek analysis (commented for speed)
            deepseek_result = self.deepseek_analyzer.analyze_authors_and_affiliation(
                paper['title'], paper['authors_str'], paper['abstract']
            )

            if deepseek_result['is_israeli_research'] and deepseek_result['israeli_authors']:
                self.logger.info(f"FOUND ISRAELI RESEARCH: {paper['title'][:50]}...")

                for author_info in deepseek_result['israeli_authors']:
                    self.logger.info(f"   • {author_info['name']} ({author_info.get('university', 'Unknown')})")

                    new_result = {
                        'arxiv_id': paper['arxiv_id'],
                        'paper_title': paper['title'],
                        'author_name': author_info['name'],
                        'university': author_info.get('university', 'Unknown'),
                        'confidence': author_info.get('confidence', 0.0),
                        'reasoning': author_info.get('reasoning', ''),
                        'published_date': paper['published'],
                        'categories': ', '.join(paper.get('categories', [])),
                        'all_authors': paper['authors_str'],
                        'similarity_score': similarity_score
                    }
                    results.append(new_result)

                self.save_incremental_results(results)
                israeli_papers_count += 1

                israeli_authors_str = json.dumps(deepseek_result['israeli_authors'])
                self.db.save_processed_paper(
                    paper['arxiv_id'], paper['title'],
                    paper['authors_str'], True, True,
                    israeli_authors_str, deepseek_result['overall_confidence'],
                    similarity_score
                )
            else:
                self.db.save_processed_paper(
                    paper['arxiv_id'], paper['title'],
                    paper['authors_str'], True, False,
                    similarity_score=similarity_score
                )

            # Progress update
            if (i + 1) % 50 == 0:
                self.logger.info("PROGRESS UPDATE:")
                self.logger.info(f"   Papers processed: {processed_count}")
                self.logger.info(f"   Passed keyword filter: {keyword_pass_count}")
                self.logger.info(f"   Passed vector filter: {vector_pass_count}")
                self.logger.info(f"   GenAI relevant: {genai_relevant_count}")
                self.logger.info(f"   Israeli papers: {israeli_papers_count}")
                self.logger.info(f"   Israeli researchers found: {len(results)}")

            if processed_count % 5 == 0:
                time.sleep(1)

        self.logger.info("PROCESSING COMPLETED!")
        self.logger.info(f"Papers processed: {processed_count}")
        self.logger.info(f"Israeli researchers found: {len(results)}")

        return results

    def save_incremental_results(self, results: List[Dict]):
        """Save results as they come in"""
        if not results:
            return

        filepath = self.config.output_dir / self.csv_file_name

        df = pd.DataFrame(results)
        df = df.drop_duplicates(subset=['author_name', 'paper_title'])
        df = df.sort_values(['confidence', 'similarity_score', 'published_date'],
                            ascending=[False, False, False])

        df.to_csv(filepath, index=False, encoding='utf-8')
        self.logger.debug(f"Live results updated: {len(df)} researchers")

    def save_to_csv(self, results: List[Dict], filename: str = None) -> Optional[Path]:
        """Save results to CSV file"""
        if not results:
            self.logger.warning("No results to save")
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"israeli_genai_researchers_{timestamp}.csv"

        filepath = self.config.output_dir / filename

        self.logger.info(f"Saving results to CSV: {filename}")
        df = pd.DataFrame(results)

        # Remove duplicates
        original_count = len(df)
        df = df.drop_duplicates(subset=['author_name', 'paper_title'])

        if len(df) < original_count:
            removed_count = original_count - len(df)
            self.logger.info(f"Removed {removed_count} duplicates")

        # Sort by confidence and similarity
        df = df.sort_values(['confidence', 'similarity_score', 'published_date'],
                            ascending=[False, False, False])

        df.to_csv(filepath, index=False, encoding='utf-8')
        self.logger.info(f"Successfully saved {len(df)} results to: {filepath}")

        # Summary
        if len(df) > 0:
            self.logger.info("RESULTS SUMMARY:")
            self.logger.info(f"   Average confidence: {df['confidence'].mean():.3f}")
            self.logger.info(f"   Average similarity: {df['similarity_score'].mean():.3f}")
            self.logger.info(f"   Unique authors: {df['author_name'].nunique()}")
            self.logger.info(f"   Unique papers: {df['paper_title'].nunique()}")
            self.logger.info(f"   Top universities: {df['university'].value_counts().head(3).to_dict()}")

        return filepath

    def run_scraper(self, start_year: int = 2020, end_year: int = None) -> Optional[Path]:
        """Run complete scraping process with immediate processing after each month"""
        start_time = datetime.now()

        self.logger.info("=" * 80)
        self.logger.info("ARXIV ISRAELI GENAI RESEARCHERS SCRAPER v3.0-MONTHLY")
        self.logger.info("IMMEDIATE PROCESSING AFTER EACH MONTH COLLECTION")
        self.logger.info("=" * 80)
        self.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Period: {start_year} - {end_year or datetime.now().year}")
        self.logger.info(f"Relevance threshold: {self.config.relevance_threshold}")

        # Test DeepSeek connection
        if not self.deepseek_analyzer.test_connection():
            self.logger.critical("CRITICAL ERROR: DeepSeek model unavailable!")
            return None

        # Collect and process papers by months (immediate processing)
        results = self.collect_and_process_by_date_range(start_year, end_year)

        # Save final consolidated results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = None

        if results:
            final_filename = f"israeli_genai_researchers_final_{timestamp}.csv"
            csv_path = self.save_to_csv(results, final_filename)

            # Also save a consolidated live results file
            self.save_incremental_results(results)
        else:
            self.logger.warning("No Israeli GenAI researchers found")

        # Final statistics
        stats = self.db.get_stats()
        elapsed = datetime.now() - start_time

        self.logger.info("=" * 80)
        self.logger.info("FINAL STATISTICS v3.0-MONTHLY")
        self.logger.info("=" * 80)
        self.logger.info(
            f"Total time: {elapsed.total_seconds():.1f} seconds ({elapsed.total_seconds() / 60:.1f} minutes)")
        self.logger.info(f"Papers processed: {stats['total_processed']:,}")
        self.logger.info(f"GenAI relevant: {stats['genai_relevant']:,}")
        self.logger.info(f"Israeli papers: {stats['israeli_papers']:,}")
        self.logger.info(f"Israeli researchers: {len(results) if results else 0}")

        if csv_path:
            self.logger.info(f"Results saved: {csv_path.name}")

        self.logger.info("=" * 80)

        return csv_path


def main():
    """Main execution function"""
    # Setup main logger
    logger = logging.getLogger(__name__)

    logger.info("ArXiv Israeli GenAI Researchers Scraper v3.0-MONTHLY")
    logger.info("   Immediate processing after each month collection")
    logger.info("   Real-time results as data is collected")
    logger.info("   Efficient pipeline with continuous processing")

    config = Config()
    scraper = ArxivScraper(config)

    # Run scraper (you can change the years)
    csv_path = scraper.run_scraper(start_year=2020, end_year=2025)

    if csv_path:
        logger.info("SUCCESS! Scraping completed successfully!")
        logger.info(f"Results saved to: {csv_path}")
    else:
        logger.info("COMPLETED: Check logs for details")


if __name__ == "__main__":
    main()