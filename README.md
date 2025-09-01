# ArXiv Israeli GenAI Researchers Scraper
___
## Overview

This scraper processes ArXiv papers month-by-month, applying multiple filtering layers to identify relevant generative AI research by Israeli authors and institutions. It uses advanced NLP techniques and AI analysis to ensure high-quality results while maintaining efficient processing.

## Features

### Core Capabilities
- **Systematic Monthly Collection**: Processes ArXiv papers chronologically by month for comprehensive coverage
- **Multi-layered Filtering**: Combines keyword matching, vector similarity, and AI analysis
- **Real-time Processing**: Analyzes papers immediately after collection for faster results
- **Intelligent Author Analysis**: Uses DeepSeek-R1 model to identify Israeli researchers and their affiliations
- **Vector Similarity Matching**: Employs sentence transformers to assess GenAI relevance
- **Incremental Results**: Saves progress continuously to prevent data loss
- **Comprehensive Logging**: Detailed logging with progress tracking and statistics

### Technical Features
- **Database Integration**: SQLite database for tracking processed papers and avoiding duplicates
- **Configurable Thresholds**: Adjustable relevance and confidence thresholds
- **Error Handling**: Robust error handling with automatic retry mechanisms
- **Rate Limiting**: Respects ArXiv API rate limits (3-second delays)
- **Batch Processing**: Efficient batch processing with configurable sizes

## Architecture

The project is structured into several specialized components:

### Core Components
- **`arxiv_scraper.py`**: Main scraper with monthly collection and processing logic
- **`deepseek_analyzer.py`**: AI-powered author and affiliation analysis using DeepSeek-R1
- **`vector_index.py`**: Vector similarity analysis for GenAI relevance detection
- **`db_manager.py`**: SQLite database management for processing state
- **`config.py`**: Configuration management and settings

### Processing Pipeline
1. **Collection**: Fetch papers from ArXiv API by month and category
2. **Keyword Filtering**: Initial filter using comprehensive GenAI keyword list
3. **Vector Analysis**: Semantic similarity analysis using sentence transformers
4. **AI Analysis**: DeepSeek-R1 powered analysis for Israeli author identification
5. **Storage**: Save results to database and CSV files

## Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai) for DeepSeek-R1 model

### Setup Steps

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/eran-dagan-academic-papers.git
cd eran-dagan-academic-papers
```

2. **Install dependencies**:
```bash
uv pip install -r requirements.txt
```

3. **Install and setup Ollama with DeepSeek-R1**:
```bash
# Install Ollama (follow instructions at https://ollama.ai)
ollama pull deepseek-r1
```

4. **Create output directory**:
```bash
mkdir output
```

## Configuration

### Key Settings in `config.py`

```python
# Processing Settings
relevance_threshold = 0.3  # Vector similarity threshold (0.0-1.0)
batch_size = 10           # Processing batch size

# AI Models
vector_model_name = "all-MiniLM-L6-v2"  # Sentence transformer model
deepseek_model = "deepseek-r1"          # DeepSeek model for analysis

# API Settings
max_results_per_query = 2000            # ArXiv API batch size
delay_between_requests = 3              # Seconds between API calls
```

### GenAI Keywords
The configuration includes an extensive list of GenAI-related keywords covering:
- Core generative AI terms (GAN, diffusion, VAE, etc.)
- Large language models (LLM, transformer, GPT, etc.)
- Computer vision generation (image synthesis, text-to-image, etc.)
- Creative AI and content generation
- Recent GenAI developments (foundation models, prompt engineering, etc.)

## Usage

### Basic Usage
```python
python arxiv_scraper.py tail -f arxiv_scraper.log
```

### Custom Date Range
```python
from arxiv_scraper import ArxivScraper
from config import Config

config = Config()
scraper = ArxivScraper(config)

# Scrape specific years
csv_path = scraper.run_scraper(start_year=2020, end_year=2025)
```

### Configuration Customization
```python
# Adjust relevance threshold
config.relevance_threshold = 0.4  # More restrictive
config.relevance_threshold = 0.2  # More inclusive

# Change output directory
config.output_dir = Path("custom_output")
```

## Output

### Generated Files
- **`israeli_genai_researchers_final_[timestamp].csv`**: Final consolidated results
- **`results_month_[year]_[month]_[timestamp].csv`**: Monthly incremental results
- **`papers_checkpoint_[checkpoint].csv`**: Periodic paper collection checkpoints
- **`processed_papers.db`**: SQLite database with processing history
- **`arxiv_scraper.log`**: Comprehensive execution logs

### CSV Structure
Each result CSV contains the following columns:
- `arxiv_id`: ArXiv paper identifier
- `paper_title`: Full paper title
- `author_name`: Identified Israeli author name
- `university`: Author's affiliated institution
- `confidence`: AI analysis confidence score (0.0-1.0)
- `reasoning`: AI reasoning for Israeli identification
- `published_date`: Paper publication date
- `categories`: ArXiv categories
- `all_authors`: Complete author list
- `similarity_score`: Vector similarity score for GenAI relevance

## Logging and Monitoring

The system provides comprehensive logging with multiple levels:

### Log Categories
- **INFO**: Major milestones and progress updates
- **DEBUG**: Detailed processing information
- **WARNING**: Non-critical issues and fallbacks
- **ERROR**: Critical errors requiring attention

### Key Metrics Tracked
- Papers processed per month
- Filter success rates
- AI analysis accuracy
- Processing time statistics
- Error rates and recovery
