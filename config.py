from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration settings for the scraper"""

    # API Settings
    arxiv_base_url = "http://export.arxiv.org/api/query"
    max_results_per_query = 1000
    delay_between_requests = 3  # seconds to respect arXiv rate limits

    # AI Models
    vector_model_name = "all-MiniLM-L6-v2"  # Sentence transformer for relevance
    deepseek_model = "deepseek-r1"  # DeepSeek-R1 7B model

    # Processing Settings - FIXED: Lower threshold for better detection
    relevance_threshold = 0.3  # Lowered from 0.7 to be less restrictive
    batch_size = 10

    # Output Settings
    output_dir = Path("output")
    db_path = "processed_papers.db"

    gen_ai_keywords = [
        # Core GenAI terms
        "generative", "generation", "generate", "generating", "generator",
        "gan", "gans", "adversarial", "diffusion", "vae", "autoencoder",
        "variational", "denoising", "latent", "synthesis", "synthesize",

        # Large Language Models
        "llm", "large language model", "transformer", "attention", "bert",
        "gpt", "t5", "chatgpt", "language model", "text generation",

        # Computer Vision Generation
        "image generation", "image synthesis", "dalle", "midjourney",
        "stable diffusion", "clip", "vision transformer", "vit", "stylegan",

        # Core ML/AI terms
        "neural network", "deep learning", "machine learning", "artificial intelligence",
        "computer vision", "natural language", "nlp", "cv", "ml", "ai",
        "embedding", "neural", "convnet", "cnn", "rnn", "lstm", "seq2seq",
        "encoder", "decoder", "attention mechanism", "self-attention",

        # Creative AI
        "text-to-image", "image-to-text", "multimodal", "creative ai",
        "content generation", "automatic generation", "neural synthesis",

        # Recent GenAI terms
        "foundation model", "pre-trained", "fine-tuning", "prompt", "prompting",
        "zero-shot", "few-shot", "in-context", "emergent", "scaling"
    ]
