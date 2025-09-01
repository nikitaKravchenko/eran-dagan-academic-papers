import logging
from typing import Tuple

import faiss
from sentence_transformers import SentenceTransformer


class VectorIndex:
    """Improved vector similarity search for GenAI relevance detection"""

    def __init__(self, model_name: str):
        logging.info(f"Loading vector similarity model: {model_name}")
        logging.info("   This may take a moment for first-time setup...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        logging.info(f"Vector model loaded successfully!")
        logging.info(f"Vector model loaded: {model_name}")

    def build_relevance_keywords(self):
        """Create embeddings for GenAI relevance detection """
        logging.info("Building improved GenAI relevance detection index...")

        # MUCH MORE COMPREHENSIVE set of GenAI descriptions
        gen_ai_descriptions = [
            # Core generative AI
            "generative artificial intelligence and machine learning models",
            "generative adversarial networks and neural synthesis",
            "diffusion models for image and content generation",
            "variational autoencoders and generative modeling",
            "neural content generation and synthesis techniques",

            # Language models
            "large language models and natural language generation",
            "transformer architectures for text generation",
            "language model pre-training and fine-tuning",
            "conversational AI and dialogue systems",
            "text generation and natural language processing",

            # Computer vision generation
            "image synthesis and computer vision generation",
            "text-to-image generation and visual synthesis",
            "neural image generation and style transfer",
            "visual content creation and generation",
            "multimodal generation and vision-language models",

            # Creative and content AI
            "creative artificial intelligence and content generation",
            "automatic content creation and neural synthesis",
            "AI-generated media and synthetic content",
            "neural creativity and generative design",
            "automated content production using AI",

            # Technical approaches
            "deep generative models and neural networks",
            "attention mechanisms for content generation",
            "neural architecture for generative tasks",
            "representation learning for generation",
            "unsupervised learning for content synthesis",

            # Applications
            "synthetic data generation and augmentation",
            "procedural content generation using AI",
            "personalized content generation systems",
            "interactive generation and creative tools",
            "AI-assisted content creation workflows"
        ]

        embeddings = self.model.encode(gen_ai_descriptions)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        self.texts = gen_ai_descriptions
        logging.info(f"Vector similarity index built with {len(gen_ai_descriptions)} reference descriptions!")
        logging.info("Improved vector index built for GenAI relevance detection")

    def is_relevant_to_genai(self, text: str, threshold: float = 0.3) -> Tuple[bool, float]:
        """Check if text is relevant to Generative AI"""
        if not self.index:
            self.build_relevance_keywords()

        # Clean and prepare text
        text_clean = text.lower().strip()
        if len(text_clean) < 10:  # Too short to analyze
            return False, 0.0

        query_embedding = self.model.encode([text_clean])
        distances, indices = self.index.search(query_embedding.astype('float32'), 3)  # Check top 3 matches

        # Convert L2 distances to similarities and take the best match
        similarities = [1 / (1 + dist) for dist in distances[0]]
        best_similarity = max(similarities)

        is_relevant = best_similarity > threshold

        logging.debug(
            f"GenAI relevance - Score: {best_similarity:.3f}, Threshold: {threshold}, Relevant: {is_relevant}")
        logging.debug(f"Text analyzed: {text[:100]}...")

        return is_relevant, best_similarity

    def debug_similarity_scores(self, text: str, top_k: int = 5):
        """Debug function to see similarity scores - helpful for tuning"""
        if not self.index:
            self.build_relevance_keywords()

        query_embedding = self.model.encode([text.lower().strip()])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        logging.info(f"Debug similarity analysis for: {text[:50]}...")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            similarity = 1 / (1 + dist)
            logging.info(f"   {i + 1}. Score: {similarity:.3f} | {self.texts[idx][:60]}...")
