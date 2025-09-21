# bert.py
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from venv import logger
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import os

load_dotenv()

class BertAnalyzer:
    def __init__(self):
        self.model_name = os.getenv('BERT_MODEL_NAME')
        self.model_path = os.getenv('MODEL_PATH')
        self.model_version = os.getenv('MODEL_VERSION', '1.0.0')
        self.timeout_seconds = int(os.getenv('MODEL_TIMEOUT', 30))
        self.initialize_model()

    def initialize_model(self):
        """Initialize BERT model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        except:
            print("Local model not found. Downloading from Hugging Face...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            os.makedirs(self.model_path, exist_ok=True)
            self.tokenizer.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)

        self.analyzer = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def analyze(self, text):
        """Analyze text and return sentiment with timeout"""
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.analyzer, text)
                result = future.result(timeout=self.timeout_seconds)
                return result
        except TimeoutError:
            logger.error("Analysis timed out.")
            return {"label": "TIMEOUT", "score": 0.0}
        except Exception as e:
            logger.error(f"BERT analysis error: {str(e)}")
            return {"label": "ERROR", "score": 0.0}
