import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AGSERConfig:
    k_ratio: float = 2/3
    lambda_balance: float = 1.0
    attention_type: str = 'mean'
    max_length: int = 100
    temperature: float = 1.0
    top_p: float = 0.9

class AGSER:
    def __init__(self, model_name: str, config: Optional[AGSERConfig] = None):
        self.config = config or AGSERConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.rouge_scorer = MLXRougeScorer()

        logger.info(f"Initialized AGSER with model {model_name}")

    def get_token_contributions(self, query: str) -> List[float]:
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states)

        token_contributions = []
        for token_idx in range(inputs['input_ids'].shape[1]):
            if self.config.attention_type == 'mean':
                contrib = np.mean([
                    hidden_states[layer_idx][0, token_idx].detach().numpy().mean()
                    for layer_idx in range(num_layers)
                ])
            elif self.config.attention_type == 'mid':
                mid = num_layers // 2
                contrib = hidden_states[mid][0, token_idx].detach().numpy().mean()
            elif self.config.attention_type == 'first':
                contrib = hidden_states[0][0, token_idx].detach().numpy().mean()
            elif self.config.attention_type == 'last':
                contrib = hidden_states[-1][0, token_idx].detach().numpy().mean()
            else:
                contrib = max([
                    hidden_states[layer_idx][0, token_idx].detach().numpy().mean()
                    for layer_idx in range(num_layers)
                ])
            token_contributions.append(contrib)

        return token_contributions

    def split_queries(self, query: str, scores: List[float]) -> Tuple[str, str]:
        inputs = self.tokenizer(query, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        if len(tokens) != len(scores):
            raise ValueError("Token-score length mismatch")

        k = int(len(tokens) * self.config.k_ratio)
        att_indices = np.argpartition(scores, -k)[-k:]
        non_att_indices = np.argpartition(scores, -k)[:-k]

        att_tokens = [tokens[i] for i in sorted(att_indices) if tokens[i] not in self.tokenizer.all_special_tokens]
        non_att_tokens = [tokens[i] for i in sorted(non_att_indices) if tokens[i] not in self.tokenizer.all_special_tokens]

        att_query = self.tokenizer.convert_tokens_to_string(att_tokens)
        non_att_query = self.tokenizer.convert_tokens_to_string(non_att_tokens)

        return att_query, non_att_query

    def generate_answer(self, query: str) -> str:
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def detect_hallucination(self, query: str) -> Dict[str, Union[float, str]]:
        logger.info(f"Processing query: {query}")
        original_answer = self.generate_answer(query)

        scores = self.get_token_contributions(query)
        att_query, non_att_query = self.split_queries(query, scores)

        att_answer = self.generate_answer(att_query)
        non_att_answer = self.generate_answer(non_att_query)

        r_att = self.rouge_scorer(att_answer, original_answer)
        r_non_att = self.rouge_scorer(non_att_answer, original_answer)

        hallucination_score = self.config.lambda_balance * r_att - r_non_att

        return {
            'hallucination_score': hallucination_score,
            'r_att': r_att,
            'r_non_att': r_non_att,
            'original_answer': original_answer,
            'attentive_answer': att_answer,
            'non_attentive_answer': non_att_answer,
            'attentive_query': att_query,
            'non_attentive_query': non_att_query
        }

class MLXRougeScorer:
    def __call__(self, candidate: str, reference: str) -> float:
        def lcs(s1: List[str], s2: List[str]) -> int:
            m, n = len(s1), len(s2)
            dp = np.zeros((m + 1, n + 1), dtype=int)
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i, j] = dp[i - 1, j - 1] + 1
                    else:
                        dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
            return dp[m, n]

        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        lcs_val = lcs(cand_tokens, ref_tokens)

        if not ref_tokens or not cand_tokens:
            return 0.0

        precision = lcs_val / len(cand_tokens)
        recall = lcs_val / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def run_example():
    config = AGSERConfig()
    agser = AGSER("gpt2", config)

    queries = [
        "Who is the author of the book The Testament, what year was it published?",
        "Who is the author of the book Building Blocks (Point), what year was it published?",
        "Who is the author of the book Pigs in Heaven, what year was it published?",
        "Who is the author of the book Each Time We Love, what year was it published?"
    ]

    for query in queries:
        result = agser.detect_hallucination(query)
        print("\nQuery:", query)
        print("Hallucination Score:", result['hallucination_score'])
        print("Original Answer:", result['original_answer'])
        print("Attentive Query:", result['attentive_query'])
        print("Attentive Answer:", result['attentive_answer'])
        print("Non-Attentive Query:", result['non_attentive_query'])
        print("Non-Attentive Answer:", result['non_attentive_answer'])


if __name__ == "__main__":
    run_example()
