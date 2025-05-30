import torch
import transformers
from transformers import LogitsProcessor
import emoji

all_emojis = list(emoji.EMOJI_DATA.keys())


class NoNumberLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, max_number=10000):
        """
        Создает список токенов для уменьшения логитов
        :param tokenizer:
        :param max_number:
        """
        self.tokenizer = tokenizer
        self.blocked_token_ids = set()

        # Убираем числа
        for number in range(max_number):
            text = f" {number}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            self.blocked_token_ids.update(token_ids)
        # Убираем смайлики
        for emj in all_emojis:
            token_ids = tokenizer.encode(f" {emj}", add_special_tokens=False)
            self.blocked_token_ids.update(token_ids)
        # Убираем другие трэш токены
        for i in ['ъъъ', 'ъуъ', 'нг', 'ъебвщ', 'ъ', 'ЪУЪ', 'Ъ', 'ням', 'нк', 'нд']:
            token_ids = tokenizer.encode(f" {i}", add_special_tokens=False)
            self.blocked_token_ids.update(token_ids)
        # Создаем список
        self.blocked_token_ids = list(self.blocked_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Возвращает scores с заниженными логитами для токенов чисел во всех батчах
        """
        scores = scores.clone()
        scores[:, self.blocked_token_ids] -= 10

        return scores
