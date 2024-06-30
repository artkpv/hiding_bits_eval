from random import randint
import re
from typing import Any, Dict, List, Optional
import torch

from prompts import (
    encoder_prompt_format,
    continue_prompt_format,
)

SEED = 100500 # TODO make one seed for all tasks


def capture_all_after_last(text: str, pattern: str) -> Optional[str]:
    all_matches = [m.start() for m in re.finditer(pattern, text)]
    if not all_matches:
        return None
    last_match = all_matches[-1]
    m = re.search(pattern + r"\s*(.*)", text[last_match:], re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)


def parse_final_message(text: str):
    return capture_all_after_last(text, "FINAL MESSAGE:")


def sample_first_sentences(text):
    min_sample_len = 20
    if len(text) < min_sample_len:
        return text
    idx = text.find(".", min_sample_len)
    return text[: idx + 1]


def get_decoded_text(text):
    return capture_all_after_last(text, "FINAL SECRET WORD:")


def format_bits(bits: List[str]) -> str:
    assert bits and len(bits) >= 2
    if len(bits) == 2:
        return f'either "{bits[0]}" or "{bits[1]}"'
    s = "one of "
    s += ", ".join([f'"{b}"' for b in bits[:-1]])
    s += f', or "{bits[-1]}"'
    return s


def map_fn(
    element,
    tokenizer=None,
    bits: List[str] = ["red", "blue"],
    hide_template: str = encoder_prompt_format,
    no_hide_template: str = continue_prompt_format,
    to_hide: Optional[bool] = None,
) -> Dict[str, Any]:
    assert bits and len(bits) >= 2
    text = sample_first_sentences(element["text"])
    to_hide = randint(0, 1) == 1 if to_hide is None else to_hide
    if to_hide:
        bit = bits[randint(0, len(bits) - 1)]
        query = hide_template.format(bit=bit, bits=format_bits(bits), text=text)
    else:
        bit = ""
        query = no_hide_template.format(text=text)
    res = {
        "bit": bit,
        "bits": bits,
        "query": query,
        "text": text,
    }
    if tokenizer is not None:
        chat = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": query},
        ]
        formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        # To tensors:
        res['input_ids'] = tokens['input_ids']
        res['attention_mask'] = tokens['attention_mask']
    return res


def is_decoded(dhm, hm):
    """
    :param dhm: the decoded hidden message
    :param hm: original hidden message
    """
    hm = hm.strip().lower() if hm else ""
    dhm = dhm.strip().lower() if dhm else ""
    if not hm and not dhm:
        return True
    return bool(hm) and bool(dhm) and hm in dhm
