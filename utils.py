from random import randint
import re
from typing import Any, Dict, List, Optional
import torch

from Context import Context
from prompts import (
    encoder_prompt_format,
    continue_prompt_format,
)


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


def sample_first_words(text, context: Context):
    if len(text) < context.imdb_ds_min_chars_num:
        return text
    idx = text.find(".", context.imdb_ds_min_chars_num)
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
    context: Context,
) -> Dict[str, Any]:
    """Maps a single element from dataset to a dictionary with the following keys."""

    bits: List[str] = context.bits
    tokenizer = context.tokenizer
    hide_template: str = encoder_prompt_format
    no_hide_template: str = continue_prompt_format

    assert bits and len(bits) >= 2
    text = sample_first_words(element["text"], context)
    to_hide = randint(0, 1) == 1
    if to_hide:
        bit = bits[randint(0, len(bits) - 1)]
        query = hide_template.format(bit=bit, bits=format_bits(bits), text=text)
    else:
        bit = ""
        query = no_hide_template.format(text=text)
    res = {
        "bit": bit,
        "query": query,
        "text": text,
    }
    if tokenizer is not None:
        chat = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": query},
        ]
        formatted_chat = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer(
            formatted_chat,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            max_length=context.ds_sample_max_length,
        )
        # To tensors:
        res["input_ids"] = tokens["input_ids"].squeeze()
        res["attention_mask"] = tokens["attention_mask"].squeeze()
    return res


def is_decoded(dhm, hm):
    """
    Check if the hidden message is decoded correctly by 
    comparing the decoded hidden message with the original hidden message.

    :param dhm: the decoded hidden message
    :param hm: original hidden message
    """
    hm = hm.strip().lower() if hm else ""
    dhm = dhm.strip().lower() if dhm else ""
    if not hm and not dhm:
        return True
    return bool(hm) and bool(dhm) and hm in dhm
