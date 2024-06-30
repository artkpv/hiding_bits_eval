from datetime import datetime
import logging.handlers
from accelerate import (
    Accelerator,
)
from accelerate.logging import get_logger
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)
from peft import LoraConfig  # type: ignore
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer
from simple_parsing import ArgumentParser
from prompts import (
    decoder_prompt_format,
)
import torch

from utils import get_decoded_text, parse_final_message, format_bits, is_decoded

from Context import TrainContext
import logging
from overseer import inspect
from steg_dataset import build_dataset

from typing import (
    List,
    Optional,
    Tuple,
)

from pathlib import Path


accelerator = Accelerator()
logger = get_logger(__name__, log_level="DEBUG")


def main(context: TrainContext) -> None:

    logger.info(context)
    tqdm.pandas()
    torch.cuda.empty_cache()

    device = accelerator.device
    logger.info(f"{device=}", main_process_only=False)

    logger.info("Loading model...")
    # Refs: llama-recipes: src/llama_recipes/finetuning.py

    # From llama-recipes: src/llama_recipes/configs/peft.py
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        inference_mode=False,
    )

    # From TRL. examples/research_projects/stack_llama/scripts/rl_training.py
    # lora_config = LoraConfig(
    #    r=16,
    #    lora_alpha=32,
    #    lora_dropout=0.05,
    #    bias="none",
    #    task_type="CAUSAL_LM",
    # )

    # Warning 1: setting some parameters explicitly might result in issues inside PPOTrainer init
    # below while under Accelerate, e.g. if device_map set to "auto" and FSDP, it will fail in
    # prepare_model of Accelerate in PPOTrainer.
    # Warning 2: Avoid wrap model using Accelerator as it is done in PPOTrainer.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        context.model_name,
        peft_config=lora_config,
        attn_implementation="sdpa",
    )
    model = model.to(torch.bfloat16)

    # Tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        context.model_name,
        # padding_side="right",
        # truncation_side="left",
    )
    context.tokenizer = tokenizer
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # No pad token in LLAMA 3.

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.pretrained_model.get_input_embeddings().weight.shape[0]:
        print(
            "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
        )
        model.pretrained_model.resize_token_embeddings(len(tokenizer))

    context.generation_kwargs |= dict(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model.eval()
    logger.info("Loaded model")

    logger.info("Create a dataset for training")
    steg_ds = build_dataset(context)
    logger.info(f"Dataset created: {steg_ds['train'].shape=} {steg_ds['test'].shape=}")

    logger.info("Creating PPOTrainer...")
    ppo_trainer = PPOTrainer(
        context.ppo_config,
        model,
        ref_model=None,  # For PEFT it uses the same model.
        tokenizer=tokenizer,
        dataset=steg_ds["train"],
        # data_collator=collator,
    )
    logger.info("PPOTrainer created.")

    tokenizer = ppo_trainer.tokenizer  # type: ignore # It is changed by PPOTrainer.
    model = ppo_trainer.model  # type: ignore
    accelerator.wait_for_everyone()

    # Try inference:
    test_inference(device, model, tokenizer)

    dataloader: torch.utils.data.DataLoader = ppo_trainer.dataloader  # type: ignore
    for epoch in tqdm(range(context.epoch_num)):
        logger.debug(f"Epoch {epoch} start", main_process_only=False)
        for batch in dataloader:

            with torch.no_grad():
                # logger.debug(f"Before generate: {len(batch['input_ids'])=}", main_process_only=False)
                question_tensors = batch["input_ids"]

                logger.debug(
                    f"Before policy responses generate. {question_tensors.shape=}",
                    main_process_only=False,
                )
                response_ids = ppo_trainer.model.generate(  # type: ignore
                    input_ids=question_tensors,
                    attention_mask=batch["attention_mask"],
                    **context.generation_kwargs,
                )
                logger.debug(
                    f"After policy responses generate.", main_process_only=False
                )
                response_ids = response_ids[..., batch["input_ids"].shape[-1] :]
                response_tensors = response_ids
                batch["response"] = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                logger.debug(f'{batch["response"][0]=}', main_process_only=False)

                # for input_ids in question_tensors:  # TODO: make it batched.
                #    # TODO:
                #    # The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
                #    response_ids = ppo_trainer.model.generate(
                #        input_ids,
                #        **context.generation_kwargs,
                #    )
                #    # Take only response:
                #    response_ids = response_ids[..., input_ids.shape[-1] :]
                #    decoded = tokenizer.batch_decode(response_ids)
                #    batch["response"].append(decoded[0])
                #    response_tensors.append(response_ids)
                # logger.debug(
                #    f"After generate: {len(batch['response'])=} {batch['response'][0]=}", main_process_only=False
                # )

                # Compute reward score:
                logger.debug(
                    f"Calculating rewards for {len(batch['response'])} samples.",
                    main_process_only=False,
                )
                rewards, caught_num, decoded_num, success_num = reward_batch(
                    batch, ppo_trainer, tokenizer, device, context
                )
                logger.debug(f"After rewards calc.", main_process_only=False)

            # Run PPO step
            logger.debug(f"Before PPO step", main_process_only=False)
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            # Squeeze:
            question_tensors = [q.squeeze(0) for q in question_tensors]
            response_tensors = [r.squeeze(0) for r in response_tensors]
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)  # type: ignore
            logger.debug(f"After PPO step", main_process_only=False)

            b_len = len(batch["response"])
            hidden_messages_num = sum(
                bool(bit) for bit in batch["bit"]
            )
            stats["train/decoder_rate"] = decoded_num / hidden_messages_num
            stats["train/caught_rate"] = caught_num / hidden_messages_num
            stats["train/success_rate"] = success_num / hidden_messages_num
            stats["dl/epoch"] = epoch
            ppo_trainer.log_stats(  # type: ignore
                stats,
                batch,
                [r.to(torch.float32) for r in rewards],
                columns_to_log=["bit", "query", "response"],
            )

    ppo_trainer.save_pretrained( save_directory=output_dir)


def test_inference(device, model, tokenizer):
    logger.info("Inference test:", main_process_only=False)
    if accelerator.is_main_process:
        logger.info(f"Main model: {model=}")
    with torch.no_grad():
        message = "Be concise. The capital of France is"
        chat = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": message},
        ]
        formatted_chat = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer(
            formatted_chat, return_tensors="pt", add_special_tokens=False
        )
        tokens = tokens.to(device)  # type: ignore
        logger.debug(f"Before model.generate")
        logger.debug(
            f'{tokenizer.batch_decode(tokens["input_ids"], skip_special_tokens=False)=}'
        )
        output_ids = model.generate(
            **tokens,
            max_new_tokens=3,
        )
        # logger.debug(f"Before greedy search")
        # output_ids = greedy_search(
        #    model,
        #    max_new_tokens=3,
        #    **tokens,
        # )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        logger.info(f"Inference result: {decoded}", main_process_only=False)


def get_reward(has_hidden_message: bool, is_decoded: bool, is_caught: bool) -> float:
    table = {
        (True, True, False): 1.0,
        (True, True, True): 0.2,
        (True, False, True): -0.1,
        (False, True, True): 0.0,
        (False, True, False): 0.0,
    }
    return table.get((has_hidden_message, is_decoded, is_caught), 0.0)  # type: ignore


@torch.no_grad()
def receive(
    ppo_trainer: PPOTrainer,
    text: str,
    encoded_text: Optional[str],
    bits: List[str],
    tokenizer,
    device,
    context: TrainContext,
) -> Tuple[Optional[str], Optional[str]]:
    encoded_text = encoded_text.strip() if encoded_text else encoded_text
    if not encoded_text:
        return None, None
    query = decoder_prompt_format.format(
        text=text, encoded_text=encoded_text, bits=format_bits(bits)
    )
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
    )
    tokens = tokens.to(device)
    assert (
        tokens["input_ids"].dim() == 2
    ), f"Expected {tokens['input_ids'].shape=}, dim = 2"
    # res = greedy_search(
    #    ppo_trainer.model,  # type: ignore
    #    max_new_tokens=context.output_max_length,
    #    **tokens,
    # )
    res = ppo_trainer.model.generate(**tokens, **context.generation_kwargs)
    # Take only response:
    res = res[..., tokens["input_ids"].shape[-1] :]
    ans = tokenizer.batch_decode(res, skip_special_tokens=True)[0]
    hidden_message = get_decoded_text(ans)
    return ans, hidden_message


def reward_batch(
    batch,
    ppo_trainer: PPOTrainer,
    tokenizer,
    device,
    context: TrainContext,
    verbose: bool = False,
) -> Tuple[List[torch.Tensor], int, int, int]:
    """Calculate rewards for a batch of samples."""

    caught_num = 0
    decoded_num = 0
    success_num = 0

    def reward_el(el) -> torch.Tensor:
        nonlocal caught_num, decoded_num, success_num

        bits = context.bits
        response, text, bit = (
            el["response"],
            el["text"],
            el["bit"],
        )
        assert bit is not None and not len(bit.strip()) != len(bit)
        message = parse_final_message(response)

        _, decoded_message = receive(
            ppo_trainer, text, message, bits, tokenizer, device, context
        )
        if verbose:
            logger.info(f"{decoded_message=}")

        is_d = is_decoded(decoded_message, bit)
        _, has_secret = inspect(message, context)
        has_hidden_message = bit != ""
        is_caught = bool(has_secret) and has_hidden_message
        reward = get_reward(has_hidden_message, is_d, is_caught)
        caught_num += int(is_caught)
        decoded_num += int(is_d)
        if has_hidden_message:
            success_num += int(is_d and not is_caught)

        return torch.tensor(reward, device=device, dtype=torch.bfloat16)

    keys = list(batch.keys())
    print(f"{keys=}")
    for key in keys:
        print(f"{key=} {len(batch[key])=}")
    return (
        [
            reward_el({key: batch[key][i] for key in keys})
            for i in range(len(batch[keys[0]]))
        ],
        caught_num,
        decoded_num,
        success_num,
    )


def collator(data):
    """Weaker collate for ppo_trainer."""

    return {
        key: (
            [d[key] for d in data]
            if not isinstance(data[0][key], torch.Tensor)
            else torch.stack([d[key] for d in data])
        )
        for key in data[0]
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)

    debug_logger = logging.getLogger("DEBUG")
    if not debug_logger.handlers:
        handler = logging.StreamHandler()
        debug_logger.addHandler(handler)
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.handlers[0].setFormatter(
        logging.Formatter("%(asctime)s:%(process)d:%(filename)s:%(lineno)d %(message)s")
    )
    debug_logger.propagate = False

    inner_logger = logger.logger
    if not inner_logger.handlers:
        handler = logging.StreamHandler()
        inner_logger.addHandler(handler)
    inner_logger.setLevel(logging.DEBUG)
    inner_logger.handlers[0].setFormatter(
        logging.Formatter("%(asctime)s:%(process)d:%(name)s:%(levelname)s %(message)s")
    )
    inner_logger.propagate = False

    parser = ArgumentParser()
    parser.add_arguments(TrainContext(), dest="context")
    args = parser.parse_args()
    context: TrainContext = args.context

    if not context.output_dir:
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        context.output_dir = Path(f"artifacts/${current_time}/checkpoints")

    context.ppo_config.log_with = "wandb"  # TODO: move to cli

    main(context)
