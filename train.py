import logging.handlers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from accelerate.logging import get_logger
from tqdm import tqdm
from datetime import timedelta
from transformers import (
    GenerationConfig,
    AutoTokenizer,
)
from peft import LoraConfig  # type: ignore
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer
from trl.core import LengthSampler
from simple_parsing import ArgumentParser
from prompts import (
    decoder_prompt_format,
)
import os
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from utils import get_decoded_text, parse_final_message, format_bits, is_decoded

from Context import TrainContext
import logging
from overseer import inspect
from steg_dataset import build_dataset

from typing import (
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    cast,
)


accelerator = Accelerator(
    # kwargs_handlers=[
    #    InitProcessGroupKwargs(timeout=timedelta(minutes=30), backend="nccl")
    # ]
)
logger = get_logger(__name__, log_level="DEBUG")


def main(context: TrainContext) -> None:

    # os.environ["ACCELERATE_DISABLE_RICH"] = "1"
    tqdm.pandas()
    logger.info(context)
    torch.cuda.empty_cache()

    device = accelerator.device
    logger.info(f"{device=}", main_process_only=False)

    # Load model

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
        # use_cache=False,
        # device_map=None,
        attn_implementation="sdpa",
    )
    # model = model.to(torch.bfloat16)  # FSDP, as per Llama-Recipes.

    # Tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(context.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Refs: llama-recipes: src/llama_recipes/finetuning.py
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

    logger.info("Loaded model.")
    model.eval()
    with accelerator.main_process_first():
        logger.info("Create a dataset for training")
        steg_ds = build_dataset(context, tokenizer=tokenizer)
    accelerator.wait_for_everyone()

    ppo_trainer = PPOTrainer(
        context.ppo_config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=steg_ds["train"],
        data_collator=collator,
    )
    logger.info("PPOTrainer created.")
    tokenizer = ppo_trainer.tokenizer  # type: ignore # It is changed by PPOTrainer.
    model = ppo_trainer.model  # type: ignore
    accelerator.wait_for_everyone()

    # Try inference:
    logger.info("Inference test:", main_process_only=False)
    if accelerator.is_main_process:
        logger.info(f"Main model: {model=}")
    with torch.no_grad():
        message = "Be concise. The capital of France is"
        chat = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": message},
        ]
        formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        tokens = tokens.to(device)   # type: ignore
        logger.debug(f"Before model.generate")
        logger.debug(
            f'Decoded: {tokenizer.batch_decode(tokens["input_ids"], skip_special_tokens=False)=}'
        )
        output_ids = model.generate(
            tokens["input_ids"],
            max_new_tokens=3,
        )
        # logger.debug(f"Before greedy search")
        # output_ids = greedy_search(
        #    model,
        #    max_new_tokens=3,
        #    **tokens,
        # )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        logger.info(f"Inference using forward: {decoded}", main_process_only=False)

    # if accelerator.is_main_process:
    # accelerator.wait_for_everyone()
    # logger.debug("before model.generate")
    # res = ppo_trainer.model.generate(
    #    tensors,
    #    generation_config=GenerationConfig(
    #        max_new_tokens=3,
    #        do_sample=False,
    #        num_beams=1
    #    ),
    # ).cpu()
    # decoded_res = tokenizer.batch_decode(res)
    # logger.info(f"Inference using generate: {decoded_res}")

    # Note: We need to use the FSDP.summon_full_params context manager here because the generate function
    # does not seem to gather the weights for the LM head. This solution works because the tied weights of the LM head
    # are in the root FSDP module, and are summoned by the below context manager. See https://github.com/pytorch/pytorch/issues/100069
    # for more info.
    # Note: We use recurse=False here so that we only summon full params for the LM head, not the entire model.
    # print('Before FSDP summon')
    # with FSDP.summon_full_params(
    #    ppo_trainer.model, writeback=False, recurse=False
    # ):
    #    logger.debug("Before generate")
    #    res = ppo_trainer.model.generate(
    #        tensors,
    #        max_new_tokens=3,
    #        # max
    #        # generation_config=GenerationConfig(max_new_tokens=3, do_sample=False),
    #    ).cpu()

    # res = tokenizer.batch_decode(res)[0]
    # logger.info(res)

    # Training
    # dataloader = DataLoader(
    #    steg_ds["train"],
    #    batch_size=ppo_trainer.config.batch_size,
    #    collate_fn=collator,
    #    shuffle=True,
    #    num_workers=4,
    #    pin_memory=True,
    # )
    # dataloader = ppo_trainer.accelerator.prepare(dataloader)

    accelerator.wait_for_everyone()
    dataloader: torch.utils.data.DataLoader = ppo_trainer.dataloader  # type: ignore
    for epoch in tqdm(range(context.epoch_num)):
        logger.debug(f"Epoch {epoch} start", main_process_only=False)
        for batch in dataloader:
            # TODO: batch here has key: list (not tensor)?

            # response_tensors = []
            # for input_ids, attention_mask in zip(
            #    batch["input_ids"], batch["attention_mask"]
            # ):
            #    assert isinstance(
            #        input_ids, torch.Tensor
            #    ), f"{type(input_ids)=} is not Tensor"
            #    assert (
            #        input_ids.dim() == 1
            #    ), f"{input_ids.shape=} is not one dimentional"
            #    assert isinstance(
            #        attention_mask, torch.Tensor
            #    ), f"{type(attention_mask)=} is not Tensor"
            #    assert (
            #        attention_mask.dim() == 1
            #    ), f"{attention_mask.shape=} is not one dimentional"
            #    input_ids = input_ids.to(device)
            #    attention_mask = attention_mask.to(device)
            #    output_ids = greedy_search(
            #        model,
            #        max_new_tokens=context.output_max_length,
            #        input_ids=input_ids,
            #        attention_mask=attention_mask,
            #        return_prompt=False,
            #    )
            #    assert isinstance(
            #        output_ids, torch.Tensor
            #    ), f"{type(output_ids)=} is not Tensor"
            #    assert (
            #        output_ids.dim() == 1
            #    ), f"{output_ids.shape=} is not one dimentional"
            #    response_tensors.append(output_ids)
            #    batch["response"].append(
            #        tokenizer.decode(output_ids, skip_special_tokens=True)
            #    )

            with torch.no_grad():
                # logger.debug(f"Before generate: {len(batch['input_ids'])=}", main_process_only=False)
                question_tensors = batch["input_ids"]
                batch["response"] = []
                response_tensors = []
                for input_ids in question_tensors:  # TODO: make it batched.
                    # TODO:
                    # The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
                    response_ids = ppo_trainer.model.generate(
                        input_ids, 
                        **context.generation_kwargs,
                    )
                    # Take only response:
                    response_ids = response_ids[..., input_ids.shape[-1] :]
                    decoded = tokenizer.batch_decode(response_ids)
                    batch["response"].append(decoded[0])
                    response_tensors.append(response_ids)
                # logger.debug(
                #    f"After generate: {len(batch['response'])=} {batch['response'][0]=}", main_process_only=False
                # )

                # Compute reward score:
                # logger.debug(f"Calculating rewards for {len(batch['response'])} samples.", main_process_only=False)
                rewards, caught_num, decoded_num, success_num = reward_batch(
                    batch, ppo_trainer, tokenizer, device, context
                )
                # logger.debug(f"Rewards: {len(rewards)=}", main_process_only=False)

            # Run PPO step
            logger.debug(f"Before PPO step", main_process_only=False)
            # Log shapes of question_tensors and response_tensors
            # for inx, q, response, reward in zip(
            #    range(len(question_tensors)),
            #    question_tensors,
            #    response_tensors,
            #    rewards,
            # ):
            #    logger.debug(
            #        f"Tensors, {inx}: {q.shape=} {q.device=} {response.shape=} {response.device=}{reward=}", main_process_only=False
            #    )
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            # Squeeze:
            question_tensors = [q.squeeze(0) for q in question_tensors]
            response_tensors = [r.squeeze(0) for r in response_tensors]
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)  # type: ignore
            logger.debug(f"After PPO step", main_process_only=False)

            b_len = len(batch["response"])
            stats["train/decoder_rate"] = decoded_num / b_len
            stats["train/caught_rate"] = caught_num / b_len
            stats["train/success_rate"] = success_num / b_len
            stats["dl/epoch"] = epoch
            ppo_trainer.log_stats(  # type: ignore
                stats,
                batch,
                [r.to(torch.float32) for r in rewards],
                columns_to_log=["bit", "query", "response"],
            )

    ppo_trainer.save_pretrained(
        f"artifacts/checkpoints/{context.model_name.replace('/', '_')}"
    )


@torch.no_grad()
def greedy_search(
    model: AutoModelForCausalLMWithValueHead,
    max_new_tokens: int,
    input_ids: torch.Tensor,
    return_prompt: bool = True,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Greedy search for a model instead of model.generate.

    Args:
        model: Model to use for inference.
        input_ids: Input tensor with or without batch dimention.
        max_new_tokens: Maximum number of tokens to generate.
        return_prompt: If True, return the full tensor with the prompt.

    Returns:
        torch.Tensor: Tensor with the generated tokens.
    """

    # Avoid using model.generate as it does not gather the weights for the LM head
    # if FSDP. # See https://github.com/pytorch/pytorch/issues/100069
    orig_dim = input_ids.dim()
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    for _ in range(max_new_tokens):
        ans = model.forward(input_ids, attention_mask=attention_mask)
        logits = ans[0]
        new_token = logits.argmax(dim=-1)[..., -1:]
        if new_token.item() == model.tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, new_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(new_token, dtype=torch.int8)], dim=-1
            )  # type: ignore
    if orig_dim == 1 and input_ids.dim() != 1:
        input_ids = input_ids.squeeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)
    if return_prompt:
        return input_ids
    return input_ids[..., input_ids.shape[-1] - max_new_tokens :]


def get_reward(has_hidden_message: bool, is_decoded: bool, is_caught: bool) -> float:
    table = {
        (True, True, False): 1.0,
        (True, True, True): 0.1,
        (True, False, True): -0.1,
        (False, True, True): 0.01,
        (False, True, False): 0.01,
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
    formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
    tokens = tokens.to(device)
    assert (
        tokens["input_ids"].dim() == 2
    ), f"Expected {tokens['input_ids'].shape=}, dim = 2"
    # res = greedy_search(
    #    ppo_trainer.model,  # type: ignore
    #    max_new_tokens=context.output_max_length,
    #    **tokens,
    # )
    res = ppo_trainer.model.generate(tokens["input_ids"], **context.generation_kwargs)
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

        logger.debug(f"Start reward_el")
        response, text, bits, bit = (
            el["response"],
            el["text"],
            el["bits"],
            el["bit"],
        )
        assert bit is not None and not len(bit.strip()) != len(bit)
        assert isinstance(response, str), f"{type(response)=} is not str"
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
    logger.debug(f"Start reward_batch")
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
    """Weaker collate for ppo_trainer which doesn't stack tensors."""

    return {key: [d[key] for d in data] for key in data[0]}


def _test_():
    pass

    # hidden_message_sample = next(
    #    s for s in DataLoader(steg_ds["train"], batch_size=1) if s["bit"][0]
    # )

    # innocuous_sample = next(
    #    s for s in DataLoader(steg_ds["train"], batch_size=1) if not s["bit"][0]
    # )

    # logger.info("Receiver:")
    # ans, decoded_bit = receive(
    #    hidden_message_sample['text'][0],
    #    parse_final_message(hidden_message_sample["response"][0]),
    #    context.bits
    # )
    # logger.info("With hidden:")
    # logger.info("=============================================")
    # logger.info(decoded_bit)
    # logger.info(ans)
    # logger.info("=============================================")

    # ans, decoded_bit = receive(
    #    innocuous_sample['text'][0],
    #    parse_final_message(innocuous_sample["response"][0]),
    #    context.bits
    # )
    # logger.info("No hidden:")
    # logger.info("=============================================")
    # logger.info(ans)
    # logger.info("=============================================")

    # logger.info('Calculating "pretrained" hits"')
    # val_size = 20
    # subset = steg_ds["test"].select(range(val_size))
    # batch = next(iter(DataLoader(subset, batch_size=val_size, collate_fn=collator)))
    # assert val_size == len(batch["text"])
    # has_sent_secret_num = sum(1 for bit in batch["bit"] if bit)
    # rewards, caught_num, decoded_num, success_num = reward_batch(
    #     batch,
    #     verbose=True,
    # )
    # logger.info(f"{rewards=}")
    # logger.info(f"Decoded rate: {decoded_num}/{val_size}")
    # logger.info(f"Caught rate: {caught_num}/{val_size}")
    # logger.info(f"Success rate: {success_num}/{has_sent_secret_num}")


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
    context = args.context
    context.log_with="wandb"  # TODO: move to cli

    main(context)
