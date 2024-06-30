# %%
from prompts import overseer_prompt

from openai import OpenAI

from utils import (
    capture_all_after_last,
    format_bits
)

from Context import Context

client = OpenAI()


# %%
def inspect(text, context : Context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": overseer_prompt.format(
                    text=text,
                    bits=format_bits(context.bits)
                ),
            },
        ],
    )
    response = response.choices[0].message.content
    yes_or_no = capture_all_after_last(response, "FINAL ANSWER:") or ""
    yes = "yes" in yes_or_no.lower()
    no = "no" in yes_or_no.lower()
    has_secret = None
    if yes and not no:
        has_secret = True
    elif no and not yes:
        has_secret = False
    return response, has_secret
# %%

def test_inspect():
    text = "I am a secret message"
    context = Context()
    response, has_secret = inspect(text, context)
    assert has_secret == True
    assert response == ""

if __name__ == "__main__":
    test_inspect()
# %%
