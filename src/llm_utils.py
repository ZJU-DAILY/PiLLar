import time
from typing import List, Dict, Any, Tuple

from openai import RateLimitError, BadRequestError, APIError

from . import pillar_globals as g
from .logging_utils import print_log


def split_response(response) -> Tuple[str, str, float]:
    reasoning_content = ""
    answer_content = ""
    created_time = None

    for chunk in response:
        delta = chunk.choices[0].delta
        if created_time is None:
            created_time = chunk.created

        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content

        if hasattr(delta, "content") and delta.content:
            answer_content += delta.content

    return reasoning_content, answer_content, created_time or time.time()


def completion_create(
    model: str,
    messages: List[Dict[str, Any]],
    extra_body: Dict[str, Any] | None = None,
    Stream: bool = False,
):
    get_response = False
    while not get_response:
        try:
            response = g.client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
                stream=Stream,
            )
            get_response = True
        except RateLimitError:
            print_log("Rate limit exceeded, waiting for 10 seconds...")
            time.sleep(10)
            g.total_time_wasted += 10
        except BadRequestError as e:
            print_log(f"Bad request error: {e}")
            print_log("Retrying...")
        except APIError as e:
            print_log(f"API error: {e}")
            print_log("Retrying...")

    return split_response(response)
