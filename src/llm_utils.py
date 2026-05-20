import time
from typing import List, Dict, Any, Tuple

from openai import RateLimitError, BadRequestError, APIError

from . import pillar_globals as g
from .logging_utils import print_log

class ErrorPromptTooLong(Exception):
    pass

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
    print_log("Sending request to model " + model + " with messages: " + str(messages))
    retry_count = 0
    while retry_count < 3:
        try:
            response = g.client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
                stream=Stream,
            )
            retry_count = 3
        except RateLimitError as e:
            print_log(f"Rate limit exceeded: {e}")
            print_log("waiting for 60 seconds...")
            time.sleep(60)
            g.total_time_wasted += 60
            retry_count += 1
        except BadRequestError as e:
            print_log(f"Bad request error: {e}")
            print_log("Prompt: " + str(messages))
            # print_log("Waiting for user to fix the error...")
            if "Range of input length" in str(e):
                raise ErrorPromptTooLong("The prompt is too long to be processed.")
            else:
                input("Press Enter to retry...\n")
                print_log("Retrying...")
            retry_count += 1
        except APIError as e:
            print_log(f"API error: {e}")
            print_log("Retrying...")
            retry_count += 1
            
    try:
        reasoning_content, answer_content, created_time = split_response(response)
        return reasoning_content, answer_content, created_time
    except Exception as e:
        print_log(f"Bad request error during response splitting: {e}")
        print_log("Prompt: " + str(messages))
        if "Range of input length" in str(e):
            raise ErrorPromptTooLong("The prompt is too long to be processed.")
        else:
            return "", "", time.time()