

from typing import Dict, List, Optional, Sequence, Tuple, Union
from .logger import get_logger
logger = get_logger(__name__)

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
OBSERVATION_ROLE = "observation"
FUNCTION_ROLE = "function"


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", data_args: "DataArguments") -> "Template":
    r"""
    Gets chat template and fixes the tokenizer.
    simple version, there are some limitations: 
    1. only use QWEN template currently
    2. not tool or func
    3. not jinja_template
    """

    # 
    template = Template()

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    return template

class Template:
    def __init__(self) -> None:            
        # name="qwen"
        self.format_prefix=''
        self.format_function=''
        self.format_tools=''
        self.format_system="<|im_start|>tool\n{content}<|im_end|>\n"
        self.format_user="<|im_start|>user\n{content}<|im_end|>\n"
        self.format_assistant="<|im_start|>assistant\n{content}<|im_end|>\n"
        self.format_separator='\n'
        self.stop_words=["<|im_end|>"]
        self.replace_eos=True
        self.replace_jinja_template=False

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        answer_ids = encoded_messages[-1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def extract_tool(self, content: str) -> Union[str, List[Tuple[str, str]]]:
        r"""
        Extracts tool message.
        """
        return self.format_tools.extract(content)

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: str,
        tools: Optional[str],
    ) -> List[List[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query                    resp
        """
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix
                elements += self.format_system.format(content=system)
                # if system or tools:
                #     tool_text = self.format_tools.format(content=tools)[0] if tools else ""
                #     elements += self.format_system.format(content=(system + tool_text))
                    

            if message["role"] == USER_ROLE:
                elements += self.format_user.format(content=message["content"])
            elif message["role"] == ASSISTANT_ROLE:
                elements += self.format_assistant.format(content=message["content"])
            elif message["role"] == OBSERVATION_ROLE:
                elements += self.format_observation.format(content=message["content"])
            elif message["role"] == FUNCTION_ROLE:
                elements += self.format_function.format(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: List[str]) -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            assert isinstance(elem, str)
            if len(elem) != 0:
                token_ids += tokenizer.encode(elem, add_special_tokens=False)
        return token_ids
