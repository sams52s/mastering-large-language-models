import torch
import pynvml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from typing import Dict, Any

def _process_line(line: str, prefix: str) -> str:
    """
    Extracts the primary content from a formatted line by removing the given prefix,
    stripping an enumeration token (e.g., "1.", "a)", etc.) and any trailing punctuation.

    The function performs the following steps:
    1. Removes the specified prefix from the line.
    2. Searches for known enumeration tokens (like "1.", "a.", "1)", "a)") and extracts the text following the first token found.
       It stops the extraction at the next token (e.g., "2.", "b.", "2)", or "b)"), if present.
    3. Strips any trailing punctuation characters (.,;:()[]{}).

    Examples:
        >>> process_line("PREFIX 1. Hello, world! 2. Goodbye", "PREFIX ")
        'Hello, world!'
        >>> process_line(">>> a) This is an example text. b) Next text", ">>> ")
        'This is an example text'
    
    Motivation:
        LLM's is prone to generate more than one definition / example. For example, for 'apple', it might easily generate something like:
        word: apple
        definition: 1. a fruit that grows on trees. 2. a company that makes phones.
    """
    # Remove the prefix and leading/trailing whitespace
    content = line[len(prefix):].strip()

    # Define pairs of tokens: the start token to extract from and the next token indicating the end of the desired segment.
    token_pairs = [("1.", "2."), ("a.", "b."), ("1)", "2)"), ("a)", "b)")]

    for start_token, end_token in token_pairs:
        if start_token in content:
            start_index = content.find(start_token) + len(start_token)
            end_index = content.find(end_token, start_index)
            if end_index == -1:
                end_index = len(content)
            content = content[start_index:end_index].strip()
            break

    # Remove trailing punctuation characters
    content = content.strip(".,;:()[]{}")

    return content


class Helpers:
    @staticmethod
    def load_model_and_tokenizer(model_name: str, quantize: bool = False, is_prompt_tuning: bool = False) -> tuple:
        """Load model with proper configuration for quantization and prompt tuning. The model will be frozen"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # unsupported in latest version of bitsandbytes for MacOS
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quantize,
            bnb_4bit_compute_dtype=torch.float32,
        ) if quantize else None

        # TO-DO: why auto + cuda_visible_devices=6 works and "cuda:6" doesn't?
        # device_map = {"": Helpers.get_cuda_device_with_most_free_memory()} if torch.cuda.is_available() else "cpu"
        # device_map = {"": "cuda:6"} if torch.cuda.is_available() else "cpu"
        device_map = "auto"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quantization_config,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )
        model.gradient_checkpointing_enable()
        if is_prompt_tuning:
            model.enable_input_require_grads()
        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        return model, tokenizer

    @staticmethod
    def get_output(model, tokenizer, prompt: str, params: Dict[str, Any], device: str) -> str:
        """Generate text output using model with given parameters"""
        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **params,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


    @staticmethod
    def get_example_and_definition(model, tokenizer, word: str, params: Dict[str, Any], device: str) -> Dict:
        """Get processed definition and example using notebook heuristics"""
        # Generate definition
        def_prompt = f"word: {word}\ndefinition: "
        def_output = Helpers.get_output(model, tokenizer, def_prompt, params, device)
        definition = _process_line(def_output.split("\n")[1], "definition: ")

        # Generate example
        ex_prompt = f"{def_prompt}{definition}\nexample: "
        ex_output = Helpers.get_output(model, tokenizer, ex_prompt, params, device)
        example = _process_line(ex_output.split("\n")[2], "example: ")

        return {
            "word": word,
            "definition": definition,
            "example": example
        }
    
    @staticmethod
    def get_cuda_device_with_most_free_memory(verbose: bool = False) -> str:
        """
        Returns the CUDA device identifier (e.g. "cuda:0") that has the most free memory available.
        
        This function uses the NVML library (via pynvml) to query each GPU's free memory.
        It compares the free memory (in bytes) across all available GPUs and returns the device
        identifier corresponding to the GPU with the highest free memory.
        
        Args:
            verbose: If True, print the free memory of each GPU.

        Raises:
            RuntimeError: If no CUDA devices are found.
        
        Returns:
            A string representing the CUDA device (e.g. "cuda:0") with the most free memory.
        """
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            raise RuntimeError("No CUDA devices found.")

        best_device = 0
        best_free_memory = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem = mem_info.free  # Free memory in bytes
            if verbose:
                print(f"### Device {i}")
                print(f"Total memory: {mem_info.total / 1024**2:.2f} MiB")
                print(f"Used memory: {mem_info.used / 1024**2:.2f} MiB")
                print(f"Free memory: {mem_info.free / 1024**2:.2f} MiB")

            if free_mem > best_free_memory:
                best_free_memory = free_mem
                best_device = i

        pynvml.nvmlShutdown()
        return f"cuda:{best_device}"

    @staticmethod
    def convert_to_serializable(obj):
        """
        Convert an object to a serializable format (e.g., dict, list, etc.) by recursively converting its components.

        :param obj: The object to convert.
        :return: The object in a serializable format.
        """
        if isinstance(obj, dict):
            return {k: Helpers.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [Helpers.convert_to_serializable(item) for item in obj]
        return obj