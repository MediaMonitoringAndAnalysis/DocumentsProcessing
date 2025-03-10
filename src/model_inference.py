
from typing import List, Dict, Literal
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from ast import literal_eval


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class InferencePipeline:
    def __init__(self, model_name: str, api_key: str = None):
        pass
    
    def inference(self, messages: List[Dict], structured_output: bool) -> str:
        pass

class VLMInferencePipeline(InferencePipeline):
    def __init__(
        self,
        model_name: Literal[
            "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"
        ],
        api_key: str = None,
    ):
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = _get_device()

    @torch.inference_mode()
    def inference(self, messages: List[Dict], structured_output: bool) -> str:
        """Run a VLM model inference for one entry."""
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if structured_output:
            return literal_eval(output_text)
        return output_text


class OpenAIInferencePipeline(InferencePipeline):
    def __init__(
        self,
        model_name: Literal["gpt-4o", "gpt-4o-mini"],
        api_key: str,
    ):
        from openai import OpenAI

        self.model = OpenAI(api_key=api_key)
        self.model_name = model_name

    def inference(self, messages: List[Dict], structured_output: bool) -> str:
        response = (
            self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"} if structured_output else None,
                temperature=0.0,
            )
            .choices[0]
            .message.content
        )
        if structured_output:
            return literal_eval(response)
        return response