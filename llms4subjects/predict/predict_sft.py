import json

from llms4subjects.instance import get_embedding_query
from llms4subjects.llm import LLM
from llms4subjects.subject import subject_eq, subject_db_all as subject_db
from llms4subjects.util import rm_leading_blanks
from llms4subjects.prompt import make_input_text
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from . import Predictor


def _find_examples(
    title: str, abstract: str, dataset_type: str = "all", topk=5
) -> str:
    text = f"""title:{title}\nabstract:{abstract}"""
    eq = get_embedding_query(dataset_type)
    instances = eq.get_instances(text, topk)
    examples = []
    for i, inst in enumerate(instances, start=1):
        gnd_names = [subject_db.get_name_by_code(c) for c in inst.gnd_codes]
        words = inst.abstract.split(" ")
        if len(words) > 500:
            words = words[0:500]
            words.append("...")
        words = " ".join(words)
        names = ", ".join([f'"{name}"' for name in gnd_names])
        examples.append(
            rm_leading_blanks(f"""## Example {i}
        - Title: ```{inst.title}```
        - Abstract: ```{words}``` 
        - Subject: [{names}]
        """)
        )

    return "\n".join(examples)


class PredictBySftLlama(Predictor):
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.float16
        device = "auto"
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device
        )
        self.device = self.model.device

    def predict(self, title, abstract) -> tuple[list[str], list[str]]:
        words = abstract.split(" ")
        if len(words) > 2000:
            abstract = " ".join(words[:2000])
        input_text = make_input_text(title, abstract)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.device
        )
        max_length = len(input_ids[0])+1000

        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=False,
            )
        out = self.tokenizer.decode(output[0], skip_special_tokens=True)
        out_text = out[len(input_text) :].strip()

        names, codes = [], []

        try:
            items = out_text.split("\n")

            for item in items:
                # 模型预测的名称不一定正确，需要反向查找subject
                code = subject_eq.get_code_by_name(item)
                name = subject_db.get_name_by_code(code)
                names.append(name)
                codes.append(code)
        except Exception as e:
            print(f"LLM_ERROR for title: {title}\n{e}\n{out}")
        return codes, names
