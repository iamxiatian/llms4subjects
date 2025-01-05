import json

from llms4subjects.instance import get_embedding_query
from llms4subjects.llm import LLM
from llms4subjects.subject import subject_eq, subject_db_all as subject_db
from llms4subjects.util import rm_leading_blanks

from . import Predictor

base_url = "http://10.96.1.43:3087/v1"
model_name = "/data/app/yangyahe/base_model/LLM-Research-Meta-Llama-3.1-8B-Instruct-AWQ-INT4"


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


def make_prompt(
    title: str, abstract: str, dataset_type: str = "all", topk=5
) -> str:
    examples = _find_examples(title, abstract, dataset_type, topk)
    words = abstract.split(" ")
    if len(words) > 500:
        words = words[0:500]
        words.append("...")
    words = " ".join(words)

    return rm_leading_blanks(f"""You are a librarian responsible for assigning a set of subject tags to technical documents based on their titles and abstracts. Here are some examples:

    {examples}

    Now, please complete the subject list based on the following title and abstract:
    - Title: {title}
    - Abstract: {words}
    - Subjects: 
    """)


class PredictByExamples(Predictor):
    def __init__(self, dataset_type: str = "all", topk: int = 5):
        self.dataset_type = dataset_type
        self.topk = topk
        self.client = LLM(base_url, model=model_name)

    def predict(self, title, abstract) -> tuple[list[str], list[str]]:
        prompt = make_prompt(title, abstract, self.dataset_type, self.topk)

        # messages = [
        #     {"role": "user", "content": prompt},
        # ]
        response_text = self.client.complete(prompt)
        record = json.loads(response_text)
        assistant_text = record["choices"][0]["text"]
        
        
        #主题所在的行，有可能因为额外输出，需要抽取
        subject_line = None 
        for line in assistant_text.split("\n"):
            line = line.strip()
            if len(line)>10 and line[0] == "-":
                line = line[1:].strip()
            if len(line)> 10 and line[0] == "[" and line[-1] =="]":
                subject_line = line
                break
        
        if subject_line is None:
            subject_line = assistant_text
        
        names, codes = [], []
        
        try:
            items = json.loads(subject_line)
            
            for item in items:
                # 模型预测的名称不一定正确，需要反向查找subject
                code = subject_eq.get_code_by_name(item)
                name = subject_db.get_name_by_code(code)
                names.append(name)
                codes.append(code)
        except Exception as e:
            print(f"LLM_ERROR for title: {title}\n{e}\n{response_text}")
        return codes, names