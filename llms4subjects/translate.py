import json
from llms4subjects.llm import LLM
from tqdm import tqdm
from pathlib import Path

"""把记录的标题和摘要翻译成英文和德文两个版本"""


bot = LLM(
    base_url="http://14.152.45.76:3073/v1",
    model="llama3.3:latest",
)

def translate_by_llm(text:str, language:str)->str:
    response = bot.complete(f"Please enter the text below for translation into {language}. If the text is already in {language}, then output it directly. Output only the translation result without any other auxiliary information.\n\n{text}", max_tokens=2048)
    response = json.loads(response)
    return response["choices"][0]["text"].strip()

def translate_jsonline_file(jl_file:str, out_file:str):
    with open(jl_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f.readlines()]
        
    # 先读取输出文件，避免重复处理
    if Path(out_file).exists():
        with open(out_file, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f.readlines() ]
            generated_ids = { item["id"] for item in items }
    else:
        generated_ids = {}
        
    with open(out_file, "a+", encoding="utf-8") as f:
        print(f'process {jl_file}, total items:', len(records))
        for record in tqdm(records):
            if record["id"] not in generated_ids:
                title = record['title']
                abstract = record['abstract']
                title_DE = translate_by_llm(title, "German")
                title_EN = translate_by_llm(title, "English")
                abstract_DE = translate_by_llm(abstract, "German")
                abstract_EN = translate_by_llm(abstract, "English")
                record["title_DE"] = title_DE
                record["title_EN"] = title_EN
                record["abstract_DE"] = abstract_DE
                record["abstract_EN"] = abstract_EN
                json_record = json.dumps(record, ensure_ascii=False)
                f.write(json_record + '\n')
            

if __name__ == '__main__':
    translate_jsonline_file("TIBKAT-core.jsonline", "translated-core.jsonline")
    translate_jsonline_file("TIBKAT-all.jsonline", "translated-all.jsonline")
    
    