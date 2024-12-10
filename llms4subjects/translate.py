import json
from llms4subjects.llm import LLM
from tqdm import tqdm

"""把记录的标题和摘要翻译成英文和德文两个版本"""


bot = LLM(
    base_url="http://14.152.45.76:3073/v1",
    model="llama3.3:latest",
)

def translate_by_llm(text:str, language:str):
    response = bot.complete(f"Please enter the text below for translation into {language}. If the text is already in {language}, then output it directly. Output only the translation result without any other auxiliary information.\n\n{text}", max_tokens=2048)
    response = json.loads(response)
    return response["choices"][0]["text"].strip()

def translate_jsonline_file(jl_file:str, out_file:str):
    with open(jl_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f.readlines()]
        
    with open(out_file, "w", encoding="utf-8") as f:
        for item in tqdm(items):
            title = item['title']
            abstract = item['abstract']
            title_DE = translate_by_llm(title, "German")
            title_EN = translate_by_llm(title, "English")
            abstract_DE = translate_by_llm(abstract, "German")
            abstract_EN = translate_by_llm(abstract, "English")
            item["title_DE"] = title_DE
            item["title_EN"] = title_EN
            item["abstract_DE"] = abstract_DE
            item["abstract_EN"] = abstract_EN
            json_record = json.dumps(item, ensure_ascii=False)
            f.write(json_record + '\n')
            

if __name__ == '__main__':
    translate_jsonline_file("TIBKAT-core.jsonline", "translated-core.jsonline")
    translate_jsonline_file("TIBKAT-all.jsonline", "translated-all.jsonline")
    