"""GND CODE信息"""
import json
from tqdm import tqdm
from llms4subjects.translate import translate_by_llm

GND_subjects_all_file = "data/shared-task-datasets/GND/dataset/GND-Subjects-all.json"

GND_subjects_core_file = "data/shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json"

with open(GND_subjects_all_file, 'r', encoding='utf-8') as f:
    # 使用json.load()方法将文件内容解析为Python对象
    subjects_all = json.load(f)
    
with open(GND_subjects_core_file, 'r', encoding='utf-8') as f:
    # 使用json.load()方法将文件内容解析为Python对象
    subjects_core = json.load(f)
    
cores = subjects_core
alls = subjects_all

# cores = { e['Code']: e for e in subjects_core}
# alls =  { e['Code']: e for e in subjects_all}

def translate_names(out_file:str = "name-mapping.jsonline"):
    """把所有的名称都翻译成英语和德语两种，并保存到文件out_file中,
    每一行的格式为： {"name": name, "EN": 英文名, "DE": 德文名}"""
    names = []
    for entry in subjects_all:
        names.append(entry['Classification Name'])
        names.append(entry['Name'])
        if "Alternate Name" in entry:
            names.extend(entry["Alternate Name"])
        if "Related Subjects" in entry:
            names.extend(entry["Related Subjects"])
            
    # 先读取输出文件，避免重复处理
    with open(out_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f.readlines() ]
        generated_names = { item["name"] for item in items }
        
    # 开始生成subject的英文和德文版本的名称  
    names = set(names)
    with open(out_file, "a+", encoding="utf-8") as f:
        for name in tqdm(names):
            if name not in generated_names:
                en = translate_by_llm(name, "English")
                de = translate_by_llm(name, "German")
                json_record = json.dumps(
                    { "name": name, "EN": en, "DE": de},
                    ensure_ascii=False)
                f.write(json_record + '\n')
                f.flush()
            
if __name__ == '__main__':
    translate_names()