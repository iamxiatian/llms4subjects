"""generate train data"""

from pyld import jsonld
import json
from pathlib import Path
from tqdm import tqdm

"""把图书、论文等文献的jsonld格式的内容，抽取出title、abstract和GND_ID，并存为jsonline格式，方便后续读取处理"""
def parse_jsonld(jsonld_file:Path) -> str|None:
    tibkat_id = jsonld_file.stem
    
    # 读取JSON-LD文件
    with open(jsonld_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 使用pyld库加载和展开JSON-LD
    expanded_data = jsonld.expand(data)
    for e in expanded_data[0]["@graph"]:
        id = e["@id"]
        if "https://www.tib.eu/de/suchen/id/TIBKAT" in id:
            title = e['http://purl.org/dc/elements/1.1/title'][0]["@value"]
            abstract = e['http://purl.org/dc/terms/abstract'][0]["@value"]
            gnd_ids = e['http://purl.org/dc/terms/subject']
            entry = { "id": tibkat_id, "title": title, "abstract": abstract, "gnd_ids": gnd_ids}
            json_record = json.dumps(entry, ensure_ascii=False)
            return json_record


def convert_jsonld_to_lines(train_dir:str, out_file:str):
    """把文件夹train_dir下的所有的jsonld文件，读取出其中的标题和摘要，
    作为jsonline文件的一行，进行保存，形成一个完整的jsonline文件。"""
    with open(out_file, "w", encoding="utf-8") as f:
        train_files = list(Path(train_dir).glob("**/*.jsonld"))
        for jsonld_file in tqdm(train_files):
            json_record = parse_jsonld(jsonld_file)
            f.write(json_record + '\n')
    

if __name__ == '__main__':
    convert_jsonld_to_lines("./data/shared-task-datasets/TIBKAT/all-subjects/data/train/", "TIBKAT-all.jsonline")
    
    convert_jsonld_to_lines("./data/shared-task-datasets/TIBKAT/tib-core-subjects/data/train/", "TIBKAT-core.jsonline")