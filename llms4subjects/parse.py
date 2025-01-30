import json
from pathlib import Path
from pyld import jsonld

def parse_jsonld(jsonld_file: Path|str) -> dict | None:
    """
    读取jsonld_file，返回id、title、abstract、gnd_codes、language、
    doctype字典。对于测试集，gnd_codes在文件中不存在，此时字典中的
    gnd_codes为None。
    
    例如：
    ```json
    {'abstract': 'About 10% of the US labor force is employed in selling related '
             'occupations and the expenditures on selling activities total '
             'close to 5% of the US GDP. Without question, selling occupies a '
             'prominent role in our economy. This chapter offers a discussion '
             'on the construct of selling, its role in economic models, and '
             'the various aspects of firm decisions that relate to it.',
    'doctype': 'Article',
    'gnd_codes': [{'@id': 'http://d-nb.info/gnd/4037589-4'}],
    'id': '3A1831630516',
    'language': 'en',
    'title': 'Chapter 8. Selling and sales management'}
    ```
    """
    if isinstance(jsonld_file, str):
        jsonld_file = Path(jsonld_file)
    tibkat_id = jsonld_file.stem
    language = jsonld_file.parent.name
    doctype = jsonld_file.parent.parent.name
    # 读取JSON-LD文件
    with open(jsonld_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 使用pyld库加载和展开JSON-LD
    expanded_data = jsonld.expand(data)
    for e in expanded_data[0]["@graph"]:
        my_id = e["@id"]
        if "https://www.tib.eu/de/suchen/id/TIBKAT" in my_id:
            title = e["http://purl.org/dc/elements/1.1/title"][0]["@value"]
            abstract = e["http://purl.org/dc/terms/abstract"][0]["@value"]
            gnd_codes = None
            if "http://purl.org/dc/terms/subject" in e:
                gnd_codes = e["http://purl.org/dc/terms/subject"]
                
            entry = {
                "id": tibkat_id,
                "title": title,
                "abstract": abstract,
                "gnd_codes": gnd_codes,
                "language": language,
                "doctype": doctype,
            }
            return entry
