"""把图书、论文等文献的jsonld格式的内容，抽取出title、abstract和gnd_codes，
并存为jsonline格式，方便后续读取处理"""

from pyld import jsonld
import json
from pathlib import Path
from tqdm import tqdm
from llms4subjects.instance import tibkat_db


def parse_jsonld(jsonld_file: Path) -> str | None:
    tibkat_id = jsonld_file.stem

    # 读取JSON-LD文件
    with open(jsonld_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 使用pyld库加载和展开JSON-LD
    expanded_data = jsonld.expand(data)
    for e in expanded_data[0]["@graph"]:
        id = e["@id"]
        if "https://www.tib.eu/de/suchen/id/TIBKAT" in id:
            title = e["http://purl.org/dc/elements/1.1/title"][0]["@value"]
            abstract = e["http://purl.org/dc/terms/abstract"][0]["@value"]
            gnd_codes = e["http://purl.org/dc/terms/subject"]
            entry = {
                "id": tibkat_id,
                "title": title,
                "abstract": abstract,
                "gnd_codes": gnd_codes,
            }
            json_record = json.dumps(entry, ensure_ascii=False)
            return json_record


def convert_jsonld_to_lines(train_dir: str, out_file: str):
    """把文件夹train_dir下的所有的jsonld文件，读取出其中的标题和摘要，
    作为jsonline文件的一行，进行保存，形成一个完整的jsonline文件。"""
    with open(out_file, "w", encoding="utf-8") as f:
        train_files = list(Path(train_dir).glob("**/*.jsonld"))
        for jsonld_file in tqdm(train_files):
            json_record = parse_jsonld(jsonld_file)
            f.write(json_record + "\n")


def _load_jsonline(jl_file: str) -> tuple[set, list]:
    records = []
    with open(jl_file, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            records.append(json.loads(line))
    gnd_codes = [e["id"] for e in records]
    return set(gnd_codes), records


def save_sqlite():
    """把导出的内容存入sqlite数据库"""
    core_gnd_codes, core_records = _load_jsonline("TIBKAT-core.jsonline")
    all_gnd_codes, all_records = _load_jsonline("TIBKAT-all.jsonline")
    for record in tqdm(core_records, leave=True):
        my_id = record["id"]
        source = "all,core" if (my_id in all_gnd_codes) else "core"

        title, abstract, gnd_codes = (
            record["title"],
            record["abstract"],
            record["gnd_codes"],
        )
        # 读出gnd_id
        start = len("http://d-nb.info/gnd/")
        gnd_codes = [e["@id"][start:] for e in gnd_codes]
        tibkat_db.insert_record(my_id, title, abstract, "", gnd_codes, source)

    for record in tqdm(all_records, leave=True):
        my_id = record["id"]
        if my_id not in core_gnd_codes:
            title, abstract, gnd_codes = (
                record["title"],
                record["abstract"],
                record["gnd_codes"],
            )
            # 读出gnd_id
            start = len("http://d-nb.info/gnd/")
            gnd_codes = [e["@id"][start:] for e in gnd_codes]
            tibkat_db.insert_record(my_id, title, abstract, "", gnd_codes, "all")

    tibkat_db.close()


if __name__ == "__main__":
    print("convert jsonld files to jsonline file")
    convert_jsonld_to_lines("./data/shared-task-datasets/TIBKAT/all-subjects/data/train/", "TIBKAT-all.jsonline")

    convert_jsonld_to_lines("./data/shared-task-datasets/TIBKAT/tib-core-subjects/data/train/", "TIBKAT-core.jsonline")

    print("save to sqlite db.")
    save_sqlite()
