def to_alpaca_entry(
    title: str, abstract: str, subject_names: list[str]
) -> dict:
    entry = {
        "instruction": "You are a librarian responsible for assigning a set of subject tags to a technical document based on its title and abstract. Based on the information provided below, please output the related subjects, with one subject per line.",
        "input": f"- Title: {title}\n- Abstract: {abstract}",
        "output": "\n".join(subject_names),
    }
    return entry


def make_input_text(title:str, abstract:str) -> str:
    return f"""You are a librarian responsible for assigning a set of subject tags to a technical document based on its title and abstract. Based on the information provided below, please output the related subjects, with one subject per line.
- Title: {title}
- Abstract: {abstract}
"""
