def rm_leading_blanks(s: str) -> str:
    # 将多行字符串按行分割成列表
    lines = s.splitlines()

    # 使用列表推导式去掉每行开头的空格
    stripped_lines = [line.lstrip() for line in lines]

    return "\n".join(stripped_lines)
