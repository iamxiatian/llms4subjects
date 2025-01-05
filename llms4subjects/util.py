import hashlib


def rm_leading_blanks(s: str) -> str:
    # 将多行字符串按行分割成列表
    lines = s.splitlines()

    # 使用列表推导式去掉每行开头的空格
    stripped_lines = [line.lstrip() for line in lines]

    return "\n".join(stripped_lines)


def md5_of_file(file_path) -> str:
    # 创建一个md5对象
    md5_hash = hashlib.md5()

    # 以二进制模式打开文件，读取文件内容并更新到md5对象中
    with open(file_path, "rb") as file:
        # 读取文件内容，每次读取1024字节（你可以根据需要调整这个值）
        for chunk in iter(lambda: file.read(1024), b""):
            md5_hash.update(chunk)

    # 获取十六进制格式的md5值
    md5_value = md5_hash.hexdigest()
    return md5_value
