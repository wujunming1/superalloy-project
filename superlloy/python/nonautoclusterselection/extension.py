import os


def file_extension(path):
    """
    获取文件路径中中的文件拓展名
    :param path: 文件路径
    :return: 文件拓展名
    """
    return os.path.splitext(path)[1].split('.')[1]


def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False


def is_bool(s):
    try:  # 如果能运行bool(s)语句，返回True（字符串s是布尔型数据）
        bool(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    return False


def is_char(s):
    if len(s) == 1:
        return True
    else:
        return False

