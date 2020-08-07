class DataTypeError(Exception):
    def __init__(self):
        super().__init__(self)  # 初始化父类
        self.message = "您输入的文件不符合规定的文件类型，请检查！"

    def __str__(self):
        return self.message
