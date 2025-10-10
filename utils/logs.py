from typing import List, Union, final

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.rule import Rule

# 表格属性
width = 50
# 训练数据集
t_datasets_title = "training datasets"
t_datasets_headers = ["Name", "Size"]
# 数据增强
pre_title = "train preprocess"
pre_headrs = ["order↓", "type", "p", "param"]
# base config
base_title = "base config"
base_headers = ["key", "value"]

def NewTableLog(headers:List, rows: List[List], title:str=""):
    table = Table(show_header=True,
                width=width,
                header_style="bold",
                title=title)

    for head in headers:
        table.add_column(head)

    for row in rows:
        table.add_row(*row)
    return table

class Logs:
    def __init__(self, filepath):
        self.filepath = filepath
        self.f = open(filepath, "a")
        self.console = Console(file=self.f)

    def print(self, content: Union[Table, Text]):
        self.console.print(content)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()


if __name__ == "__main__":
    with open("output.log", "a", encoding="utf-8") as f:
        console = Console(file=f, width=width)
        data_rows = [
            ["coco cm", "30000"],
            ["coco cm", "30000"],
            ["coco cm", "50000"]
        ]
        console.print(NewTableLog(t_datasets_headers, data_rows, t_datasets_title))

        pre_rows = [
            ["1", "resize", "1", ""],
            ["2", "gnoise", "0.5","[0.0117,0.0588]"]
            ]
        console.print(NewTableLog(pre_headrs, pre_rows, pre_title))
        base_rows = [
            ["epochs", "30"],
            ["batchsize", "16"],
            ["adamw-lr", "0.001"],
            ["adamw-weight_decay", "0.01"],
            ["polynomialLR-power", "0.9"]
        ]
        console.print(NewTableLog(base_headers, base_rows, base_title))

        console.print(Rule(characters="*"))
        # 使用 Text.assemble 配合 f-string
        header = Text.assemble(
            (f"{'epoch/batch':^15}", "bold"),
            (f"{'loss':^10}", "bold"),
            (f"{'lr':^6}", "bold")
        )

        row1 = Text.assemble(
            (f"{1:>7}/{10000:<8}", "bold"),
            (f"{80.9999:^10}", "bold"),
            (f"{80.9999:^10}", "bold"),
        )


        console.print(header)
        console.print(row1)

