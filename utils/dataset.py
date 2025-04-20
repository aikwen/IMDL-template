from pathlib import Path
from typing import Callable, Tuple, List, Dict, Union
import json
from pprint import pprint

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH.joinpath("data")

def pattern_coverage(tp: str,)-> str:
    return f"{tp[:-1]}forged"

def pattern_casia(tp: str)->str:
    return f"{tp}_gt"

def pattern_nist16(tp: str, gt_json:dict) -> str:
    return gt_json[tp]

def pattern_columbia(tp: str):
    pass

def load_json(path:str) -> Union[List, Dict, None]:
    if not Path(path).exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        gt_json = json.load(file)
    return gt_json

def data_names(dir:str,
              pattern:Callable[..., str]) -> Tuple[List[str], List[str]]:
    """
    Args:
        dir (str):  the dataset in the data directory
        pattern (Callable): the name correspondence pattern between tp and gt
    Returns:
        tp (list), gt (list)
    """
    assert DATA_PATH.joinpath(dir).exists(), \
        f"dataset {dir} is not exists"
    assert DATA_PATH.joinpath(dir, "gt").exists(), \
        f"dataset {dir}'s gt is not exists"
    assert DATA_PATH.joinpath(dir, "tp").exists(), \
        f"dataset {dir}'s tp is not exists"
    gt_dir = DATA_PATH.joinpath(dir, "gt")
    tp_dir = DATA_PATH.joinpath(dir, "tp")
    # gt_items dict { gt_item.stem : gt_item.suffix}
    gt_items = {gt_item.stem:gt_item.suffix for gt_item in gt_dir.iterdir()}
    # load json
    gt_json = load_json(str(DATA_PATH.joinpath(dir, "gt.json")))
    gt_names = []
    tp_names = []
    for tp_item in tp_dir.iterdir():
        if gt_json != None:
            gt_stem = pattern(tp_item.stem, gt_json)
        else:
            gt_stem = pattern(tp_item.stem)
        suffix = gt_items.get(gt_stem, ".invalid")
        assert suffix != ".invalid", \
            f"No matching gt file {gt_stem} found for tp file {tp_item.name} in {dir}"
        gt_names.append(f"{gt_stem}{suffix}")
        tp_names.append(tp_item.name)
    return tp_names, gt_names


if __name__ == "__main__":
    """
    test
    """
    tp, gt = data_names("coverage", pattern_coverage)
    print(tp[:10], len(tp))
    print(gt[:10], len(gt))

    tp, gt = data_names("casia_v1", pattern_casia)
    pprint(tp[:10])
    pprint(gt[:10])