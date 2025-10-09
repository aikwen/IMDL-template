import yaml
from pathlib import Path
import pprint

conf_path = Path(__file__).parent.joinpath("config.yaml")

with open(str(conf_path), 'rb') as f:
	c = yaml.safe_load(f)
	pprint.pprint(c)