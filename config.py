import tomli
from pathlib import Path
import pprint

conf_path = Path(__file__).parent.joinpath("config.toml")

with open(str(conf_path), 'rb') as f:
	c = tomli.load(f)
	pprint.pprint(c)