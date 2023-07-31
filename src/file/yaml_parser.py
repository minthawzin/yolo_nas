import os
import yaml
from easydict import EasyDict as edict

class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, config_file=None):
        cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read(), Loader=yaml.SafeLoader))

        super(YamlParser, self).__init__(cfg_dict)
             

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.load(fo.read(), Loader=yaml.SafeLoader))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)
