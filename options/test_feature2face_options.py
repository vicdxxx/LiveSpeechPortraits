import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from options.base_options_feature2face import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--test_dataset_names', type=str, default='name', help='chooses validation datasets.')
  
        self.isTrain = False
