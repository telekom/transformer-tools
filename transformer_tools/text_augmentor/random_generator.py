# Copyright (c) 2021 Sitong Ye, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import yaml
import random

# load available methods in yaml
# current
class RamdomAugGenerator:

    def __init__(self, aug_config, object_map, shuffle_weight=[0.5, 0.5], swap_prob=0.8):
        """
        input:
            aug_config: takes in both ".yaml" (specific for augmentation) or directly dictionary type
            object_map: dictionary, which maps the namespace in config file to the object
        """
        if isinstance(aug_config, str):
            if aug_config.endswith(".yaml") or aug_config.endswith(".yml"):
                with open(aug_config, 'r') as config:
                    self.cfg = yaml.safe_load(config)
        elif isinstance(aug_config, dict):
            self.cfg = aug_config
        self.object_map = object_map
        self.shuffle_weight = shuffle_weight
        assert len(self.object_map) == len(shuffle_weight)
        self.swap_prob = swap_prob

        # initialize everything
        self.object_factory = {}
        for method in self.cfg:
            self.object_factory[method] = self.object_map[method](**self.cfg[method])

    def __call__(self, text):

        # when the superclass is called after initialization, it randomly choice from available subclasses,
        # currently it's hard coded here, which class is available. can be moved to yaml file
        if random.random() > self.swap_prob:
            # it does not swap, in this case, just return the input text
            # print("not augmented")
            return text
        # otherwise it will be swapped
        # shuffle the methods
        selected_method = random.choices(list(self.object_factory.keys()), weights=self.shuffle_weight)
        print("selected augmentation method:", selected_method[0])
        # initialise the selected_method correspondently
        out = self.object_factory[selected_method[0]].generate_augmentation(text)[0]
        return out
