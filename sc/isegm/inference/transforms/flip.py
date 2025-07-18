import jittor

from typing import List
from isegm.inference.clicker import Click
from .base import BaseTransform


class AddHorizontalFlip(BaseTransform):
    def transform(self, image_nd, clicks_lists: List[List[Click]]):
        assert len(image_nd.shape) == 4
        image_nd = jittor.cat([image_nd, jittor.flip(image_nd, dim=[3])], dim=0)

        image_width = image_nd.shape[3]
        clicks_lists_flipped = []
        for clicks_list in clicks_lists:
            clicks_list_flipped = [click.copy(coords=(click.coords[0], image_width - click.coords[1] - 1))
                                   for click in clicks_list]
            clicks_lists_flipped.append(clicks_list_flipped)
        clicks_lists = clicks_lists + clicks_lists_flipped

        return image_nd, clicks_lists

    def inv_transform(self, prob_map):
        assert len(prob_map.shape) == 4 and prob_map.shape[0] % 2 == 0
        num_maps = prob_map.shape[0] // 2
        prob_map, prob_map_flipped = prob_map[:num_maps], prob_map[num_maps:]
        return 0.5 * (prob_map + jittor.flip(prob_map_flipped, dim=[3]))

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def reset(self):
        pass
