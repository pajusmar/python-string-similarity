from .shingle_based import ShingleBased
from .string_distance import NormalizedStringDistance
from .string_similarity import NormalizedStringSimilarity


class OverlapCoefficient(ShingleBased, NormalizedStringDistance, NormalizedStringSimilarity):

    def __init__(self, k=3):
        super().__init__(k)

    def distance(self, s0, s1):
        return 1.0 - self.similarity(s0, s1)

    def similarity(self, s0, s1):
        if s0 is None:
            raise TypeError("Argument s0 is NoneType.")
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s0 == s1:
            return 1.0
        profile0, profile1 = self.get_profile(s0), self.get_profile(s1)
        union = set().union(profile0.keys(), profile1.keys())
        inter = int(len(profile0.keys()) + len(profile1.keys()) - len(union))
        return inter / min(len(profile0), len(profile1))
