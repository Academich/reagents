from typing import Callable, List


class ReactionPreprocessingPipeline:
    """
    pipeline = ReactionPreprocessingPipeline([
        f1,
        f2,
        ...,
        fn
    ])
    pipeline.run(x)
    """

    def __init__(self, stages: List[Callable[[str], str]]):
        self.stages = stages

    def run(self, smi: str) -> str:
        x = smi
        for stage in self.stages:
            try:
                x = stage(x)
            except Exception as e:
                return "!" + str(stage.__name__)
        return x
