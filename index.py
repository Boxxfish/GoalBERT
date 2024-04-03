"""
Indexing script for GoalBERT.
Before training or evaluating, an index must be generated.
"""

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer


def main():
    with Run().context(RunConfig(nranks=1, experiment="wiki2017")):
        config = ColBERTConfig(
            nbits=2,
            root="./index",
        )
        indexer = Indexer(checkpoint="../colbertv2.0", config=config)
        indexer.index(name="wiki2017.nbits=2", collection="../fact_collection.tsv")


if __name__ == "__main__":
    main()
