"""
Directly queries the HoVer index with the query.
This is more meant for debugging and analysis than for end users.
"""
from argparse import ArgumentParser
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from tqdm import tqdm
import ujson

from goalbert.training.checkpoint import GCheckpoint

def load_collectionX(collection_path, dict_in_dict=False):
    """
    Same exact function as in Baleen, only reimported since Baleen fns don't seem to work.
    """
    print("Loading collection...")
    collectionX = {}

    with open(collection_path) as f:
        for line_idx, line in tqdm(enumerate(f)):
            line = ujson.loads(line)

            assert type(line['text']) is list
            assert line['pid'] == line_idx, (line_idx, line)

            passage = [line['title'] + ' | ' + sentence for sentence in line['text']]

            if dict_in_dict:
                collectionX[line_idx] = {}

            for idx, sentence in enumerate(passage):
                if dict_in_dict:
                    collectionX[line_idx][idx] = sentence
                else:
                    collectionX[(line_idx, idx)] = sentence

    return collectionX

def main():
    parser = ArgumentParser()
    parser.add_argument("--goalbert", action="store_true")
    parser.add_argument("--hops", type=int, default=1)
    args = parser.parse_args()

    with open("../sid_to_pid_sid.json", "r") as f:
        sid_to_pid_sid = ujson.load(f)
    collectionX = load_collectionX("../wiki.abstracts.2017/collection.json")
    with Run().context(RunConfig(nranks=1, experiment="wiki2017")):
        config = ColBERTConfig(
            root="./index",
            query_maxlen=32 if args.hops > 1 else 64,
        )
        searcher = Searcher(index="wiki2017.nbits=2", config=config)
        
        # Replace the checkpoint
        if args.goalbert:
            colbert = searcher.checkpoint
            goalbert = GCheckpoint(colbert.name, colbert_config=config)
            goalbert.load_state_dict(colbert.state_dict())
            searcher.checkpoint = goalbert
            del colbert

        while True:
            query = input("> ")
            context = []
            print("Searching...")
            for i in range(args.hops):
                print("Hop", i + 1)
                print("Context:", context)
                ranking = searcher.search(query, context=(" [SEP] ".join(context)) if context else None, k=10)
                doc_ids = ranking[0]
                for doc_id in doc_ids:
                    pid_sid = tuple(sid_to_pid_sid[str(doc_id)])
                    sent = collectionX.get(pid_sid)
                    print(sent)
                pid_sid = tuple(sid_to_pid_sid[str(doc_ids[0])])
                sent = collectionX.get(pid_sid)
                context.append(sent)

if __name__ == "__main__":
    main()