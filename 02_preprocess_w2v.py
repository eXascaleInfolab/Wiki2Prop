import argparse

parser = argparse.ArgumentParser(description='Join attributes from Wikidata with Embeddings from Wikipedia2Vec')
parser.add_argument('--embeddings', '-e', nargs='+', default=["data/wikipedia2vec/enwiki-20180901_300d.pkl", "data/wikipedia2vec/dewiki-20180901_300d.pkl", "data/wikipedia2vec/frwiki-20180901_300d.pkl"], help='the wikipedia2vec embedding')
parser.add_argument('--Y', '-Y', type=argparse.FileType('r'), default="data/Y_20180813.pkl.bz2", help='the attributes for all ')
parser.add_argument('--languagemap', '-lm', type=argparse.FileType('rb'), default="data/sitelinks-wikidata-20180813.pkl", help='the languagemap')
args = parser.parse_args()

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

import pickle
logging.info("loading: " + args.languagemap.name)
langm = pickle.load(args.languagemap)

import pandas as pd
logging.info("loading: " + args.Y.name)
Y = pd.read_pickle(args.Y.name)

version_Y = args.Y.name[-16:][:8]

from wikipedia2vec import Wikipedia2Vec
emb = {}
version_emb = ""
for embf in args.embeddings:
    logging.info("loading: " + embf)
    name = embf[embf.rfind("/")+1:-4]
    emb[name] = Wikipedia2Vec.load(embf)


logging.info("extracting features / rearranging data")

XCount = {} 
for e in emb:
    XCount[e] = len(emb[e].syn0[0])

X = pd.DataFrame(0, index=Y.index, columns=range(sum(XCount.values())), dtype='float32')

skipped = 0
for Q in Y.index:
    try:
        XCountTmp = 0
        for e in emb:
            lang = e[:6]
            if lang == "enwiki":
                X.loc[Q][XCountTmp:XCountTmp+XCount[e]] = emb[e].get_entity_vector(langm[Q][lang])
            else:
                if Q in langm and langm[Q][lang] != None and emb[e].get_entity(langm[Q][lang]):
                    X.loc[Q][XCountTmp:XCountTmp+XCount[e]] = emb[e].get_entity_vector(langm[Q][lang])
            XCountTmp += XCount[e]
    except KeyError:
        skipped +=1
    except KeyboardInterrupt:
        break

logging.info("skipped " + str(skipped) + " of " + str(len(Y)))

logging.info("X (embedding)  shape: " + str(X.shape))
logging.info("Y (properties) shape: " + str(Y.shape))

fn = "data/X_"+version_Y+"__"+"__".join(emb.keys())+".pkl.bz2"

X.to_pickle(fn)
logging.info("written to: " + fn)
