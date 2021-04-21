import argparse

parser = argparse.ArgumentParser(description='Extract attributes from a GraphTool Binary Graph. (https://github.com/eXascaleInfolab/wd-graph)')
parser.add_argument('--graph', '-g', type=argparse.FileType('r'),  default="data/wikidata/wikidata-20180813-all.json.bz2.universe.gt.bz2", help='the wikidata dump to load (wikidata-*-all.json.bz2)')
args = parser.parse_args()

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from graph_tool.all import *

logging.info("loading: " + args.graph.name)
universe = load_graph(args.graph.name)
version = args.graph.name[args.graph.name.rfind("/")+10:][:8]

def get_property(y):
    if y['datatype'] != 'external-id':
        return y['property']


logging.info("collect properties")

properties = {}
for x in universe.vertices():
    if universe.vp.enwiki[x]: # only if present in wikipedia
        attributes = filter(None, list(map(lambda x : get_property(x), universe.vp.attributes[universe.vertex(x)])))
        relationships = ['P'+str(universe.ep.p[o]) for o in x.out_edges()]
        relationships.extend(attributes)
        properties[universe.vp.q[x]] = relationships
def get_property(y):
    if y['datatype'] != 'external-id':
        return y['property']

all_attributes = sorted(list(set([s for p in properties.values() for s in p])), key=lambda x: int(x[1:]))

import pandas as pd

Y = pd.DataFrame(0, index=properties, columns=all_attributes, dtype="int8")
logging.info("arrange dataframe: " + str(Y.shape))

from tqdm import tqdm

for Q in tqdm(properties):
    for P in properties[Q]:
        Y.at[Q,P] = 1

#Y.sort_index(inplace=True)

Y.to_pickle("data/Y_"+version+".pkl.bz2") 
