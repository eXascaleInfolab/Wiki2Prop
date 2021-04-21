#import IPython
#IPython.embed()

import argparse

parser = argparse.ArgumentParser(description='Extract classes.')
parser.add_argument('--graph', '-g', default="data/wikidata/wikidata-20180813-all.json.bz2.universe.noattr.gt.bz2", help='the wikidata graph to load')
args = parser.parse_args()

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

import pickle

from graph_tool.all import *
logging.info("loading: "+args.graph)
universe = load_graph(args.graph)

import operator
from itertools import islice

logging.info("preparing q2v index...")
q2v = {}
p2v = {}

for v in universe.vertices():
    if universe.vp.item[v]: #items => Q
        q2v[universe.vp.q[v]] = v
    else:                   #property => P
        p2v[universe.vp.q[v]] = v

def get_property(y):
    if y['datatype'] != 'external-id':
        return y['property']

def get_classes(vertex):

    temp_class = [] 

    for x in vertex.out_edges():
        if universe.ep.p[x] == 31 or universe.ep.p[x] == 279:
            tclass = universe.vp.q[x.target()]
            temp_class.append(tclass) #get class

            if tclass == 5:
                for o in vertex.out_edges():
                    if universe.ep.p[o] == 106:
                        temp_class.append(universe.vp.q[o.target()])

    return temp_class

logging.info("collecting")

data = {} 

from tqdm import tqdm
for Q in tqdm(universe.vertices()):
    try:
        temp_classes = get_classes(Q)
        if temp_classes:
            data[universe.vp.q[Q]] = temp_classes

    except KeyboardInterrupt:
        break

pickle.dump(data, open("data/classes.pickle","wb"))
logging.info("written to: data/classes.pickle")

import IPython
IPython.embed()
