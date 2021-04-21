#import IPython
#IPython.embed()

import argparse

parser = argparse.ArgumentParser(description='Build recoin-like baseline')
parser.add_argument('--index', '-i', type=argparse.FileType('rb'),  default="data/cleaned_data/indexes_test.npy", help='index')
parser.add_argument('--typestat', '-t', default="data/typestat_20180813.pickle", help='the statistics per type/class')
parser.add_argument('--typestatq5', '-tq5', default="data/typestat_20180813_q5.pickle", help='the statistics per human')
parser.add_argument('--graph', '-g', default="data/wikidata/wikidata-20180813-all.json.bz2.universe.noattr.gt.bz2", help='the wikidata graph to load')
args = parser.parse_args()

version = args.typestat[args.typestat.rfind("/")+10:][:-7]

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

import numpy as np
logging.info("loading: " + args.index.name)
index = np.load(args.index)
version_index = args.index.name[args.index.name.rfind('/')+1:args.index.name.rfind('.')]

import pickle
logging.info("loading: "+args.typestat)
typestat = pickle.load( open(args.typestat, 'rb'))

logging.info("loading: "+args.typestatq5)
typestatq5 = pickle.load( open(args.typestatq5, 'rb'))

from graph_tool.all import *
logging.info("loading: "+args.graph)
universe = load_graph(args.graph)

import operator
import numpy as np
from itertools import islice

logging.info("preparing q2v index...")
q2v = {}
p2v = {}

for v in universe.vertices():
    if universe.vp.item[v]: #items => Q
        q2v[universe.vp.q[v]] = v
    else:                   #property => P
        p2v[universe.vp.q[v]] = v

def vertex_by_qid(q):
    if q in q2v:
        return q2v[q]
    return False

def get_typestat(q):
    temp_propb_t = {}
    tclasses = []
    vertex = vertex_by_qid(q)
    if not vertex:
        return [], []
    else:
        for x in vertex.out_edges():
            if universe.ep.p[x] == 31:
                tclass = universe.vp.q[x.target()] #get class
                try:
                    icount = typestat[tclass][(31,)] #get count of instances
                except KeyError:
                    return [], []
                except TypeError:
                    return [], []
                for z in typestat[tclass]:
                    prop_name = 'P'+str(z[0])
                    if prop_name in temp_propb_t:
                        temp_propb_t[prop_name] = (temp_propb_t[prop_name][0]+(typestat[tclass][z] / icount), temp_propb_t[prop_name][1]+1)
                    else:
                        temp_propb_t[prop_name] = (typestat[tclass][z] / icount, 1)lass == 5:

                    for o in vertex.out_edges():
                        if universe.ep.p[o] == 106:
                            toccupation = universe.vp.q[o.target()]
                            for p in list(filter(lambda x: x[0] == toccupation , typestatq5)):
                                tclasses.append(p)
                                try:
                                    icount = typestatq5[(toccupation, 31)] #get count of instances
                                except KeyError:
                                    return [], []
                                except TypeError:
                                    return [], []
                                prop_name = 'P'+str(p[1])
                                if prop_name in temp_propb_t:
                                    temp_propb_t[prop_name] = (temp_propb_t[prop_name][0]+(typestatq5[p] / icount), temp_propb_t[prop_name][1]+1)
                                else:
                                    temp_propb_t[prop_name] = (typestatq5[p] / icount, 1)


        temp_propb = {}

        for z in temp_propb_t:
            temp_propb[z] = temp_propb_t[z][0] / temp_propb_t[z][1]

        return temp_propb, tclasses



logging.info("collecting")

data = {}
classes = {}

from tqdm import tqdm
for i in tqdm(index):
    try:
        temp_prop_bl, tclasses = get_typestat(i)
        data[i] = temp_prop_bl
        classes[i] = tclasses

    except KeyboardInterrupt:
        break


import pandas as pd
Y_pred = pd.DataFrame(0, index=data.keys(), columns=sorted(list(set([s for p in data.values() for s in p])), key=lambda x: int(x[1:])), dtype='float32')

for Q in data:
    for P in data[Q]:
        Y_pred.loc[Q][P] = data[Q][P]

#import IPython
#IPython.embed()

fn = "data/prediction_RECOIN_"+version+".pkl.bz2"
Y_pred.to_pickle(fn)
logging.info("written to: " + fn)
