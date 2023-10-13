import json

def  prepare_graph_data(path):

    with open(path) as f:
        d = json.load(f)
    
    d.pop("directed")
    d.pop("multigraph")
    d.pop("graph")
    
    return d
