from flask import Flask, render_template, request
from data_preprocessing import prepare_graph_data

app = Flask(__name__)

@app.route("/")
def home():

    graph_count = prepare_graph_data("./graph_communities_count.json")
    graph_corr = prepare_graph_data("./graph_communities_corr.json")

    return render_template('graph.html', graph= graph_corr) # change to "graph = graph_count" for the other graph

if __name__ == "__main__":
    app.run(debug=True)