from flask import Flask, render_template, request
from matplotlib import pyplot as plt
import PyAce.datasets.Dataset as ds, PyAce.datasets.utils as dsu
import tensorflow as tf
import os

app = Flask(__name__)

class ModelInfo:
    def __init__(self, ):
        self.dataset:ds.Dataset = None
        self.tfmodel = ""
        
def find_values(text):
    csv = ""
    for c in text:
        if c != ' ':
            csv += c
    return csv.split(',')

info = ModelInfo()
options = [["", "Folder with files and labels", "Single table"], 
               ["", "Regression", "Classification"],
               ["", "Mean squared error", "Cross entropy"],
               ["", "Fully connected", "Convolutional"],
               [""]+os.listdir("PyAce/tests"),
               [""]+["BBB", "FSVI", "HMC", "SLGD", "SWAG"]]
mspecs = ["dcat", "lcat", "loss", "tfrac", "nnjson", "nncat", "ipd"]
tspecs = ["dfolder", "dtable", "batch", "kernel", "filters", "hidden", "activations", "optim"]

@app.route('/reinforce', methods=['GET', 'POST'])    # main page
def reinforce():
    return render_template('reinforce.html')

@app.route('/', methods=['GET', 'POST'])    # main page
def index():
    fm = request.form
    print(fm)
    if not fm:
        return render_template('index.html', options=options)

    elif fm.get("f1"):
        # form no.1 dataset and neural netork category
        inputs = [fm.get(k) for k in mspecs]
        if "" in inputs[:4] or (inputs[4]=="" and "" in inputs[5:]):
            return render_template('index.html', options=options)
        if inputs[4]:
            info.tfmodel = "static/models/"+inputs[4]
        else:
            info.nncat = inputs[5]
            info.ipd = find_values(inputs[6])
        if inputs[2] == options[2][0]:
            info.lfunc = tf.keras.losses.MeanSquaredError()
        elif inputs[2] ==  options[2][1]:
            info.lfunc = tf.keras.losses.SparseCategoricalCrossentropy()
        info.lcat = inputs[1]
        info.tfrac = int(inputs[3])/100
    elif fm.get("f2"):
        # form no.2 neural network config
        inputs = [fm.get(k) for k in tspecs]
        x_train, y_train = None, None
        if inputs[0] == options[0][1]: # images folder and labels
            info.ipd = [int(i) for i in find_values(ic)] # input dimension
            x_train, y_train = dsu.imgdata_preprocess("static/datasets/"+inputs[0], info.tfrac, info.ipd)
            info.n_classes = dsu.get_n_classes(y_train)
            info.dataset.__init__(
                tf.data.Dataset.from_tensor_slices((x_train, y_train)),
                info.lfunc,
                info.lcat
            )
        elif inputs[0] == options[0][1]: # dataframe table
            pass
        if info.nncat == options[3][1]: # fully connected
            hidden = fm.get("hidden")
            if hidden:
                shape[1] = [0] + [int(d) for d in find_values(hidden)] + [info.n_classes]
                activations = fm.get("activations")
                if activations:
                    acts = find_values(activations)
                    activations = []
                    for a in acts:
                        match a:
                            case 'r':
                                activations.append('relu')
                            case's':
                                activations.append('sigmoid')
                            case 't':
                                activations.append('tanh')
                            case _:
                                activations.append('linear')

                    if len(activations) == len(shape[1])-1:
                        info.tfmodel.add(tf.keras.layers.Dense(shape[1][1], activation=activations[0]))
                        for i in range():
                            layers.append(linears[i])
                            
                                
                        base_model = nn.Sequential(*layers)
        
        elif info.nncat == options[3][2]: # Convolutional
            filters = fm.get("filters")
            if filters:
                shape[0] = [0] + [int(f) for f in find_values(filters)]
                kernel = fm.get("kernel")
                if kernel:
                    kernel = int(kernel)
                    padding =  int((kernel - 1) / 2)
                    features = fm.get("features")
                    if features:
                        features = int(features)
                        convs = [nn.Conv2d(shape[0][i], shape[0][i+1], kernel_size=kernel, padding=padding) for i in range(len(shape[0])-1)]
                        f_size = int(32 / (2**(len(shape[0])-1)))
                        ic = shape[0][-1]*f_size*f_size
                        lins = [nn.Linear(ic, features), nn.Linear(features, n_classes)]
                        shape[1] = [ic, features, n_classes]
                        base_model = nn.Sequential(*(convs+lins))
    if base_model:
        draw_nn(shape=shape)
    
    return render_template('index.html', options=options, inputs=inputs, graph=(base_model is not None))

def draw_nn(shape):
    plt.title("Visualize neural network")
    x = 0
    xs = [[],[]]
    ys = [[],[]]
    lines = ("dashed", "solid")
    title = "Fully connected "
    scale = 1
    for s in range(2):
        if s == 1 and shape[0]:
            title = "Convolutional "
            scale = shape[0][-1]/shape[1][0]
        for i in range(len(shape[s])-1):
            xs[s] += [x,x+5]
            ys[s] += [shape[s][i]/2*scale, shape[s][i+1]/2*scale]
            x += 6
           
    plt.title(title+"neural network")
    plt.xticks([])
    plt.yticks([])
    for s in range(2):
        plt.plot(xs[s], ys[s], linestyle=lines[s], c="k")
        plt.plot(xs[s], [-y for y in ys[s]], linestyle=lines[s], c="k")
        for i in range(len(xs[s])):
            plt.plot([xs[s][i],xs[s][i]],[ys[s][i], -ys[s][i]], c="k")
    ymax = max(ys[1])
    if shape[0]:
        ymax = max(ymax, max(ys[0]))
    plt.vlines([0], -ymax, ymax, linestyles="dashed")
    plt.annotate("input layer", (0,-ymax/2))
    plt.savefig("static/drawnn.png")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
