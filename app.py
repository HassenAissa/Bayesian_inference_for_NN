from flask import Flask, render_template, request
from matplotlib import pyplot as plt
import package.src.datasets.utils as dsu
import package.src.datasets.Dataset as ds
import tensorflow as tf
import os

app = Flask(__name__)
dataset:ds.Dataset = None

def find_values(text):
    csv = ""
    for c in text:
        if c != ' ':
            csv += c
    return csv.split(',')

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


@app.route('/', methods=['GET', 'POST'])    # main page
def index():
    fm = request.form
    print(fm)
    options = [["", "Images", "Table"], 
               ["", "Regression", "Classification"],
               ["", "Mean squared error", "Cross entropy"],
               ["", "Fully connected", "Convolutional"],
               [""]+os.listdir("test")]
    dcat = fm.get("dcat")
    nncat = fm.get("nncat")
    dfolder = fm.get("dataset")
    loss = fm.get("loss")
    lcat = fm.get("lcat")
    tfrac = fm.get("tfrac")
    ic = fm.get("ic")
    shape = [[],[]]
    base_model = None
    if not tfrac:
        tfrac = "0.1"
    inputs = [dcat, nncat, dfolder, loss, tfrac, ic]
    if "" in inputs:
        return render_template('index.html', options=options)
    
    lfunc = None
    if loss == options[2][0]:
        lfunc = tf.keras.losses.MeanSquaredError()
    elif loss ==  options[2][1]:
        lfunc = tf.keras.losses.SparseCategoricalCrossentropy()

    x_train, y_train = None, None
    if dcat == options[0][1]:
        ipd = [int(i) for i in find_values(ic)]
        x_train, y_train = dsu.imgdata_preprocess("test/"+dfolder, float(tfrac), ipd)
        n_classes = dsu.get_n_classes(y_train)
        dataset.__init__(
            tf.data.Dataset.from_tensor_slices((x_train, y_train)),
            lfunc,
            lcat
        )
    elif dcat == options[0][2]:
        pass

    if nncat == options[3][1]:
        hidden = fm.get("hidden")
        if hidden:
            shape[1] = [0] + [int(d) for d in find_values(hidden)] + [n_classes]
            activations = fm.get("activations")
            if activations:
                activations = find_values(activations)
                if len(activations) == len(shape[1])-1:
                    base_model = tf.keras.Sequential()
                    linears = [nn.Linear(shape[1][i], shape[1][i+1]) for i in range(len(shape[1])-1)]
                    layers = []
                    for i in range(len(linears)):
                        layers.append(linears[i])
                        match activations[i]:
                            case 'r':
                                layers.append(nn.ReLU())
                            case's':
                                layers.append(nn.Sigmoid())
                            case 't':
                                layers.append(nn.Tanh())
                    base_model = nn.Sequential(*layers)
    
    elif nncat == options[3][2]:
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
