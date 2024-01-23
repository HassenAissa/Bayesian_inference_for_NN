from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

app = Flask(__name__)

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
            title = "Concolutional "
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
    print(request.form)
    categories = [None, "Fully connected", "Convolutional"]
    category = request.form.get("category")
    shape = [[],[]]
    base_model = None
    input_channels = 100
    n_classes = 2
    if category:
        if category == categories[1]:
            hidden = request.form.get("hidden")
            if hidden:
                shape[1] = [0] + [int(d) for d in find_values(hidden)] + [n_classes]
                activations = request.form.get("activations")
                if activations:
                    activations = find_values(activations)
                    if len(activations) == len(shape[1])-1:
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
        
        elif category == categories[2]:
            filters = request.form.get("filters")
            if filters:
                shape[0] = [0] + [int(f) for f in find_values(filters)]
                kernel = request.form.get("kernel")
                if kernel:
                    kernel = int(kernel)
                    padding =  int((kernel - 1) / 2)
                    features = request.form.get("features")
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
    
    return render_template('index.html', categories=categories, category=category, ic=input_channels, n_classes=n_classes, graph=(base_model is not None))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
