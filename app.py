from flask import Flask, render_template, request
from matplotlib import pyplot as plt
import PyAce.datasets.Dataset as ds, PyAce.datasets.utils as dsu
import PyAce.optimizers as om
import tensorflow as tf, gymnasium as gym
import os

app = Flask(__name__)

class ModelInfo:
    def __init__(self, ):
        self.dataset:ds.Dataset = None
        self.model_file = None
        self.model_ready = False
        
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
               [""]+["BBB", "FSVI", "HMC", "SGLD", "SWAG"]]
mspecs = ["dcat", "lcat", "loss", "tfrac", "nnjson", "nncat", "ipd", "opd", "mname"]
tspecs = ["dfolder", "dtable", "batch", "kernel", "filters", "hidden", "activations", "optim", "hyp", "ite"]

@app.route('/reinforce', methods=['GET', 'POST'])    # main page
def reinforce():
    return render_template('reinforce.html')

@app.route('/', methods=['GET', 'POST'])    # main page
def index():
    fm = request.form
    print(fm)
    inputs = []
    if not fm:
        return render_template('index.html', options=options)
    elif fm.get("f1"):
        # form no.1 dataset and neural netork category
        inputs = [fm.get(k) for k in mspecs]
        if "" in inputs[:4] or (inputs[4]=="" and "" in inputs[5:-1]):
            return render_template('index.html', options=options)
        if inputs[4]:
            info.model_file = "static/models/"+inputs[4]
            info.model_ready = True
        else:
            info.model_file = "static/models/"+inputs[8]+".json"
            info.nncat = inputs[5]
            info.ipd = find_values(inputs[6])
            info.opd = int(inputs[7])
        if inputs[2] == options[2][0]:
            info.lfunc = tf.keras.losses.MeanSquaredError()
        elif inputs[2] ==  options[2][1]:
            info.lfunc = tf.keras.losses.SparseCategoricalCrossentropy()
        info.lcat = inputs[1]
        info.tfrac = int(inputs[3])/100
    elif fm.get("f2"):
        # form no.2 neural network config
        inputs = [fm.get(k) for k in tspecs]
        # Create dataset
        x_train, y_train = None, None
        if inputs[0] == options[0][1]: # images folder and labels
            ipd = [int(i) for i in find_values(info.ipd)] # input dimension
            x_train, y_train = dsu.imgdata_preprocess("static/datasets/"+inputs[0], info.tfrac, ipd)
            if info.lcat == options[1][2]:
                # classification
                info.n_classes = dsu.get_n_classes(y_train)
            else:
                info.n_classes = info.opd
            info.dataset.__init__(
                tf.data.Dataset.from_tensor_slices((x_train, y_train)),
                info.lfunc,
                info.lcat,
                target_dim=info.opd
            )
        elif inputs[0] == options[0][1]: # dataframe table
            pass

        # Create basic nn model
        layers = []
        model_config = ""
        if not info.model_ready and inputs[6] and inputs[5]: 
            acts = find_values(inputs[6])
            activations = []
            ai = 1
            hiddens = [h for h in find_values(inputs[5])]
            for a in acts:
                match a:
                    case 'r':
                        activations.append('relu')
                    case'sg':
                        activations.append('sigmoid')
                    case 't':
                        activations.append('tanh')
                    case 'sm':
                        activations.append('softmax')
                    case _:
                        activations.append('linear')

            if info.nncat == options[3][1]: # Fully connected specific
                layers.append(tf.keras.layers.Dense(hiddens.pop(0), activation=activations[0], input_shape=info.ipd))
            
            elif info.nncat == options[3][2]: # Convolutional specific
                if inputs[4] and inputs[3]:
                    filters = [int(f) for f in find_values(inputs[4])]
                    kernel = int(inputs[3])
                    layers += [tf.keras.layers.Conv2D(filters[0], kernel, activation=activations[0], input_shape=info.ipd),
                            tf.keras.layers.MaxPooling2D(2)]
                    for f in filters:
                        layers += [tf.keras.layers.Conv2D(f, kernel, activation=activations[ai]), tf.keras.layers.MaxPooling2D(2)]
                        ai += 1
                    layers.append(tf.keras.layers.Flatten())
            for h in hiddens:   # Common to both types
                layers.append(tf.keras.layers.Dense(h, activation=activations[ai]))
                ai += 1
            if layers:
                # nn layers have been added properly
                model = tf.keras.Sequential(layers)
                model_config= model.to_json()
                info.model_ready = True
                f = open(info.model_file, "w")
                f.write(info.model_config)
                f.close()

        # Start training
        if info.dataset and info.model_ready and inputs[7]:
            if not model_config:
                # model json file is uploaded by user
                f = open(info.model_file, "r")
                model_config = f.read()
                f.close()
            
            optim = None
            oname = inputs[7]
            if oname == options[0][1]:
                optim = om.BBB()
            elif oname == options[0][2]:
                optim = om.FSVI()
            elif oname == options[0][3]:
                optim = om.HMC()
            elif oname == options[0][4]:
                optim = om.SGLD()
            elif oname == options[0][1]:
                optim = om.SWAG()

            hyp = om.HyperParameters()
            if optim and inputs[9]:
                optim.compile(hyp, model_config, info.dataset)
                optim.compile_extra_components()

                optim.train(int(inputs[9]))
                
    return render_template('index.html', options=options, inputs=inputs)

def draw_nn(shape):
    # Auxilliary function, currently not used 
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
