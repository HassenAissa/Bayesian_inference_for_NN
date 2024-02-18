from flask import Flask, render_template, request
from matplotlib import pyplot as plt
from PyAce.datasets.Dataset import Dataset
import PyAce.datasets.utils as dsu
import PyAce.optimizers as om
import tensorflow as tf, gymnasium as gym
import os, json

connectors = "-._"
mandatory = ["lcat", "loss", ("nnjson", ["ipd", "opd", "mname", "hidden", "activations"]), "dfile", "batch", "optim", "ite"]
opkeys = ["lcat", "loss", "optim"]
    
class ModelInfo:
    def __init__(self, ):
        self.dataset = None
        self.model_file = ""
        self.model_ready = False
        self.form = None
        self.options = {"sessions": [""]+read_sessions(),
            "lcat": ["", "Classification","Regression"],
            "loss": ["", "Cross entropy","Mean squared error"],
            "optim": [""]+["BBB", "FSVI", "HMC", "SGLD", "SWAG"]}

    def find_shapes(self):
        f = open(self.model_file,)
        layers:list = json.load(f)["config"]["layers"]
        f.close()
        self.ipd = layers[0]["config"]["batch_input_shape"]
        self.opd = layers[-1]["config"]["units"]

    def store(self, fn):
        f = open(fn,"w")
        json.dump(self.form, f)
        f.close()

    def load(self, fn):
        f = open(fn,)
        self.form = json.load(f)
        f.close()

def find_values(text:str):
    res = []
    word = ""
    for c in text:
        if c.isalnum() or c in connectors:
            word += c
        elif word:
            res.append(word)
            word = ""
    if word:
        res.append(word)
    return word

def check_mandatory(form, term):
    if not term:
        print("list end")
        return True
    if isinstance(term,str):
        print("single", form.get(term))
        return form.get(term) != ""
    if isinstance(term, list):
        print("and")
        return check_mandatory(form, term[0]) and check_mandatory(form, term[1:])
    if isinstance(term, tuple):
        print("or")
        return check_mandatory(form, term[0]) or check_mandatory(form, term[1:])

def read_sessions():
    res = []
    f = open("static/sessions/db.csv", "r")
    f.readline()
    while True:
        l = f.readline()
        if not l:
            return res
        res.append(l[:-1]+".json")

def add_sessions(sname:str):
    f = open("static/sessions/db.csv", "r")
    lim = int(f.readline())
    names = []
    found = False
    while True:
        l = f.readline()
        if not l:
            break
        text = l[:-1]
        if not found:
            if text == sname:
                found = True
                continue
        names.append(text)
    f.close()

    if len(names) == lim:
        names.pop()
    if not sname:
        i = 1
        while True:
            sname = str(i)
            if sname not in names:
                break
            i += 1
    names = [sname] + names
    f = open("static/sessions/db.csv", "w")
    f.write(str(lim)+"\n")
    for n in names:
        f.write(n+"\n")
    f.close()
    return sname

app = Flask(__name__) 
info = ModelInfo()

@app.route('/settings', methods=['GET', 'POST'])    # main page
def settings():
    fm = request.form
    snum = fm.get("snum")
    if fm.get("session"):
        f = open("static/sessions/db.csv", "r")
        lines = f.readlines()
        f.close()
        if snum:
            lines[0] = snum+"\n"
        ssel = fm.get("ssel")
        if ssel:
            ssel = ssel[:-5]
            snew, sdel = fm.get("snew"), fm.get("sdel")
            if snew or sdel:
                i = 1
                while i < len(lines):
                    if lines[i][:-1] == ssel:
                        if snew:
                            lines[i] = snew+"\n"
                            os.rename("static/sessions/"+ssel+".json",
                                      "static/sessions/"+snew+".json")
                        elif sdel:
                            lines.pop(i)
                            os.remove("static/sessions/"+ssel+".json")
                        break
                    else:
                        i += 1
        f = open("static/sessions/db.csv", "w")
        f.writelines(lines)
        f.close()
    elif fm.get("model"):
        msel = fm.get("msel")
        if msel:
            mnew, mdel = fm.get("mnew"), fm.get("mdel")
            if mnew:
                os.rename("static/models/"+msel,
                          "static/models/"+mnew+".json")
            elif mdel:
                os.remove("static/models/"+msel)

    return render_template('settings.html', options=info.options)

@app.route('/reinforce', methods=['GET', 'POST'])    # main page
def reinforce():
    return render_template('reinforce.html')

@app.route('/', methods=['GET', 'POST'])    # main page
def index():
    fm = request.form
    print(fm,info.model_ready)
    options = info.options
    if not fm or "sessions" in fm or not check_mandatory(fm, mandatory):
        # Clear previous inputs and load main page 
        info.__init__()
        sessions = fm.get("sessions")
        if sessions:
            info.load("static/sessions/"+sessions)
            for key in opkeys:
                options[key][0] = info.form[key]
        return render_template('index.html',options=options, info=info)
    print("data suff")
    info.form = fm
    # dataset and model setup
    model_file = fm.get("nnjson")
    if model_file:
        info.model_file = "static/models/"+model_file
        info.model_ready = True
    else:
        info.model_file = "static/models/"+fm.get("mname")+".json"
        info.ipd = [int(d) for d in find_values(fm.get("ipd"))]
        info.opd = int(fm.get("opd"))
    loss = fm.get("loss")
    lfunc = None
    if loss == options["loss"][2]:
        lfunc = tf.keras.losses.MeanSquaredError()
    elif loss == options["loss"][1]:
        lfunc = tf.keras.losses.SparseCategoricalCrossentropy()
    lcat = fm.get("lcat")

    # training information
    model_config = ""
    if info.model_ready:
        f = open(info.model_file, "r")
        model_config = f.read()
        f.close()
        info.find_shapes()

    x_train, y_train = None, None
    data_name = fm.get("dfile")
    if "." not in data_name:    # dataset FOLDER
        tfrac = int(fm.get("tfrac"))/100
        x_train, y_train = dsu.imgdata_preprocess("static/datasets/"+data_name, tfrac, info.ipd)
        info.dataset = Dataset(
            tf.data.Dataset.from_tensor_slices((x_train, y_train)),
            lfunc,
            lcat,
            target_dim=info.opd
        )
    else: # dataframe table
        info.dataset = Dataset(
            "static/datasets/"+data_name,
            lfunc,
            lcat,
            target_dim=info.opd
        )
        y_train = info.dataset.target_values

    n_classes = info.opd
    if lcat == options["lcat"][1]:
        # classification
        n_classes= dsu.get_n_classes(y_train, dim=info.opd)

    # Create basic nn model
    layers = []
    if not info.model_ready: 
        acts = find_values(fm.get("activations"))
        activations = []
        ai = 1
        hiddens = [h for h in find_values(fm.get("hidden"))]
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

        if not fm.get("kernel") or not fm.get("filters"): # fully connected specific
            layers.append(tf.keras.layers.Dense(hiddens.pop(0), activation=activations[0], input_shape=info.ipd))
        else: # Convolutional specific
            filters = [int(f) for f in find_values(fm.get("filters"))]
            kernel = int(fm.get("kernel"))
            layers += [tf.keras.layers.Conv2D(filters[0], kernel, activation=activations[0], input_shape=info.ipd),
                    tf.keras.layers.MaxPooling2D(2)]
            for f in filters:
                layers += [tf.keras.layers.Conv2D(f, kernel, activation=activations[ai]), tf.keras.layers.MaxPooling2D(2)]
                ai += 1
            layers.append(tf.keras.layers.Flatten())
        # Common to both types
        for h in hiddens:  
            layers.append(tf.keras.layers.Dense(h, activation=activations[ai]))
            ai += 1
        layers.append(tf.keras.layers.Dense(n_classes, activation=activations[ai]))
        if layers:
            # nn layers have been added properly
            model = tf.keras.Sequential(layers)
            model_config = model.to_json()
            info.model_ready = True
            f = open(info.model_file, "w")
            f.write(model_config)
            f.close()

    # Start training
    print("start training", info.dataset, info.model_ready)
    fn = add_sessions(fm.get("sname"))
    info.store("static/sessions/"+fn+".json")
    oname = fm.get("optim")
    if info.dataset and info.model_ready and oname:           
        optim = None
        if oname == options["optim"][1]:
            optim = om.BBB()
        elif oname == options["optim"][2]:
            optim = om.FSVI()
        elif oname == options["optim"][3]:
            optim = om.HMC()
        elif oname == options["optim"][4]:
            optim = om.SGLD()
        elif oname == options["optim"][5]:
            optim = om.SWAG()

        hyperparams = om.HyperParameters()
        hyp, hypf = fm.get("hyp"), fm.get("hypf")
        if hyp:
            hyp.parse(hyp)
        elif hypf:
            hyp.from_file("static/hyperparams/"+hypf)
        if optim:
            optim.compile(hyperparams, model_config, info.dataset)
            optim.compile_extra_components()

            optim.train(int(fm.get("ite")))
            
    return render_template('index.html', options=options, info=info)

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
