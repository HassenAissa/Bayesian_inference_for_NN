import PyAce.optimizers as om
import tensorflow as tf
from PyAce.distributions import GaussianPrior
from matplotlib import pyplot as plt
import json

connectors = "._-"

def find_values(connectors, text:str):
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
        print(term[0])
        if term[0] == "or":
            return check_mandatory(form, term[1]) or check_mandatory(form, term[2])
        elif term[0] == "if":
            val = form.get(term[1])
            if val and (not term[2] or val == term[2]):
                return check_mandatory(form, term[3])
            return True

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

def nn_create(acts, hidden, kernel, filters, ipd=None, n_classes=None):
    # Depending on whether input/output shape is defined, create a template/complete nn model
    layers = []
    acts = find_values(connectors, acts)
    activations = []
    ai = 1
    hiddens = [h for h in find_values(connectors, hidden)]
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
    
    if not kernel or not filters: # fully connected specific
        if ipd:
            layers.append(tf.keras.layers.Dense(hiddens.pop(0), activation=activations[0], input_shape=ipd))
        else:
            layers.append(tf.keras.layers.Dense(hiddens.pop(0), activation=activations[0]))
    else: # Convolutional specific
        filters = [int(f) for f in find_values(connectors, filters)]
        kernel = int(kernel)
        if ipd:
            layers.append(tf.keras.layers.Conv2D(filters[0], kernel, activation=activations[0]))
        else:
            layers.append(tf.keras.layers.Conv2D(filters[0], kernel, activation=activations[0], input_shape=ipd))
        layers.append(tf.keras.layers.MaxPooling2D(2))
        for f in filters:
            layers += [tf.keras.layers.Conv2D(f, kernel, activation=activations[ai]), tf.keras.layers.MaxPooling2D(2)]
            ai += 1
        layers.append(tf.keras.layers.Flatten())
    # Common to both types
    for h in hiddens:  
        layers.append(tf.keras.layers.Dense(h, activation=activations[ai]))
        ai += 1
    if n_classes:
        layers.append(tf.keras.layers.Dense(n_classes, activation=activations[ai]))
    if layers:
        # nn layers have been added properly
        model = tf.keras.Sequential(layers)
        return model
    return None

def optim_select(oname, options, fm):
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
    hyp, hypf = fm.get("hyp"), fm.get("hypf")
    hyperparams = om.HyperParameters()
    if hyp:
        hyperparams.parse(hyp)
    elif hypf:
        hyperparams.from_file("static/hyperparams/"+hypf)
    pr1 = [fm.get("pri1m"), fm.get("pri1s")]
    pr2 = [fm.get("pri2m"), fm.get("pri2s")]
    start_model = fm.get("startjson")
    extra = {}
    if "" not in pr1:
        extra["prior"] = GaussianPrior(float(pr1[0], float(pr1[1])))
    if "" not in pr2:
        extra["prior2"] = GaussianPrior(float(pr2[0], float(pr2[1])))
    if start_model:
        f = open("static/models/"+start_model,)
        config = json.load(f)
        f.close()
        extra["starting_model"] = tf.keras.models.model_from_json(config)
    return optim, hyperparams, extra

def optim_dataset(oname, options, fm, model_config, dataset):
    optim, hyperparams, extra = optim_select(oname, options, fm)
    if optim:
        optim.compile(hyperparams, model_config, dataset)
        optim.compile_extra_components(extra)
    return optim
    
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

