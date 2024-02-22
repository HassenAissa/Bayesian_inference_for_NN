import PyAce.optimizers as om
import tensorflow as tf
from PyAce.distributions import GaussianPrior
from PyAce.tests.gym_example_1 import test_srlz
from matplotlib import pyplot as plt
import json, os, shutil

connectors = "._-"

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
    return res

def access_file(pref, fs, form, key):
    res = ""
    fn = form[key]
    if fn:
        return pref+fn
    if fs:
        res = pref+fs
        form[key] = fs
    return res

def check_mandatory(form, term, missing):
    if not term:
        return missing
    if isinstance(term,str):
        if form.get(term) != "":
            return missing
        return missing+[term]
    if isinstance(term, list):
        m1 = check_mandatory(form, term[0], missing)
        m2 = check_mandatory(form, term[1:], m1)
        return m2
    if isinstance(term, tuple):
        if term[0] == "or":
            m1 = check_mandatory(form, term[1], missing)
            m2 = check_mandatory(form, term[2], missing)
            if m1 == missing:
                return m1
            return m2
        elif term[0] == "if":
            val = form.get(term[1])
            if val and (not term[2] or val == term[2]):
                return check_mandatory(form, term[3], missing)
            return missing

def read_sessions(scat):
    res = []
    f = open("static/sessions/"+scat+"/db.csv", "r")
    f.readline()
    while True:
        l = f.readline()
        if not l:
            return res
        segs = l.split(',')
        res.append(segs)

def add_sessions(sname:str, scat, desc, envname=""):
    pref = "static/sessions/"+scat
    if not sname:
        sname = "default"
    f = open(pref+"/db.csv", "r")
    lim = int(f.readline())
    entries = []
    found = False
    while True:
        l = f.readline()
        if not l:
            break

        if not found:
            if l.split(',')[0] == sname:
                found = True
                continue
        entries.append(l)
    f.close()

    if len(entries) == lim:
        rem = entries.pop()[0]
        if scat == "sl":
            os.remove(pref+"/"+rem+".json")
        elif scat == "rl":
            shutil.rmtree(pref+"/"+rem)
    entries = [sname+','+envname+','+desc+'\n'] + entries
    f = open(pref+"/db.csv", "w")
    f.write(str(lim)+"\n")
    for e in entries:
        f.write(e)
    f.close()
    return sname

def nn_create(acts, hidden, kernel, filters, ipd=None, n_classes=None):
    # Depending on whether input/output shape is defined, create a template/complete nn model
    layers = []
    acts = find_values(acts)
    activations = []
    ai = 1
    hiddens = [int(h) for h in find_values(hidden)]
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
    print(hiddens, activations, kernel, filters)
    if not kernel or not filters: # fully connected specific
        u = hiddens.pop(0)
        print("layer", u, activations[0])
        if ipd:
            layers.append(tf.keras.layers.Dense(u, activation=activations[0], input_shape=ipd))
        else:
            layers.append(tf.keras.layers.Dense(u, activation=activations[0]))
    else: # Convolutional specific
        filters = [int(f) for f in find_values(filters)]
        kernel = int(kernel)
        u = filters.pop(0)
        print("conv", u, activations[0])
        if ipd:
            layers.append(tf.keras.layers.Conv2D(u, kernel, activation=activations[0]))
        else:
            layers.append(tf.keras.layers.Conv2D(u, kernel, activation=activations[0], input_shape=ipd))
        layers.append(tf.keras.layers.MaxPooling2D(2))
        for f in filters:
            layers += [tf.keras.layers.Conv2D(f, kernel, activation=activations[ai]), tf.keras.layers.MaxPooling2D(2)]
            ai += 1
        layers.append(tf.keras.layers.Flatten())
    # Common to both types
    for h in hiddens:  
        print("layer", h, activations[ai])
        layers.append(tf.keras.layers.Dense(h, activation=activations[ai]))
        ai += 1
    if n_classes:
        print("layer", n_classes, activations[ai])
        layers.append(tf.keras.layers.Dense(n_classes, activation=activations[ai]))
    if layers:
        # nn layers have been added properly
        model = tf.keras.Sequential(layers)
        return model
    return None

def hyp_get(hypf, hyp):
    hyperparams = om.HyperParameters()
    if hyp:
        hyperparams.parse(hyp)
    elif hypf:
        hyperparams.from_file("static/hyperparams/"+hypf)
    return hyperparams

def optim_select(options, fm):
    optim = None
    oname = fm.get("optim")
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
    
    pr1 = [fm.get("pri1m"), fm.get("pri1s")]
    pr2 = [fm.get("pri2m"), fm.get("pri2s")]
    extra = {}
    if "" not in pr1:
        extra["prior"] = GaussianPrior(float(pr1[0], float(pr1[1])))
    if "" not in pr2:
        extra["prior2"] = GaussianPrior(float(pr2[0], float(pr2[1])))
    return optim, extra

def optim_mstart(fm, model_config):
    start_model = fm.get("startjson")
    config = None
    if start_model:
        f = open("static/models/"+start_model,)
        config = json.load(f)
        f.close()
    elif fm.get("mstart"):
        config = model_config
    if config:
        return tf.keras.models.model_from_json(config)
    return None

def optim_dataset(options, fm, hypf, hyp, model_config, dataset):
    optim, extra = optim_select(options, fm)
    hyperparams = hyp_get(hypf, hyp)
    extra["starting_model"] = optim_mstart(fm, model_config)
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

# test_srlz()