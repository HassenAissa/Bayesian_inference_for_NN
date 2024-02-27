import PyAce.optimizers as om
import tensorflow as tf, numpy as np
from PyAce.distributions import GaussianPrior
from PyAce.optimizers.hyperparameters import HyperParameters
from matplotlib import pyplot as plt
import json, os, shutil, pickle

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
        if form.get(term):
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
    hyperparams = HyperParameters()
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

def store_hyp(hyp, fn):
    f = open(fn, "w")
    json.dump(hyp._params, f) 
    f.close()

def load_hyp(fn):
    f = open(fn,)
    res = HyperParameters(**json.load(f))  
    f.close()
    return res

def store_optim(optim, pref):
    for k in ["_dataset", "_training_dataset", "_data_iterator", "_dataloader"]:
        setattr(optim, k, None)
    store_hyp(optim._hyperparameters, pref+"dynhyp.json")
    optim._hyperparameters = None
    f = open(pref+"dyn.pkl", "wb")
    pickle.dump(optim, f)
    f.close()
    print("pickle finish")

def load_optim(pref):
    f = open(pref+"dyn.pkl", "rb")
    optim = pickle.load(f)
    f.close()
    optim._hyperparameters = load_hyp(pref+"dynhyp.json")
    print("load pickle finish")
    print(optim._frequency, optim._k)
    return optim

def plot_task(rewards, states, actions):
    pref = "static/results/"
    ts = range(len(rewards))
    plt.title("Rewards over time")

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time step')
    ax1.set_ylabel('height/displacement')
    ax1.scatter(ts, [state[1] for state in states], color='b')
    ax1.scatter(ts, [state[0] for state in states], color='r')
    ax1.plot(ts, actions)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('speeds')
    ax2.scatter(ts, [state[4] for state in states], color='c')
    plt.savefig(pref+"record.png")

    plt.clf()
    plt.title("Reward over time")
    plt.plot(ts, rewards)
    plt.savefig(pref+"reward.png")

# def store_optim(optim, pref):
#     arrays = []
#     def check_arrtype(a):
#         if isinstance(a, np.ndarray):
#             return "array"
#         if isinstance(a, tf.Tensor):
#             return "tensor"
#         return ""
#     def array_collector(value):
#         if isinstance(value, list):
#             t = check_arrtype(value[0])
#             if t:
#                 for i in range(len(value)):
#                     arrays.append(value[i])
#                     value[i] = t+str(len(arrays))
#                 return True
#             return array_collector(value[0])
#         if isinstance(value, dict):
#             i = 0
#             kvps = list(value.items())
#             while i < len(kvps):
#                 k,v = kvps[i]
#                 t = check_arrtype(value[0])
#                 if t:
#                     arrays.append(v)
#                     value[k] = t+str(len(arrays))
#                     i += 1
#                 else:
#                     return all(array_collector(v) for k,v in kvps[i:])
#             return True

#     attrs = optim.__dict__
#     collectors = []
#     for k, v in attrs.items():
#         t = check_arrtype(v)
#         if t:
#             arrays.append(v)
#             setattr(optim, k, t+str(len(arrays)))
#             collectors.append(k)
#         elif array_collector(v):
#             setattr(optim, k, v)
#             collectors.append(k)

#     array_cnt = 0
#     while array_cnt < len(arrays):
#         a = arrays[array_cnt]
#         if isinstance(a, np.ndarray):
#             np.save(pref+"array"+str(array_cnt+1)+".npy", a)
#         elif isinstance(a, tf.Tensor):
#             np.save(pref+"tensor"+str(array_cnt+1)+".npy", a.numpy())
#         array_cnt += 1
#     print(optim.__dict__)
#     f = open(pref+"dyn.pkl", "wb")
#     pickle.dump(optim, f)
#     f.close()
            
    # value = {"a0": {"a1":np.array([1,2])}, "a2":np.array([3,4])}
    # print(array_collector(value), value, arrays)


# from PyAce.tests.gym_example_1 import test_srlz
# test_srlz()
print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
# print(tf.version.VERSION)