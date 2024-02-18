from flask import Flask, render_template, request
from PyAce.datasets.Dataset import Dataset
import PyAce.datasets.utils as dsu, application.utils as apu
import tensorflow as tf, gymnasium as gym
import os, json

sl_mandatory = [("if", "f1", "", ["lcat", ("if", "lcat", "Classification", "dfile"), "ipd", "opd", "mname", "hidden", "activations"]),
    ("if", "f2", "", ["lcat", "loss", ("or", "nnjson", ["ipd", "opd", "mname", "hidden", "activations"]), "dfile", "batch", "optim", "ite"])]
opkeys = ["lcat", "loss", "optim"]
    
class ModelInfo:
    def __init__(self, ):
        self.dataset = None
        self.model_file = ""
        self.model_ready = False
        self.form = None
        self.options = {"sessions": [""]+apu.read_sessions(),
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

app = Flask(__name__) 
info = ModelInfo()

@app.route('/reinforce', methods=['GET', 'POST'])    # main page
def reinforce():
    fm = request.form
    env = gym.make(fm.get("envname", render_mode=fm.get("rmode")))
    return render_template('reinforce.html')

@app.route('/', methods=['GET', 'POST'])    # main page
def index():
    fm = request.form
    print(fm,info.model_ready)
    options = info.options
    if not fm or "sessions" in fm or not apu.check_mandatory(fm, sl_mandatory):
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
    # Both f1: create neural network only and f2: train bayesian model
    model_file = fm.get("nnjson")
    if model_file:
        info.model_file = "static/models/"+model_file
        info.model_ready = True
    else:
        info.model_file = "static/models/"+fm.get("mname")+".json"
        info.ipd = [int(d) for d in apu.find_values(fm.get("ipd"))]
        info.opd = int(fm.get("opd"))
    model_config = ""
    if info.model_ready:
        f = open(info.model_file, "r")
        model_config = f.read()
        f.close()
        info.find_shapes()
    lcat = fm.get("lcat")
    
    x_train, y_train = None, None
    if fm.get("f2") or (fm.get("f1") and lcat==options["lcat"][1]):
        # Requires dataset construction
        loss = fm.get("loss")
        lfunc = None
        if loss == options["loss"][2]:
            lfunc = tf.keras.losses.MeanSquaredError()
        elif loss == options["loss"][1]:
            lfunc = tf.keras.losses.SparseCategoricalCrossentropy()

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

    # Create basic nn model if not uploaded
    if not info.model_ready: 
        n_classes = info.opd
        if lcat == options["lcat"][1]:
            # classification
            n_classes= dsu.get_n_classes(y_train, dim=info.opd)
        model = apu.nn_create(info.ipd, n_classes, fm)
        if model:
            info.model_ready = True
            f = open(model_file, "w")
            model_config = model.to_json()
            f.write(model_config)
            f.close()

    if not fm.get("f2"):
        fn = apu.add_sessions(fm.get("sname"))
        info.store("static/sessions/"+fn+".json")
    else:
        # Create optimizer and start training
        print("start training", info.dataset, info.model_ready)
        oname = fm.get("optim")
        if info.dataset and info.model_ready and oname:           
            optim = apu.optim_dataset(oname, options, fm, model_config, info.dataset)
            fn = apu.add_sessions(fm.get("sname"))
            info.store("static/sessions/"+fn+".json")
            optim.train(int(fm.get("ite")))
            
    return render_template('index.html', options=options, info=info)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
