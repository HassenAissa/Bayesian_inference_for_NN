from flask import Flask, render_template, request
from PyAce.datasets.Dataset import Dataset
from PyAce.dynamics.deep_pilco import gym, DynamicsTraining, NNPolicy, BayesianDynamics
from static.rewards import all_rewards
import PyAce.datasets.utils as dsu, utils as apu
import tensorflow as tf
import os, json

sl_mandatory = [("if", "f1", "", ["lcat", ("if", "lcat", "Classification", "dfile"), "ipd", "opd", "mname", "hidden", "activations"]),
    ("if", "f2", "", ["lcat", "loss", ("or", "nnjson", ["ipd", "opd", "mname", "hidden", "activations"]), "dfile", "batch", "optim", "ite"])]
rl_mandatory = ["envname", "hor", "dynep", "npar", "disc", "lep", ("or", "pnnjson", ["pmname", "phidden", "pactivations"]), "poact",
                ("or", "dnnjson", ["dmname", "dhidden", "dactivations"]), "doact", "optim", "reward"]
sl_opkeys = ["lcat", "loss", "optim"]
rl_opkeys = ["rmode","oact","optim", "reward"]
    
class ModelInfo:
    def __init__(self):
        self.form = None
        self.options = {}

    def store(self, fn):
        f = open(fn,"w")
        json.dump(self.form, f)
        f.close()

    def load(self, fn):
        f = open(fn,)
        self.form = json.load(f)
        f.close()

class SLInfo(ModelInfo):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.model_file = ""
        self.model_ready = False
        self.options = {"sessions": [""]+apu.read_sessions("sl"),
            "lcat": ["", "Classification","Regression"],
            "loss": ["", "Cross entropy","Mean squared error"],
            "optim": [""]+["BBB", "FSVI", "HMC", "SGLD", "SWAG"]}
    def find_shapes(self):
        f = open(self.model_file,)
        layers:list = json.load(f)["config"]["layers"]
        f.close()
        self.ipd = layers[0]["config"]["batch_input_shape"]
        self.opd = layers[-1]["config"]["units"]

class RLInfo(ModelInfo):
    def __init__(self):
        super().__init__()
        self.policy = None
        self.dyn_train = None
        self.options = {"sessions": [""]+apu.read_sessions("rl"),
                        "rmode": [""]+["human", "rgb_array"],
                        "oact": [""]+["sigmoid", "relu", "tanh", "softmax", "linear"],
                        "optim": [""]+["BBB", "FSVI", "HMC", "SGLD", "SWAG"],
                        "reward": [""]+list(all_rewards.keys())}

app = Flask(__name__) 
sl = SLInfo()
rl = RLInfo()

@app.route('/reinforce', methods=['GET', 'POST'])    # main page
def reinforce():
    fm = request.form
    options = rl.options
    if not fm or "sessions" in fm or not apu.check_mandatory(fm, rl_mandatory):
        # Clear previous inputs and load main page 
        rl.__init__()
        sessions = fm.get("sessions")
        if sessions:
            rl.load("static/sessions/rl/"+sessions)
            for key in rl_opkeys:
                options[key][0] = rl.form[key]
        return render_template('reinforce.html',options=options, info=rl)
    env = gym.make(fm.get("envname", render_mode=fm.get("rmode")))
    # Policy network
    policy_json = fm.get("pnnjson")
    policy_nn = None
    policy_config = ""
    if policy_json:
        pf = "static/models/policy/"+policy_json
        f = open(pf,"r")
        policy_config = f.read()
        f.close()
        policy_nn = tf.keras.models.model_from_json(policy_config)
    else:
        model_file = "static/models/policy/"+fm.get("pmname")+".json"
        acts,hidden,kernel,filters = fm.get("pactivations"),fm.get("phidden"),fm.get("pkernel"),fm.get("pfilters")
        policy_nn = apu.nn_create(acts,hidden,kernel,filters)
        f = open(model_file, "w")
        policy_config = policy_nn.to_json()
        f.write(policy_config)
        f.close()
    # Dynamics network
    dyn_json = fm.get("dnnjson")
    dyn_nn = None
    dyn_config = ""
    if dyn_json:
        pf = "static/models/dynamics/"+dyn_json
        f = open(pf,"r")
        dyn_config = f.read()
        f.close()
        dyn_nn = tf.keras.models.model_from_json(dyn_config)
    else:
        model_file = "static/models/policy/"+fm.get("dmname")+".json"
        acts,hidden,kernel,filters = fm.get("dactivations"),fm.get("dhidden"),fm.get("dkernel"),fm.get("dfilters")
        dyn_nn = apu.nn_create(acts,hidden,kernel,filters)
        f = open(model_file, "w")
        dyn_config = dyn_nn.to_json()
        f.write(dyn_config)
        f.close()
    policy_hyp = apu.hyp_get(fm.get("phypf"), fm.get("phyp"))
    poact = fm.get("poact")
    policy = NNPolicy(policy_nn, poact, policy_hyp)
    dyn_hyp = apu.hyp_get(fm.get("dhypf"), fm.get("dhyp"))
    doact = fm.get("doact")
    optim, extra = apu.optim_select(fm.get("optim"))
    dyn_training = DynamicsTraining(optim, {"loss":tf.keras.losses.MeanSquaredError(), "likelihood": "Regression"},
        dyn_nn, doact, dyn_hyp)
    extra["starting_model"] = apu.optim_mstart(fm, dyn_config)
    dyn_training.compile_more(extra)
    agent = BayesianDynamics(
        env=env,
        horizon=int(fm.get("hor")),
        dyn_training=dyn_training,
        policy=policy,
        state_reward=fm.get("reward"),
        learn_config=(fm.get("dynep"),fm.get("npar"),fm.get("disc")), # dynamic epochs, particle number, discount factor
    )

    agent.learn(nb_epochs=int(fm.get("lep")))
    return render_template('reinforce.html', options=rl.options, info=rl)

@app.route('/', methods=['GET', 'POST'])    # main page
def index():
    fm = request.form
    print(fm,sl.model_ready)
    options = sl.options
    if not fm or "sessions" in fm or not apu.check_mandatory(fm, sl_mandatory):
        # Clear previous inputs and load main page 
        sl.__init__()
        sessions = fm.get("sessions")
        if sessions:
            sl.load("static/sessions/sl/"+sessions)
            for key in sl_opkeys:
                options[key][0] = sl.form[key]
        return render_template('index.html',options=options, info=sl)
    print("data suff")
    sl.form = fm
    # Both f1: create neural network only and f2: train bayesian model
    model_file = fm.get("nnjson")
    if model_file:
        sl.model_file = "static/models/sl/"+model_file
        sl.model_ready = True
        f = open(sl.model_file, "r")
        model_config = f.read()
        f.close()
        sl.find_shapes()
    else:
        sl.model_file = "static/models/sl/"+fm.get("mname")+".json"
        sl.ipd = [int(d) for d in apu.find_values(fm.get("ipd"))]
        sl.opd = int(fm.get("opd"))
    
    model_config = ""
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
            x_train, y_train = dsu.imgdata_preprocess("static/datasets/"+data_name, tfrac, sl.ipd)
            sl.dataset = Dataset(
                tf.data.Dataset.from_tensor_slices((x_train, y_train)),
                lfunc,
                lcat,
                target_dim=sl.opd
            )
        else: # dataframe table
            sl.dataset = Dataset(
                "static/datasets/"+data_name,
                lfunc,
                lcat,
                target_dim=sl.opd
            )
            y_train = sl.dataset.target_values

    # Create basic nn model if not uploaded
    if not sl.model_ready: 
        n_classes = sl.opd
        if lcat == options["lcat"][1]:
            # classification
            n_classes= dsu.get_n_classes(y_train, dim=sl.opd)
        acts, hidden, kernel, filters = fm.get("activations"),fm.get("hidden"),fm.get("kernel"),fm.get("filters")
        model = apu.nn_create(acts, hidden, kernel, filters, sl.ipd, n_classes)
        if model:
            sl.model_ready = True
            f = open(model_file, "w")
            model_config = model.to_json()
            f.write(model_config)
            f.close()

    if not fm.get("f2"):
        fn = apu.add_sessions(fm.get("sname"), "sl")
        sl.store("static/sessions/sl/"+fn+".json")
    else:
        # Create optimizer and start training
        print("start training", sl.dataset, sl.model_ready)
        oname = fm.get("optim")
        if sl.dataset and sl.model_ready and oname:           
            optim = apu.optim_dataset(oname, options, fm, fm.get("hypf"), fm.get("hyp"),
                                       model_config, sl.dataset)
            fn = apu.add_sessions(fm.get("sname"), "sl")
            sl.store("static/sessions/sl/"+fn+".json")
            optim.train(int(fm.get("ite")))
            
    return render_template('index.html', options=options, info=sl)

@app.route('/settings', methods=['GET', 'POST'])    # main page
def settings():
    fm = request.form
    snum = fm.get("snum")
    if fm.get("session"):
        f = open("static/sessions/"+scat+"/db.csv", "r")
        lines = f.readlines()
        f.close()
        if snum:
            lines[0] = snum+"\n"
        ssel = fm.get("ssel")
        scat = fm.get("scat")
        if ssel and scat:
            ssel = ssel[:-5]
            snew, sdel = fm.get("snew"), fm.get("sdel")
            if snew or sdel:
                i = 1
                while i < len(lines):
                    if lines[i][:-1] == ssel:
                        if snew:
                            lines[i] = snew+"\n"
                            os.rename("static/sessions/"+scat+"/"+ssel+".json",
                                      "static/sessions/"+scat+"/"+snew+".json")
                        elif sdel:
                            lines.pop(i)
                            os.remove("static/sessions/"+scat+"/"+ssel+".json")
                        break
                    else:
                        i += 1
        f = open("static/sessions/"+scat+"/db.csv", "w")
        f.writelines(lines)
        f.close()

    return render_template('settings.html', options=sl.options)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
