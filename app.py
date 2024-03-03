from flask import Flask, render_template, request
from static.rewards import all_rewards
from PyAce.datasets.Dataset import Dataset
from PyAce.dynamics.deep_pilco import gym, DynamicsTraining, NNPolicy, BayesianDynamics, complete_model
import PyAce.datasets.utils as dsu, utils as apu
import tensorflow as tf
import os, json, pickle, time

sl_mandatory = [("if", "f1", "", ["lcat", ("if", "lcat", "Classification", "dfile"), "ipd", "opd", "mname", "hidden", "activations"]),
    ("if", "f2", "", ["lcat", "loss", ("or", ("or", "nnjson", "nnjsons"), ["ipd", "opd", "mname", "hidden", "activations"]), ("or", "dfile", "dfiles"), "batch", "optim", "ite"])]
rl_mandatory = ["envname", "hor", "dynep", "npar", "disc", "lep", "rbfu", #("or", ("or", "pnnjson", "pnnjsons"), ["pmname", "phidden", "pactivations"]),
                ("or", ("or","dnnjson", "dnnjsons"), ["dmname", "dhidden", "dactivations"]), "optim", "reward"]
rl_f2mand = ["envname", "session", ("if", "resume", "", ["envname", "lep"])]
sl_opkeys = ["lcat", "loss", "optim"]
rl_opkeys = ["optim", "reward"]
    
class ModelInfo:
    def __init__(self):
        self.sname = ""
        self.form = None
        self.options = {}
        self.notice = []

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
        self.options = {"sessions": [[]]+apu.read_sessions("sl"),
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
        self.envname = ""
        self.agent = None
        self.real_task = False
        self.options = {"sessions": [[]]+apu.read_sessions("rl"),
                        "rmode": [""]+["human", "rgb_array"],
                        "optim": [""]+["BBB", "FSVI", "HMC", "SGLD", "SWAG"],
                        "reward": [""]+list(all_rewards.keys())}
        
    def pause(self, tot_epochs):
        pref = "static/sessions/rl/"+self.sname+"/"
        self.agent.store(pref, tot_epochs)
        apu.store_hyp(self.agent.policy.hyperparams, pref+"policyhyp.json")
        self.agent.policy.hyperparams = None
        # policy config
        f = open("static/models/policy/config.json", "w")
        content = {"ipd": rl.agent.state_d, "opd": rl.agent.action_fd, "oact": rl.agent.policy.oacts[1]}
        content.update(rl.policy_config)
        json.dump(content, f)
        f.close()
        # policy weights
        f = open(pref+"policy_nn.pkl", "wb")
        pickle.dump(self.agent.policy.network.get_weights(), f)
        f.close()
        self.agent.policy.network = None
        f = open(pref+"policy.pkl", "wb")
        pickle.dump(self.agent.policy, f)
        f.close()
        f = open(pref+"dyn_data.pkl", "wb")
        pickle.dump({"features": self.agent.dyn_training.features, 
                     "targets": self.agent.dyn_training.targets}, f)
        f.close()
        apu.store_optim(self.agent.dyn_training.optimizer, pref)

    def restore(self, env):
        pref = "static/sessions/rl/"+self.sname+"/"
        f = open(pref+"policy.pkl", "rb")
        policy = pickle.load(f)
        f.close()
        policy.hyperparams = apu.load_hyp(pref+"policyhyp.json")
        f = open("static/models/policy/config.json", "r")
        config = json.load(f)
        f.close()
        rbfu, rbfscale = config["rbfu"], config["rbfscale"]
        rl.policy_config = {"rbfu": rbfu, "rbfscale": rbfscale}
        policy_template = tf.keras.Sequential()
        policy_template.add(
           tf.keras.layers.experimental.RandomFourierFeatures(output_dim=rbfu, scale=rbfscale)
        ) 
        policy_nn = complete_model(policy_template, config["ipd"], config["opd"], config["oact"])
        f = open(pref+"policy_nn.pkl", "rb")
        policy_weights = pickle.load(f)
        policy_nn.set_weights(policy_weights)
        policy.network = policy_nn
        optim = apu.load_optim(pref)
        f = open(pref+"loss.pkl", "rb")
        loss = pickle.load(f)
        f.close()
        f = open(pref+"agent.json")
        args = json.load(f)
        dyn_training = DynamicsTraining(optim, {"loss": loss, "likelihood": args["likelihood"]})
        f = open(pref+"dyn_data.pkl", "rb")
        dyn_data = pickle.load(f)
        f.close()
        dyn_training.features = dyn_data["features"]
        dyn_training.targets = dyn_data["targets"]
        self.agent = BayesianDynamics(env, args["horizon"], dyn_training, policy, 
                         args["rew_name"], args["learn_config"])
        return args["tot_epochs"]

app = Flask(__name__) 
sl = SLInfo()
rl = RLInfo()

@app.route('/reinforce', methods=['GET', 'POST'])    # main page
def reinforce():
    fm = request.form
    print(fm)
    options = rl.options
    record_file="static/results/learning.txt"
    ep_count = 0
    missing = apu.check_mandatory(fm, rl_f2mand, [])
    print("f2missing", missing)
    if not missing:
        rl.sname = fm.get("session")
        rl.envname, rmode = fm.get("envname"), fm.get("rmode")
        if not rmode:
            rmode = None
        env = gym.make(rl.envname, render_mode=rmode)
        observation, info = env.reset(seed=42)
        ep_count = rl.restore(env)
        if fm.get("resume"): 
            print(">> Learning resumed")
            nb_epochs = int(fm.get("lep"))
            rl.agent.learn(nb_epochs, record_file=record_file)
            rl.pause(ep_count+nb_epochs)
        elif fm.get("render"):
            total_reward = 0
            done = False
            # Run the game loop
            print(">> Start real task")
            f = open("static/sessions/rl/"+rl.sname+"/policy.pkl", "rb")
            policy = rl.agent.policy
            f.close()
            rl.real_task = True
            rewards = []
            states = []
            actions = ([],[])
            while not done:
                states.append(observation)
                a, at = policy.act(tf.reshape(policy.vec_normalize("obs", observation), (1,-1)))
                actions[0].append(a[0])
                actions[1].append(at[0])
                # action = action.numpy()
                observation, reward, terminated, truncated, info = env.step(at[0].numpy())
                total_reward += reward  # Accumulate the reward
                rewards.append(total_reward)
                if terminated or truncated:
                    done = True
                # You can add a delay here if the visualization is too fast
                time.sleep(0.02)
            apu.plot_cart(rewards, states, actions)
            
        return render_template('reinforce.html', options=options, info=rl)
    missing = []
    if fm and "session" not in fm:
        missing = apu.check_mandatory(fm, rl_mandatory, [])
    if not fm or "session" in fm or missing:
        # Clear previous inputs and load main page
        rl.__init__()
        session = fm.get("session")
        if session:
            rl.load("static/sessions/rl/"+session+"/rl.json")
            for key in rl_opkeys:
                options[key][0] = rl.form[key]
        rl.notice = missing
        return render_template('reinforce.html',options=options, info=rl)
    rl.real_task = False
    rl.form = dict(fm)
    rl.envname = fm.get("envname")
    rl.sname = apu.add_sessions(fm.get("sname"), "rl", fm.get("sdesc"), rl.envname)
    if not os.path.exists("static/sessions/rl/"+rl.sname):
        os.mkdir("static/sessions/rl/"+rl.sname)
    rl.store("static/sessions/rl/"+rl.sname+"/rl.json")
    options["sessions"] = [""]+apu.read_sessions("rl")
    env = gym.make(rl.envname)
    # Policy network
    
    # policy_nn = None
    # policy_config = ""
    # pf = apu.access_file("static/models/policy/", fm.get("pnnjsons"), rl.form, "pnnjson")
    # if pf:
    #     f = open(pf,"r")
    #     policy_config = f.read()
    #     f.close()
    #     policy_nn = tf.keras.models.model_from_json(policy_config)
    # else:
    #     model_file = "static/models/policy/"+fm.get("pmname")+".json"
    #     acts,hidden,kernel,filters = fm.get("pactivations"),fm.get("phidden"),fm.get("pkernel"),fm.get("pfilters")
    #     policy_nn = apu.nn_create(acts,hidden,kernel,filters)
    #     f = open(model_file, "w")
    #     policy_config = policy_nn.to_json()
    #     f.write(policy_config)
    #     f.close()
    # Dynamics network
    pf = apu.access_file("static/models/dynamics/", fm.get("dnnjsons"), rl.form, "dnnjson")
    dyn_nn = None
    dyn_config = ""
    if pf:
        f = open(pf,"r")
        dyn_config = f.read()
        f.close()
        dyn_nn = tf.keras.models.model_from_json(dyn_config)
    else:
        model_file = "static/models/dynamics/"+fm.get("dmname")+".json"
        acts,hidden,kernel,filters = fm.get("dactivations"),fm.get("dhidden"),fm.get("dkernel"),fm.get("dfilters")
        dyn_nn = apu.nn_create(acts,hidden,kernel,filters)
        f = open(model_file, "w")
        dyn_config = dyn_nn.to_json()
        f.write(dyn_config)
        f.close()
    policy_hyp = apu.hyp_get(fm.get("phyp"))
    rbfu, rbfscale = int(fm.get("rbfu")), fm.get("rbfscale")
    if rbfscale:
        rbfscale = int(rbfscale)
    else:
        rbfscale = None
    rl.policy_config = {"rbfu": rbfu, "rbfscale": rbfscale}
    policy_template = tf.keras.Sequential()
    policy_template.add(
        tf.keras.layers.experimental.RandomFourierFeatures(output_dim=rbfu, scale=rbfscale)
    ) 
    policy = NNPolicy(policy_template, policy_hyp)
    dyn_hyp = apu.hyp_get(fm.get("dhyp"))
    optim, extra = apu.optim_select(options, fm)
    dyn_training = DynamicsTraining(optim, {"loss":tf.keras.losses.MeanSquaredError(), "likelihood": "Regression"},
        dyn_nn, dyn_hyp)
    rl.agent = BayesianDynamics(
        env=env,
        horizon=int(fm.get("hor")),
        dyn_training=dyn_training,
        policy=policy,
        rew_name=fm.get("reward"),
        learn_config=(int(fm.get("dynep")),int(fm.get("npar")),float(fm.get("disc"))) 
        # dynamic epochs, particle number, discount factor
    )
    if fm.get("mstart"):
        extra["starting_model"] = dyn_training.model
    elif fm.get("start_json"):
        extra["starting_model"] = apu.optim_mstart(fm, dyn_config)
    dyn_training.compile_more(extra)
    nb_epochs = int(fm.get("lep"))
    rl.agent.learn(nb_epochs,record_file=record_file)
    f = open(record_file, "r")
    rl.process =f.read() 
    f.close()
    rl.pause(nb_epochs)
    return render_template('reinforce.html', options=rl.options, info=rl)

@app.route('/', methods=['GET', 'POST'])    # main page
def index():
    fm = request.form
    print(fm,sl.model_ready)
    options = sl.options
    missing = []
    if fm and "session" not in fm:
        missing = apu.check_mandatory(fm, sl_mandatory, [])
    if not fm or "session" in fm or missing:
        # Clear previous inputs and load main page 
        sl.__init__()
        session = fm.get("session")
        if session:
            sl.load("static/sessions/sl/"+session+".json")
            for key in sl_opkeys:
                options[key][0] = sl.form[key]
        sl.notice = missing
        return render_template('index.html',options=options, info=sl)
    print("data suff")
    sl.form = dict(fm)
    fn = apu.add_sessions(fm.get("sname"), "sl", fm.get("sdesc"))
    sl.store("static/sessions/sl/"+fn+".json")
    # Both f1: create neural network only and f2: train bayesian model
    model_file = apu.access_file("static/models/sl/", fm.get("nnjsons"), sl.form, "nnjson")
    if model_file:
        sl.model_file = model_file
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

        data_name = apu.access_file("static/datasets/", fm.get("dfiles"), sl.form, "dfile")
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

    if fm.get("f2"):
        # Create optimizer and start training
        print("start training", sl.dataset, sl.model_ready)
        oname = fm.get("optim")
        if sl.dataset and sl.model_ready and oname:           
            optim = apu.optim_dataset(options, fm, fm.get("hyp"),
                                       model_config, sl.dataset)
            optim.train(int(fm.get("ite")))
            
    return render_template('index.html', options=options, info=sl)

@app.route('/settings', methods=['GET', 'POST'])    # main page
def settings():
    fm = request.form
    print(fm)
    nums = [0,0]
    for c in range(2):
        f = open("static/sessions/"+"sr"[c]+"l/db.csv", "r")
        nums[c] = int(f.readline())
        f.close()
    scat = "s"
    if fm.get("r"):
        scat = "r"
    num = fm.get(scat + "num")
    sel = fm.get(scat+"sel")
    lines = []
    if num or sel:
        f = open("static/sessions/"+scat+"l/db.csv", "r")
        lines = f.readlines()
        f.close()
        if num:
            lines[0] = num+"\n"
        if sel:
            n, d = fm.get(scat+"new"), fm.get(scat+"del")
            if n or d:
                i = 1
                while i < len(lines):
                    if lines[i][:-1] == sel[:-5]:
                        if n:
                            lines[i] = n+"\n"
                            os.rename("static/sessions/"+scat+"l/"+sel,
                                        "static/sessions/"+scat+"l/"+n+".json")
                        elif d:
                            lines.pop(i)
                            os.remove("static/sessions/"+scat+"l/"+sel)
                        break
                    else:
                        i += 1
        f = open("static/sessions/"+scat+"l/db.csv", "w")
        f.writelines(lines)
        f.close()

    return render_template('settings.html', nums=nums,
            ssessions=[""]+apu.read_sessions("sl"), rsessions=[""]+apu.read_sessions("rl"))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
