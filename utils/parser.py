import os
import yaml
from argparse import ArgumentParser


def add_training_options(parser):
    group = parser.add_argument_group('Training options')
    group.add_argument("--batch_size", type=int, required=True, help="size of the batches")
    group.add_argument("--num_epochs", type=int, required=True, help="number of epochs of training")
    group.add_argument("--lr", type=float, required=True, help="AdamW: learning rate")
    group.add_argument("--snapshot", type=int, required=True, help="frequency of saving model/viz")

def add_misc_options(parser):
    group = parser.add_argument_group('Miscellaneous options')
    group.add_argument("--expname", default="exps", help="general directory to this experiments, use it if you don't provide folder name")
    group.add_argument("--folder", help="directory name to save models")
    

def add_cuda_options(parser):
    group = parser.add_argument_group('Cuda options')
    group.add_argument("--cuda", dest='cuda', action='store_true', help="if we want to try to use gpu")
    group.add_argument('--cpu', dest='cuda', action='store_false', help="if we want to use cpu")
    group.set_defaults(cuda=True)

def add_dataset_options(parser):
    group = parser.add_argument_group('Dataset options')
    group.add_argument("--dataset", default='beat_sep_lower', help="Dataset to load")
    group.add_argument("--datapath", help="Path of the data")
    group.add_argument("--num_frames", required=True, type=int, help="number of frames or -1 => whole, -2 => random between min_len and total")
    group.add_argument("--sampling", default="conseq", choices=["conseq", "random_conseq", "random"], help="sampling choices")
    group.add_argument("--sampling_step", default=1, type=int, help="sampling step")
    group.add_argument("--pose_rep", required=True, help="xyz or rotvec etc")

    group.add_argument("--max_len", default=-1, type=int, help="number of frames maximum per sequence or -1")
    group.add_argument("--min_len", default=-1, type=int, help="number of frames minimum per sequence or -1")
    group.add_argument("--num_seq_max", default=-1, type=int, help="number of sequences maximum to load or -1")

    group.add_argument("--glob", dest='glob', action='store_true', help="if we want global rotation")
    group.add_argument('--no-glob', dest='glob', action='store_false', help="if we don't want global rotation")
    group.set_defaults(glob=True)
    group.add_argument("--glob_rot", type=int, nargs="+", default=[3.141592653589793, 0, 0],
                       help="Default rotation, usefull if glob is False")
    group.add_argument("--translation", dest='translation', action='store_true',
                       help="if we want to output translation")
    group.add_argument('--no-translation', dest='translation', action='store_false',
                       help="if we don't want to output translation")
    group.set_defaults(translation=True)

    group.add_argument("--debug", dest='debug', action='store_true', help="if we are in debug mode")
    group.set_defaults(debug=False)

def add_model_options(parser):
    group = parser.add_argument_group('Model options')
    group.add_argument("--modelname", help="Choice of the model, should be like cvae_transformer_rc_rcxyz_kl")
    group.add_argument("--latent_dim", default=256, type=int, help="dimensionality of the latent space")
    group.add_argument("--lambda_kl", required=True, type=float, help="weight of the kl divergence loss")
    group.add_argument("--lambda_rc", default=1.0, type=float, help="weight of the rc divergence loss")
    group.add_argument("--lambda_rcxyz", default=1.0, type=float, help="weight of the rc divergence loss")
    group.add_argument("--jointstype", default="vertices", help="Jointstype for training with xyz")

    group.add_argument('--vertstrans', dest='vertstrans', action='store_true', help="Training with vertex translations in the SMPL mesh")
    group.add_argument('--no-vertstrans', dest='vertstrans', action='store_false', help="Training without vertex translations in the SMPL mesh")
    group.set_defaults(vertstrans=False)

    group.add_argument("--num_layers", default=4, type=int, help="Number of layers for GRU and transformer")
    group.add_argument("--activation", default="gelu", help="Activation for function for the transformer layers")

    # Ablations
    group.add_argument("--ablation", choices=[None, "average_encoder", "zandtime", "time_encoding", "concat_bias"],
                       help="Ablations for the transformer architechture")

def parse_modelname(modelname):
    modeltype, archiname, *losses = modelname.split("_")

    return modeltype, archiname, losses


def construct_checkpointname(parameters, folder):
    implist = [parameters["modelname"],
               parameters["dataset"],
               parameters["extraction_method"],
               parameters["pose_rep"]]
    if parameters["pose_rep"] != "xyz":
        # [True, ""] to be compatible with generate job
        if "glob" in parameters:
            implist.append("glob" if parameters["glob"] in [True, ""] else "noglob")
        else:
            implist.append("noglob")
        if "translation" in parameters:
            implist.append("translation" if parameters["translation"] in [True, ""] else "notranslation")
        else:
            implist.append("notranslation")
            
        if "rcxyz" in parameters["modelname"]:
            implist.append("joinstype_{}".format(parameters["jointstype"]))

    if "num_layers" in parameters:
        implist.append("numlayers_{}".format(parameters["num_layers"]))
            
    for name in ["num_frames", "min_len", "max_len", "num_seq_max"]:
        pvalue = parameters[name]
        pname = name.replace("_", "")
        if pvalue != -1:
            implist.append(f"{pname}_{pvalue}")
    
    if "view" in parameters:
        if parameters["view"] == "frontview":
            implist.append("frontview")

    if "use_z" in parameters:
        if parameters["use_z"] != 0:
            implist.append("usez")
        else:
            implist.append("noz")

    if "vertstrans" in parameters:
        implist.append("vetr" if parameters["vertstrans"] else "novetr")
        
    if "ablation" in parameters:
        abl = parameters["ablation"]
        if abl not in ["", None]:
            implist.append(f"abl_{abl}")
            
    if parameters["num_frames"] != -1:
        implist.append("sampling_{}".format(parameters["sampling"]))
        if parameters["sampling"] == "conseq":
            implist.append("samplingstep_{}".format(parameters["sampling_step"]))
    if "lambda_kl" in parameters:
        implist.append("kl_{:.0e}".format(float(parameters["lambda_kl"])))

    if "activation" in parameters:
        act = parameters["activation"]
        implist.append(act)

    implist.append("bs_{}".format(parameters["batch_size"]))
    implist.append("ldim_{}".format(parameters["latent_dim"]))
    
    checkpoint = "_".join(implist)
    return os.path.join(folder, checkpoint)

def save_args(opt, folder):
    os.makedirs(folder, exist_ok=True)
    
    # Save as yaml
    optpath = os.path.join(folder, "opt.yaml")
    with open(optpath, 'w') as opt_file:
        yaml.dump(opt, opt_file)

def adding_cuda(parameters):
    import torch
    if parameters["cuda"] and torch.cuda.is_available():
        parameters["device"] = torch.device("cuda")
    else:
        parameters["device"] = torch.device("cpu")

def parser():
    parser = ArgumentParser()

    # misc options
    add_misc_options(parser)

    # cuda options
    add_cuda_options(parser)
    
    # training options
    add_training_options(parser)

    # dataset options
    add_dataset_options(parser)

    # model options
    add_model_options(parser)

    opt = parser.parse_args()
    
    # remove None params, and create a dictionnary
    #parameters = {key: val for key, val in vars(opt).items() if val is not None}

    from types import SimpleNamespace

    parameters = SimpleNamespace(**{key: val for key, val in vars(opt).items() if val is not None})


    # parse modelname
    ret = parse_modelname(parameters["modelname"])
    parameters["modeltype"], parameters["archiname"], parameters["losses"] = ret
    
    # update lambdas params
    lambdas = {}
    for loss in parameters["losses"]:
        lambdas[loss] = opt.__getattribute__(f"lambda_{loss}")
    parameters["lambdas"] = lambdas
    
    if "folder" not in parameters:
        parameters["folder"] = construct_checkpointname(parameters, parameters["expname"])

    os.makedirs(parameters["folder"], exist_ok=True)
    save_args(parameters, folder=parameters["folder"])

    adding_cuda(parameters)
    
    return parameters
