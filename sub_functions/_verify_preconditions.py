from ._ischarstr import ischarstr
from ._isrealvector import isrealvector
from ._isrealmatrix import isrealmatrix
import pdb

def verify_preconditions(fun, x0, options):
    r'''
    VERIFY_PRECONDITIONS verifies the preconditions for the input arguments of 
    the function.
    '''

    if not (callable(fun) or ischarstr(fun)):
        raise ValueError("fun should be a function handle.")
    
    if not isrealvector(x0):
        raise ValueError("x0 should be a real vector.")
    
    if "direction_set" in options:
        if not (isrealmatrix(options["direction_set"]) and \
                options["direction_set"].shape[0] == len(x0)) \
                and options["direction_set"].shape[1] == len(x0):
            raise ValueError("direction_set should be a real vector.")

    if "MaxFunctionEvaluations" in options:
             if not (isinstance(options["MaxFunctionEvaluations"], int) \
                        or options["MaxFunctionEvaluations"] <= 0):
                raise ValueError("MaxFunctionEvaluations should be an integer \
                                 scalar.")

    
    if not (isinstance(options["num_blocks"], int) and options["num_blocks"] > 0):
        raise ValueError("options.num_blocks should be an integer scalar.")

    bds_list = ["DS", "CBDS", "PBDS", "RBDS", "PADS", "sCBDS"]
    if "Algorithm" in options:
        if not any(options["Algorithm"].lower() == alg.lower() for alg in bds_list):
            print("Algorithm should be a string in BDS_list")

    if "expand" in options:
        if not (isinstance(options["expand"], float) and options["expand"] >= 1):
            raise ValueError("expand should be a real number greater than or \
                             equal to 1.")

    if "shrink" in options:
        if not (isinstance(options["shrink"], float) and \
                0 <= options["shrink"] < 1):
            raise ValueError("shrink should be a real number in [0, 1).")

    if "reduction_factor" in options:
        if not (isrealvector(options["reduction_factor"]) and \
                options["reduction_factor"].size == 3): 
            raise ValueError("reduction_factor should be a 3-dimensional\
                              real vector.")

        if not (options["reduction_factor"][2] >= options["reduction_factor"][1]\
                 >= options["reduction_factor"][0] >= 0 and \
                options["reduction_factor"][1] > 0):
                raise ValueError("reduction_factor should satisfy 0 <= reduction_factor[0]\
                                 <= reduction_factor[1] <= reduction_factor[2] and\
                                 reduction_factor[1] > 0.")

    if "forcing_function_type" in options:
        if not ischarstr(options["forcing_function_type"]):
            raise ValueError("options.forcing_function_type should be a string.")

    if "alpha_init" in options:
        if not ((isinstance(options["alpha_init"], float) and options["alpha_init"] > 0) \
                or (isrealvector(options["alpha_init"]) and len(options["alpha_init"]) == options["num_blocks"] and all(alpha > 0 for alpha in options["alpha_init"])) \
                or (ischarstr(options["alpha_init"]) and options["alpha_init"].lower() == "auto")):
            raise ValueError("alpha_init should be a real number greater than 0.")

    if "alpha_all" in options:
        if not (isinstance(options["alpha_all"], float) and options["alpha_all"] > 0):
            raise ValueError("alpha_all should be a real number greater than 0.")

    if "StepTolerance" in options:
        if not (isinstance(options["StepTolerance"], float) and options["StepTolerance"] >= 0):
            raise ValueError("StepTolerance should be a real number greater than or equal to 0.")

    if "shuffle_period" in options:
        if not (isinstance(options["shuffle_period"], int) and options["shuffle_period"] > 0):
            raise ValueError("shuffle_period should be a positive integer.")

    if "replacement_delay" in options:
        if not (isinstance(options["replacement_delay"], int) and options["replacement_delay"] >= 0):
            raise ValueError("replacement_delay should be a nonnegative integer integer.")

    if "seed" in options:
        if not (isinstance(options["seed"], int) and options["seed"] > 0):
            raise ValueError("seed should be a positive integer.")

    if "polling_inner" in options:
        if not ischarstr(options["polling_inner"]):
            raise ValueError("polling_inner should be a string.")

    if "cycling_inner" in options:
        if not (isinstance(options["cycling_inner"], int) and 0 <= options["cycling_inner"] <= 4):
            raise ValueError("cycling_inner should be a nonnegative integer less than or equal to 4.")

    if "with_cycling_memory" in options:
        if not isinstance(options["with_cycling_memory"], bool):
            raise ValueError("with_cycling_memory should be a logical value.")

    if "output_xhist" in options:
        if not isinstance(options["output_xhist"], bool):
            raise ValueError("output_xhist should be a logical value.")

    if "output_alpha_hist" in options:
        if not isinstance(options["output_alpha_hist"], bool):
            raise ValueError("output_alpha_hist should be a logical value.")

    if "output_block_hist" in options:
        if not isinstance(options["output_block_hist"], bool):
            raise ValueError("output_block_hist should be a logical value.")
