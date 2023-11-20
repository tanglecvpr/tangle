#----> internal imports
from core.utils.utils_mmssl import get_single_config, process_args, set_determenistic_mode, get_custom_result_dir, check_config_entries
from core.experiment_handler import ExperimentHandler

#----> general imports 
import os
import yaml
import traceback
from datetime import datetime
import pandas as pd

if __name__ == "__main__":
    
    start_time = '%s' % datetime.now().strftime("%d%m_%H%M%S")
    args = process_args()
    
    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)

            config["results_dir"] = args.results_dir

            configs = get_single_config(config=config)

            for config_idx, config in enumerate(configs):
                check_config_entries(config, config_idx=config_idx)

            # loop over all experiments
            results = pd.DataFrame()
            for exp_idx, config in enumerate(configs):

                config = get_custom_result_dir(config)

                os.makedirs(config["results_dir"], exist_ok=True)

                with open(os.path.join(config["results_dir"], 'experiment_config.yaml'), 'w') as file:
                    yaml.dump(config, file)

                set_determenistic_mode(config["seed"])

                print("\n####### Initialize Experiment {} #######".format(exp_idx))
                Experiment = ExperimentHandler(config=config)

                if config["debug"]:
                    result = Experiment.run_experiment()
                    results = pd.concat([results, result], ignore_index=True)
                    
                else:
                    try:
                        result = Experiment.run_experiment()
                        results = pd.concat([results, result], ignore_index=True)
                        
                    except Exception:
                        print('\n!!! Error in Experiment {}!!!'.format(exp_idx))
                        print(traceback.format_exc())
        
        except yaml.YAMLError as exc:
            print(exc)
            
    try:
        results.to_csv(os.path.join(args.results_dir, f"exp_summary_{start_time}.csv"))
    except Exception:
        print('Could not create results csv file.')
