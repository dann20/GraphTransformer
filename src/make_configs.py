import os
import json
import itertools

def main():
    params = {
                 "data_dir": ["WADI"],
                 "dataset": ["wadi"],
                 "model": ["GraphTransformer"],
                 "n_features": [36],
                 "kernel_size": [7],
                 "feature_embed_dim": [100],
                 "use_gatv2": [True],
                 "alpha": [0.2],
                 "window_size": [100, 200],
                 "batch_size": [128],
                 "lr": [0.001, 0.0001, 0.0005],
                 "num_transformer_stacks": [2, 3],
                 "d_ff": [128],
                 "num_heads": [1],
                 "dropout": [0.1],
                 "num_epochs": [10, 15],
                 "shuffle": [1],
                 "val_split": [0.2],
                 "dataloader_num_workers": [4],
                 "device": ["gpu"],
                 "load_dir": ["default"]
             }

    keys, values = zip(*params.items())
    combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f'GENERATING {len(combs)} NEW CONFIGS ...')

    for comb in combs:
        filename = "{}_{}_{}stacks_{}winsize_{}lr_{}batch_{}epcs".format(comb["model"].upper(),
                                                                         comb["dataset"].upper(),
                                                                         comb["num_transformer_stacks"],
                                                                         comb["window_size"],
                                                                         comb["lr"],
                                                                         comb["batch_size"],
                                                                         comb["num_epochs"]).replace(".", "_")
        config_path = os.path.join("../configs/", "{}.json".format(filename))
        config = {"experiment": filename,
                  "pre_mask": int(2/5 * int(comb["window_size"])),
                  "post_mask": int(3/5 * int(comb["window_size"]))}
        config.update(comb)
        print(filename)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)

    print('DONE.')

if __name__ == '__main__':
    main()
