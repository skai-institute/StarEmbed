import pandas as pd
import argparse
import datasets


def filter_missing_or_short_lc(datapoint, bands):
    if 'g' in bands:
        if datapoint['bands_data']['g'] is None:
            return False
        if len(datapoint['bands_data']['g']) == 1:
            return False
    if 'r' in bands:
        if datapoint['bands_data']['r'] is None:
            return False
        if len(datapoint['bands_data']['r']) == 1:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bands", type=str, default='gr')
    args = parser.parse_args()

    bands_str = args.bands
    if bands_str == 'gr':
        bands = ['g', 'r']
    else:
        bands = [bands_str]

    # Load dataset
    dataset = datasets.load_from_disk(
        "/projects/p32795/hongyu/hf_csdr1_multiband_raw_lc_subclass_class_str_v2"
    )
    print(f"total number of training lc: {len(dataset['train'])}")
    print(f"total number of validation lc: {len(dataset['validation'])}")
    print(f"total number of test lc: {len(dataset['test'])}")

    dataset_dict = {}
    for split in ['train', 'validation', 'test']:
        print(f"Combining {split} embeddings from {bands} with dataset and applying filtering...")
        dataset_split = dataset[split]

        # Load embeddings for this split
        split_embs = pd.read_csv(
            f"../data/hc_feats_{split}_hf_csdr1_multiband_raw_lc_subclass_class_str_v2_mp.csv",
            index_col=None
        )

        # Extract embeddings for each band
        for band in bands:
            # Create a mask marking which columns are for the specified band
            mask = [col_name[0] == band for col_name in split_embs.columns.values]

            # Grab columns in embedding dataframe for this band
            band_embs = split_embs[split_embs.columns[mask]]

            # Add band embeddings to dataset
            dataset_split = dataset_split.add_column(f"{band}_embedding", band_embs.values.tolist())

        # Apply filtering
        dataset_split = dataset_split.filter(lambda x: filter_missing_or_short_lc(x, bands))
        print(f"total number of remaining {split} lc: {len(dataset_split)}")

        # Store augmented and filtered dataset
        dataset_dict[split] = dataset_split

    # Save the augmented and filtered dataset to disk
    output_path = "/projects/b1094/rehemtulla/SkAI/skai_universal_forecaster/data/embs/" +\
        f"csdr1_raw4_catflags_filtered_embs_trn_val_tst_band{bands_str}"
    print(f"Saving dataset with embeddings to {output_path}")
    dataset_to_save = datasets.DatasetDict(dataset_dict)
    dataset_to_save.save_to_disk(output_path)

    print("Done!")
