# from datasets import load_from_disk

# # 1. Load your on-disk DatasetDict
# ds = load_from_disk(
#     "/projects/p32795/hongyu/hf_csdr1_multiband_raw_lc_subclass_class_str_v2"
# )

# # 2. Push it (you’ll be prompted to log in if needed)
# ds.push_to_hub("wormyu/StarEmbed")

# import requests
# headers = {"Authorization": f"Bearer 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"}
# API_URL = "https://huggingface.co/datasets/123anonymous123/StarEmbed"
# def query():
#     response = requests.get(API_URL, headers=headers)
#     return response.json()
# data = query()

from huggingface_hub import HfApi

api = HfApi()
# api.upload_folder(
#     repo_id="wormyu/StarEmbed",        # your dataset on the Hub
#     repo_type="dataset",
#     folder_path="/projects/p32795/weijian/embs/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr",  # local folder you want to push
#     path_in_repo="embs_hand_crafted_trn_val_tst_bandgr",             # the remote subfolder name you want
#     ignore_patterns="**/cache-*.arrow"  # optional: skip unwanted files
# )

# api.upload_folder(
#     repo_id="wormyu/StarEmbed",        # your dataset on the Hub
#     repo_type="dataset",
#     folder_path="/projects/p32795/dennis/random",  # local folder you want to push
#     path_in_repo="embs_random_trn_val_tst_bandgr",             # the remote subfolder name you want
#     ignore_patterns="**/cache-*.arrow"  # optional: skip unwanted files
# )
# api.upload_folder(
#     repo_id="wormyu/StarEmbed",        # your dataset on the Hub
#     repo_type="dataset",
#     folder_path="/projects/p32795/hongyu/hf_csdr1_multiband_raw4_embeddings_astromer_1_subclass_pad_correct",  # local folder you want to push
#     path_in_repo="embs_astromer_1_trn_val_tst_bandgr",             # the remote subfolder name you want
#     ignore_patterns="**/cache-*.arrow"  # optional: skip unwanted files
# )
# api.upload_folder(
#     repo_id="wormyu/StarEmbed",        # your dataset on the Hub
#     repo_type="dataset",
#     folder_path="/projects/p32795/weijian/embs/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr",  # local folder you want to push
#     path_in_repo="embs_chronos_bolt_tiny_trn_val_tst_bandgr",             # the remote subfolder name you want
#     ignore_patterns=["**/cache-*.arrow", "**/tmp*"]  # optional: skip unwanted files
# )
# api.upload_folder(
#     repo_id="wormyu/StarEmbed",        # your dataset on the Hub
#     repo_type="dataset",
#     folder_path="/projects/p32795/weijian/embs/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr",  # local folder you want to push
#     path_in_repo="embs_moiral_small_trn_val_tst_bandgr",             # the remote subfolder name you want
#     ignore_patterns=["**/cache-*.arrow", "**/tmp*"]  # optional: skip unwanted files
# )
# api.upload_folder(
#     repo_id="wormyu/StarEmbed",        # your dataset on the Hub
#     repo_type="dataset",
#     folder_path="/projects/p32795/weijian/embs/csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr",  # local folder you want to push
#     path_in_repo="embs_chronos_t5_tiny_trn_val_tst_bandgr",             # the remote subfolder name you want
#     ignore_patterns=["**/cache-*.arrow", "**/tmp*"]  # optional: skip unwanted files
# )




# api.upload_folder(
#     repo_id="wormyu/StarEmbed",        # your dataset on the Hub
#     repo_type="dataset",
#     folder_path="/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True",  # local folder you want to push
#     path_in_repo="/embeddings_with_anom/hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True",             # the remote subfolder name you want
#     ignore_patterns=["**/cache-*.arrow", "**/tmp*"]  # optional: skip unwanted files
# )
api.upload_folder(
    repo_id="123anonymous123/StarEmbed",        # your dataset on the Hub
    repo_type="dataset",
    folder_path="/projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr",  # local folder you want to push
    path_in_repo="/embeddings_with_anom/csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr",             # the remote subfolder name you want
    ignore_patterns=["**/cache-*.arrow", "**/tmp*"]  # optional: skip unwanted files
)


# api.upload_folder(
#     repo_id="wormyu/StarEmbed",        # your dataset on the Hub
#     repo_type="dataset
# ",
#     folder_path="/projects/p32795/hongyu/hf_macho_70-10-20",  # local folder you want to push
#     path_in_repo="/macho_raw_light_curve",             # the remote subfolder name you want
#     ignore_patterns=["**/cache-*.arrow", "**/tmp*"]  # optional: skip unwanted files
# )

# from huggingface_hub import HfApi
# api = HfApi()
# api.upload_file(
#     path_or_fileobj="/projects/p32795/hongyu/README.md",
#     path_in_repo="README.md",
#     repo_id="wormyu/StarEmbed",
#     repo_type="dataset",
# )