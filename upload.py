
from huggingface_hub import HfApi, login

login()  # Will ask for your token

api = HfApi()

api.upload_folder(
    folder_path='cnn_s1_saved',
    path_in_repo='cnn_s1_saved',
    repo_id='your hf spaec name ',
    repo_type='space'
)
api.upload_folder(
    folder_path='tft_s1_saved',
    path_in_repo='tft_s1_saved',
    repo_id='your hf spaec name ',
    repo_type='space'
)
api.upload_folder(
    folder_path='cnn_s2_saved',
    path_in_repo='cnn_s2_saved',
    repo_id='your hf spaec name ',
    repo_type='space'
)
print('Done!')

