import gradio as gr
import os
import zipfile
from huggingface_hub import HfApi, HfFolder
from modules import script_callbacks

def zip_folder_and_upload(folder_path, hf_token, repo_id):
    # Ensure token and repository information
    try:
        HfFolder.save_token(hf_token)
        api = HfApi()
    except Exception as e:
        return f"Error saving token: {str(e)}"
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        return "Error: Folder does not exist!"

    base_name = os.path.basename(folder_path.rstrip("/\\"))
    zip_name = f"{base_name}.zip"
    zip_path = os.path.join(os.getcwd(), zip_name)

    try:
        # Zip the folder
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, folder_path)
                    zipf.write(full_path, arcname=relative_path)

        # Upload to Hugging Face
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=zip_name,
            repo_id=repo_id,
            repo_type="dataset"
        )

        # Remove the temporary zip file
        os.remove(zip_path)

        return f"Folder successfully uploaded as {zip_name} to Hugging Face repository: {repo_id}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio UI for the SD WebUI
def on_ui_tabs():
    with gr.Blocks() as folder_uploader_tab:
        gr.Markdown("## Upload Folder to Hugging Face")
        
        with gr.Row():
            folder_input = gr.Textbox(label="Folder Path", placeholder="Enter the full folder path")
            token_input = gr.Textbox(label="Hugging Face Token", placeholder="Enter your HF token", type="password")
            repo_input = gr.Textbox(label="Repository ID", placeholder="username/repository_name")
        
        status_output = gr.Textbox(label="Status", interactive=False)
        
        upload_button = gr.Button("Upload Folder")
        upload_button.click(
            zip_folder_and_upload, 
            inputs=[folder_input, token_input, repo_input], 
            outputs=status_output
        )
        
    return [(folder_uploader_tab, "Folder Uploader", "folder_uploader")]

# Register the extension
script_callbacks.on_ui_tabs(on_ui_tabs)
