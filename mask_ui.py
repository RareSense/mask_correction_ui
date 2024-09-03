import json
import gradio as gr
from PIL import Image
import os
import uuid
import numpy as np

# Define paths for the master JSON files
json_file_paths = {
    'set1': 'C:/Users/Dell/Desktop/work/mask_ui/master1.json',
    'set2': 'C:/Users/Dell/Desktop/work/mask_ui/master2.json'
}

# Define paths for the correct mask folders
correct_mask_folders = {
    'set1': 'C:/Users/Dell/Desktop/work/mask_ui/correct_mask1/',
    'set2': 'C:/Users/Dell/Desktop/work/mask_ui/correct_mask2/'
}

# Define paths for the correct JSON files
correct_json_paths = {
    'set1': 'C:/Users/Dell/Desktop/work/mask_ui/correct1.json',
    'set2': 'C:/Users/Dell/Desktop/work/mask_ui/correct2.json'
}

def get_json_filepath(req: gr.Request):
    query_params = req.query_params
    selected_set = query_params.get('set', 'set2')  # Default to 'set2' if not provided
    main_json_filepath = json_file_paths.get(selected_set, json_file_paths['set1'])

    progress_json_filename = os.path.splitext(main_json_filepath)[0] + '_progress.json'
    output_wrong_json_filename = os.path.splitext(main_json_filepath)[0] + '_output_wrong_segments.json'
    correct_mask_folder = correct_mask_folders.get(selected_set)
    correct_json_filename = correct_json_paths.get(selected_set)

    data = []
    with open(main_json_filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    current_index = load_progress(progress_json_filename)
    return data, current_index, progress_json_filename, output_wrong_json_filename, correct_mask_folder, correct_json_filename

def load_progress(progress_json_filename):
    try:
        with open(progress_json_filename, 'r') as progress_file:
            progress_data = json.load(progress_file)
            return progress_data.get('current_index', 0)
    except FileNotFoundError:
        return 0

def save_progress(index, progress_json_filename):
    progress_data = {
        'current_index': index
    }
    with open(progress_json_filename, 'w') as progress_file:
        json.dump(progress_data, progress_file)

def load_image(data, index):
    if index < len(data):
        target_image_path = data[index]['target']
        mask_image_path = data[index]['mask']
        source_image_path = data[index].get('source', None)
        ai_name = data[index]["ai_name"]

        target_image = Image.open(target_image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("RGB")
        source_image = Image.open(source_image_path).convert("RGB") if source_image_path else None

        return target_image, mask_image, source_image, ai_name
    return None, None, None, None

def update_image(data, current_index):
    if current_index >= len(data):
        return None, None, None, "Finished! All images processed.", current_index, None

    target_image, mask_image, source_image, ai_name = load_image(data, current_index)
    progress_text = f"Current Index: {current_index + 1}/{len(data)}"
    
    return target_image, mask_image, source_image, progress_text, current_index, ai_name

def delete_item(data, current_index, progress_json_filename, output_wrong_json_filename):
    if current_index < len(data):
        item_to_delete = data.pop(current_index)
        
        with open(output_wrong_json_filename, 'a') as output_wrong_file:
            json.dump(item_to_delete, output_wrong_file)
            output_wrong_file.write('\n')
        
        with open(progress_json_filename.replace('_progress.json', '.json'), 'w') as master_file:
            for item in data:
                json.dump(item, master_file)
                master_file.write('\n')
        
        save_progress(current_index, progress_json_filename)
        
        if current_index >= len(data):
            current_index = len(data) - 1
        
        return update_image(data, current_index)
    
    return None, None, None, "Finished! All images processed.", current_index, None

def save_mask(data, current_index, mask_image_dict, progress_json_filename, correct_mask_folder, correct_json_filename):
    if current_index < len(data):
        mask_image = mask_image_dict.get("composite", None)
        if mask_image is None:
            raise ValueError("The 'composite' key is missing from the mask image dictionary.")
        
        if isinstance(mask_image, np.ndarray):
            mask_image = Image.fromarray(mask_image)

        if mask_image.mode == 'RGBA':
            mask_image = mask_image.convert('RGB')
        
        if not os.path.exists(correct_mask_folder):
            os.makedirs(correct_mask_folder)
        mask_filename = f"{uuid.uuid4()}_human_image1_mask.jpg"
        mask_save_path = os.path.join(correct_mask_folder, mask_filename)
        mask_image.save(mask_save_path)
        data[current_index]['mask'] = mask_save_path
        item_to_save = data.pop(current_index)
        with open(correct_json_filename, 'a') as correct_file:
            json.dump(item_to_save, correct_file)
            correct_file.write('\n')
        with open(progress_json_filename.replace('_progress.json', '.json'), 'w') as master_file:
            for item in data:
                json.dump(item, master_file)
                master_file.write('\n')

        save_progress(current_index, progress_json_filename)
        
        if current_index >= len(data):
            current_index = len(data) - 1
        
        return update_image(data, current_index)
    
    return None, None, None, "Finished! All images processed.", current_index, None

# Function to superimpose mask on the target image with 50% opacity
def superimpose_images(target_img, mask_img, original_mask_state):
    # If mask_img is a dictionary, extract the composite image
    if isinstance(mask_img, dict):
        mask_img = mask_img.get("composite", None)
        if mask_img is None:
            raise ValueError("The 'composite' key is missing from the mask image dictionary.")
    
    # Convert target_img from numpy array to PIL image if necessary
    if isinstance(target_img, np.ndarray):
        target_img = Image.fromarray(target_img)
    
    # Convert images to RGBA if necessary
    target_img = target_img.convert("RGBA")
    mask_img = mask_img.convert("RGBA")
    # Store the original mask image in the state
    original_mask_state = np.array(mask_img)
    # Resize mask to match target image size
    mask_img = mask_img.resize(target_img.size)
    # Blend the images with 50% opacity for the mask
    blended_img = Image.blend(target_img, mask_img, alpha=0.5)
    
    return np.array(blended_img), original_mask_state

def disable_superimpose(original_mask_state, mask_display):
    # Extract the 'composite' key from the mask_display dictionary, which contains the annotated image
    annotated_mask = mask_display.get("composite", None)

    if annotated_mask is None:
        raise ValueError("The 'composite' key is missing from the mask image dictionary.")

    # Convert the annotated mask to an image if necessary
    if isinstance(annotated_mask, np.ndarray):
        annotated_mask = Image.fromarray(annotated_mask)

    # Convert the original mask state from NumPy array to Image, if necessary
    if isinstance(original_mask_state, np.ndarray):
        original_mask_state = Image.fromarray(original_mask_state)

    # Ensure both images are in RGBA mode to handle transparency
    annotated_mask = annotated_mask.convert("RGBA")
    original_mask_state = original_mask_state.convert("RGBA")

    # Create a transparent image to extract the brush strokes
    brush_strokes = Image.new("RGBA", annotated_mask.size, (0, 0, 0, 0))

    # Use the alpha channel of the annotated mask to isolate the brush strokes
    brush_strokes.paste(annotated_mask, (0, 0), mask=annotated_mask.split()[3])

    # Save the brush strokes as a separate image
    brush_strokes_path = "brush_strokes.png"
    brush_strokes.save(brush_strokes_path)

    # Convert the original mask back to opaque (RGB)
    opaque_mask = original_mask_state.convert("RGB")

    # Overlay the brush strokes on the opaque mask
    final_mask = Image.alpha_composite(opaque_mask.convert("RGBA"), brush_strokes)

    # Convert the final image back to RGB to remove any remaining transparency
    final_mask = final_mask.convert("RGB")

    return np.array(final_mask)

def start_interface(req: gr.Request):
    data, current_index, progress_json_filename, output_wrong_json_filename, correct_mask_folder, correct_json_filename = get_json_filepath(req)
    target_image, mask_image, source_image, progress_text, current_index, ai_name = update_image(data, current_index)

    return target_image, mask_image, source_image, progress_text, data, current_index, progress_json_filename, output_wrong_json_filename, correct_mask_folder, correct_json_filename, ai_name

def update_brush_color(color_name):
    return gr.ImageEditor(brush=gr.Brush(colors=[color_options[color_name]]))

# Gradio Interface
with gr.Blocks() as block:
    color_options = {
        "Yellow": "#fee725",
        "Purple": "#450057"
    }

    color_picker = gr.Dropdown(
        label="Select Brush Color",
        choices=list(color_options.keys()),
        value="Yellow"
    )

    with gr.Row():
        target_display = gr.Image(interactive=False, label="Target Image")
        mask_display = gr.ImageEditor(type="pil", interactive=True, label="Editable Mask", brush=gr.Brush(colors=["#fee725"]))
        source_display = gr.Image(interactive=False, label="Source Image")

    with gr.Row():
        name = gr.Textbox(interactive=False, label="ai_name")
        progress_text = gr.Textbox(interactive=False, label="Progress")
        next_button = gr.Button(value="Next")
        delete_button = gr.Button(value="Delete")
        save_mask_button = gr.Button(value="Save Mask")
        superimpose_button = gr.Button(value="Superimpose")
        disable_superimpose_button = gr.Button(value="Disable Superimpose")

    original_mask_state = gr.State()
    data_state = gr.State()
    index_state = gr.State()
    progress_state = gr.State()
    output_wrong_state = gr.State()
    correct_mask_state = gr.State()
    correct_json_state = gr.State()

    block.load(
        fn=start_interface,
        inputs=None,
        outputs=[target_display, mask_display, source_display, progress_text, data_state, index_state, progress_state, output_wrong_state, correct_mask_state, correct_json_state, name]
    )
    color_picker.change(fn=update_brush_color, inputs=color_picker, outputs=mask_display)

    superimpose_button.click(
        fn=superimpose_images,
        inputs=[target_display, mask_display, original_mask_state],
        outputs=[mask_display, original_mask_state]
    )
    disable_superimpose_button.click(
        fn=disable_superimpose,
        inputs=[original_mask_state, mask_display],
        outputs=mask_display
    )

    next_button.click(
        fn=update_image,
        inputs=[data_state, index_state],
        outputs=[target_display, mask_display, source_display, progress_text, index_state, name]
    )

    delete_button.click(
        fn=delete_item,
        inputs=[data_state, index_state, progress_state, output_wrong_state],
        outputs=[target_display, mask_display, source_display, progress_text, index_state, name]
    )

    save_mask_button.click(
        fn=save_mask,
        inputs=[data_state, index_state, mask_display, progress_state, correct_mask_state, correct_json_state],
        outputs=[target_display, mask_display, source_display, progress_text, index_state, name]
    )

block.launch(server_name="127.0.0.1", server_port=8000)
