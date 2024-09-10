import json
import gradio as gr
from PIL import Image
import os
import uuid
import numpy as np

#Define paths for the master JSON files
json_file_paths = {
    'set1': '/home/nimra/viton_segmentation_mask_checker/master1.json',
    'set2': '/home/nimra/viton_segmentation_mask_checker/master2.json'
}

# Define paths for the correct mask folders
correct_mask_folders = {
    'set1': '/home/nimra/viton_segmentation_mask_checker/correct_mask1/',
    'set2': '/home/nimra/viton_segmentation_mask_checker/correct_mask2/'
}

# Define paths for the correct JSON files
correct_json_paths = {
    'set1': '/home/nimra/viton_segmentation_mask_checker/correct1.json',
    'set2': '/home/nimra/viton_segmentation_mask_checker/correct2.json'
}


def get_json_filepath(req: gr.Request):
    query_params = req.query_params
    selected_set = query_params.get('set', 'set1')  # Default to 'set2' if not provided
    main_json_filepath = json_file_paths.get(selected_set, json_file_paths['set1'])

    progress_json_filename = os.path.splitext(main_json_filepath)[0] + '_progress.json'
    print(progress_json_filename)
    output_wrong_json_filename = os.path.splitext(main_json_filepath)[0] + '_output_wrong_segments.json'
    correct_mask_folder = correct_mask_folders.get(selected_set)
    correct_json_filename = correct_json_paths.get(selected_set)

    data = []
    with open(main_json_filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Only process non-empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {line}, Error: {e}")  
    

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
        mask_image_path = os.path.join('mask/',data[index]['mask'])
        source_image_path = data[index].get('source', None)
        ai_name = data[index]["ai_name"]
        if not os.path.exists(target_image_path) or not os.path.exists(mask_image_path) or (source_image_path and not os.path.exists(source_image_path)):
            return None, None, None, None

        target_image = Image.open(target_image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("RGB")
        source_image = Image.open(source_image_path).convert("RGB") if source_image_path else None

        return target_image, mask_image, source_image, ai_name
    return None, None, None, None


def update_image(data, current_index, progress_json_filename, output_wrong_json_filename):
    if current_index >= len(data):
        return None, None, None, "Finished! All images processed.", current_index, None

    target_image, mask_image, source_image, ai_name = load_image(data, current_index)

    # If any path is missing (i.e., one of the images is None)
    if target_image is None or mask_image is None or (source_image is None and 'source' in data[current_index]):
        # Move the item to the wrong segments file
        item_to_delete = data.pop(current_index)
        with open(output_wrong_json_filename, 'a') as output_wrong_file:
            json.dump(item_to_delete, output_wrong_file)
            output_wrong_file.write('\n')

        # Save the updated data back to the master JSON file
        with open(progress_json_filename.replace('_progress.json', '.json'), 'w') as master_file:
            for item in data:
                json.dump(item, master_file)
                master_file.write('\n')

        # Save progress after deleting the item
        save_progress(current_index, progress_json_filename)

        # Adjust the current index if needed
        if current_index >= len(data):
            current_index = len(data) - 1

        # Try to load the next image after deleting the current one
        return update_image(data, current_index, progress_json_filename, output_wrong_json_filename)

    # Proceed if all images are valid
    progress_text = f"Current Index: {current_index + 1}/{len(data)}"
    save_progress(current_index, progress_json_filename)

    return target_image, mask_image, source_image, progress_text, current_index, ai_name




def delete_item(data, current_index, progress_json_filename, output_wrong_json_filename):
    if current_index < len(data):
        item_to_delete = data.pop(current_index)
        
        # Save the deleted item in the wrong segments file
        with open(output_wrong_json_filename, 'a') as output_wrong_file:
            json.dump(item_to_delete, output_wrong_file)
            output_wrong_file.write('\n')
        
        # Save the updated data back to the master JSON file
        with open(progress_json_filename.replace('_progress.json', '.json'), 'w') as master_file:
            for item in data:
                json.dump(item, master_file)
                master_file.write('\n')
        
        # Save progress
        save_progress(current_index, progress_json_filename)
        
        # Adjust current_index if needed
        if current_index >= len(data):
            current_index = len(data) - 1
        
        return update_image(data, current_index, progress_json_filename, output_wrong_json_filename)
    
    return None, None, None, "Finished! All images processed.", current_index, None


def save_mask(data, current_index, mask_image_dict, progress_json_filename, correct_mask_folder, correct_json_filename, superimpose_enabled,output_wrong_json_filename):
    if superimpose_enabled:
        return None, None, None, "Cannot save while superimpose is enabled.", current_index, None
    else:
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
            # Use the current mask filename from the data
            mask_filename = os.path.basename(data[current_index]['mask']) 
            mask_save_path = os.path.join(correct_mask_folder, mask_filename)
            mask_image.save(mask_save_path)
            data[current_index]['mask'] = mask_filename
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
            return update_image(data, current_index, progress_json_filename,output_wrong_json_filename)

        
        return None, None, None, "Finished! All images processed.", current_index, None



def superimpose_images(target_img, mask_img, original_mask_state):
    
    if isinstance(mask_img, dict):
        mask_img = mask_img.get("composite", None)
        if mask_img is None:
            raise ValueError("The 'composite' key is missing from the mask image dictionary.")
    
    # Convert target_img from numpy array to PIL image
    if isinstance(target_img, np.ndarray):
        target_img = Image.fromarray(target_img)

    target_img = target_img.convert("RGBA")
    mask_img = mask_img.convert("RGBA")
    original_mask_state = np.array(mask_img)
    mask_img = mask_img.resize(target_img.size)
    # Blend the images with 50% opacity for the mask
    blended_img = Image.blend(target_img, mask_img, alpha=0.5)
    return np.array(blended_img), original_mask_state, True

def disable_superimpose(original_mask_state, mask_display):
    # Extract the 'composite' key from the mask_display dictionary, which contains the annotated image
    annotated_mask = mask_display.get("composite", None)

    if annotated_mask is None:
        raise ValueError("The 'composite' key is missing from the mask image dictionary.")

    # Convert the annotated mask to an image if necessary
    if isinstance(annotated_mask, np.ndarray):
        annotated_mask = Image.fromarray(annotated_mask)
        

    # Ensure the annotated mask is in RGBA mode to handle transparency
    annotated_mask = annotated_mask.convert("RGBA")
  

    # Create an empty (transparent) image for brush strokes
    brush_strokes = Image.new("RGBA", annotated_mask.size, (0, 0, 0, 0))

    # Extract the RGBA data of the annotated mask
    datas = annotated_mask.getdata()
    yellow = (254, 231, 37, 255)  
    purple = (69, 0, 87, 255)     

    new_data = []
    for item in datas:
        # If the pixel matches yellow or purple, keep it; otherwise, make it transparent
        if item == yellow or item == purple:
            new_data.append(item)
        else:
            new_data.append((0, 0, 0, 0))  # Transparent

    # Update brush_strokes image with the filtered data
    brush_strokes.putdata(new_data)

    # Convert the original mask state from NumPy array to Image, if necessary
    if isinstance(original_mask_state, np.ndarray):
        original_mask_state = Image.fromarray(original_mask_state)

    # Ensure the original mask is in RGBA mode to handle transparency
    original_mask_state = original_mask_state.convert("RGBA")

    # Overlay the brush strokes on the original mask
    final_mask = Image.alpha_composite(original_mask_state, brush_strokes)

    # Convert the final image back to RGB to remove any remaining transparency
    final_mask = final_mask.convert("RGB")

    # Return the final mask and the filename of the saved brush strokes
    return np.array(final_mask), False




def start_interface(req: gr.Request):
    data, current_index, progress_json_filename, output_wrong_json_filename, correct_mask_folder, correct_json_filename = get_json_filepath(req)
    
    target_image, mask_image, source_image, progress_text, current_index, ai_name = update_image(data, current_index, progress_json_filename, output_wrong_json_filename)

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
    superimpose_enabled = gr.State()


    
    
    color_picker.change(fn=update_brush_color, inputs=color_picker, outputs=mask_display)
    block.load(
        fn=start_interface,
        inputs=None,
        outputs=[target_display, mask_display, source_display, progress_text, data_state, index_state, progress_state, output_wrong_state,  correct_mask_state, correct_json_state, name
    ]
)


    superimpose_button.click(
    fn=superimpose_images,
    inputs=[target_display, mask_display, original_mask_state],
    outputs=[mask_display, original_mask_state, superimpose_enabled]
    )

    disable_superimpose_button.click(
        fn=disable_superimpose,
        inputs=[original_mask_state, mask_display],
        outputs=[mask_display, superimpose_enabled]
    )

    save_mask_button.click(
        fn=save_mask,
        inputs=[data_state, index_state, mask_display, progress_state, correct_mask_state, correct_json_state, superimpose_enabled],
        outputs=[target_display, mask_display, source_display, progress_text, index_state, name]
    )


    #)
    next_button.click(
        fn=lambda data, index, progress_file, output_wrong_json_filename: update_image(
        data, index + 1, progress_file, output_wrong_json_filename
        ),
        inputs=[data_state, index_state, progress_state, output_wrong_state], 
        outputs=[target_display, mask_display, source_display, progress_text, index_state, name]
)





    delete_button.click(
        fn=delete_item,
        inputs=[data_state, index_state, progress_state, output_wrong_state],
        outputs=[target_display, mask_display, source_display, progress_text, index_state, name]
    )

block.launch(server_name="0.0.0.0", server_port=8006)
