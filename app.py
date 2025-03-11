import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import io
from PIL import Image
import os

# Define the FurnitureModel class (same as in your Flask app)
class FurnitureModel(nn.Module):
    def __init__(self, input_size=14, output_size=12):  # 2 + 2*6 for input, 2*6 for output
        super(FurnitureModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define furniture pool for reference (default sizes)
furniture_pool = [
    {"name": "Bed", "width": 2, "height": 3},
    {"name": "Table", "width": 2, "height": 2},
    {"name": "Sofa", "width": 3, "height": 2},
    {"name": "Chair", "width": 1, "height": 1},
    {"name": "Wardrobe", "width": 2, "height": 2},
    {"name": "Desk", "width": 2, "height": 1},
    {"name": "Bookshelf", "width": 1, "height": 3},
    {"name": "Lamp", "width": 1, "height": 1},
    {"name": "Dresser", "width": 2, "height": 1},
    {"name": "TV Stand", "width": 3, "height": 1},
]

# Load the PyTorch model
model = FurnitureModel()
try:
    model.load_state_dict(torch.load("furniture_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully from furniture_model.pth.")
except FileNotFoundError:
    print("Warning: furniture_model.pth not found. Using random placement as fallback.")
    model = None

# Check for overlaps between furniture items
def check_overlaps(positions, items):
    for i in range(len(positions)):
        x1, y1 = positions[i]
        w1, h1 = items[i]["width"], items[i]["height"]
        for j in range(i + 1, len(positions)):
            x2, y2 = positions[j]
            w2, h2 = items[j]["width"], items[j]["height"]
            if (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2):
                return True
    return False

# Check if the Bed is near a wall
def check_bed_near_wall(positions, items, room_width, room_height):
    bed_index = next((i for i, item in enumerate(items) if item["name"].lower() == "bed"), None)
    if bed_index is not None:
        bed_pos = positions[bed_index]
        bed_width, bed_height = items[bed_index]["width"], items[bed_index]["height"]
        return (bed_pos[0] == 0 or bed_pos[0] + bed_width == room_width or
                bed_pos[1] == 0 or bed_pos[1] + bed_height == room_height)
    return True

# Adjust positions to avoid overlaps and fit within room
def adjust_positions(positions, items, room_width, room_height):
    adjusted_positions = positions.copy()
    for i in range(len(adjusted_positions)):
        w, h = items[i]["width"], items[i]["height"]
        placed = False
        for attempt in range(200):
            x = (adjusted_positions[i][0] + (attempt % (room_width // 2))) % (room_width - w + 1)
            y = (adjusted_positions[i][1] + (attempt // (room_width // 2))) % (room_height - h + 1)
            adjusted_positions[i] = (x, y)
            has_overlap = False
            for j in range(i):
                x1, y1 = adjusted_positions[j]
                w1, h1 = items[j]["width"], items[j]["height"]
                x2, y2 = adjusted_positions[i]
                w2, h2 = items[i]["width"], items[i]["height"]
                if (x2 < x1 + w1 and x2 + w2 > x1 and
                    y2 < y1 + h1 and y2 + h2 > y1):
                    has_overlap = True
                    break
            if items[i]["name"].lower() == "bed" and not (x == 0 or x + w == room_width or y == 0 or y + h == room_height):
                has_overlap = True
            if not has_overlap:
                placed = True
                break
        if not placed:
            if items[i]["name"].lower() == "bed":
                adjusted_positions[i] = (0, 0)
            else:
                base_x = i * 2 if i * 2 + w <= room_width else room_width - w
                base_y = 0 if i * 2 + h <= room_height else room_height - h
                adjusted_positions[i] = (base_x, base_y)
    return adjusted_positions

# Generate 2D visualization with Matplotlib and return as PIL Image
def plot_layout(positions, items, room_size, title="Furniture Layout"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, room_size[0])
    ax.set_ylim(0, room_size[1])
    colors = {
        "Bed": "coral", "Table": "skyblue", "Sofa": "forestgreen",
        "Chair": "gold", "Wardrobe": "sienna", "Desk": "teal",
        "Bookshelf": "plum", "Lamp": "lightpink", "Dresser": "lightgray",
        "TV Stand": "darkorange"
    }
    for i, (x, y) in enumerate(positions):
        item_name = items[i]["name"]
        width = items[i]["width"]
        height = items[i]["height"]
        edgecolor = "black" if check_overlaps(positions[:i + 1], items[:i + 1]) else "none"
        ax.add_patch(plt.Rectangle((x, y), width, height, facecolor=colors.get(item_name, "gray"), alpha=0.6, label=item_name, edgecolor=edgecolor))
        ax.text(x + width / 2, y + height / 2, item_name, ha="center", va="center", fontsize=8, color="black")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.title(title)

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# Main function to generate the layout
def generate_furniture_layout(room_width, room_height, furniture_inputs):
    try:
        # Validate room dimensions
        room_width = float(room_width)
        room_height = float(room_height)
        if room_width <= 0 or room_height <= 0:
            return "Error: Room dimensions must be positive numbers.", None, None

        # Parse furniture inputs
        items = []
        for furniture in furniture_inputs.split("\n"):
            if not furniture.strip():
                continue
            parts = furniture.split(",")
            if len(parts) != 3:
                return f"Error: Invalid furniture input format: {furniture}. Expected: name,width,height", None, None
            name, width, height = parts
            width = float(width.strip())
            height = float(height.strip())
            if width <= 0 or height <= 0:
                return f"Error: Invalid dimensions for {name}. Width and height must be positive.", None, None
            items.append({"name": name.strip(), "width": width, "height": height})

        if not items:
            return "Error: No furniture items provided.", None, None

        num_items = len(items)
        if num_items > 6:
            return "Error: Maximum 6 furniture items allowed.", None, None

        # Prepare input data for the model
        input_data = [room_width, room_height] + [dim for item in items for dim in [item["width"], item["height"]]]
        input_data += [0] * (14 - len(input_data))

        # Predict positions
        if model is not None:
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predictions = model(input_tensor).numpy().flatten()
            positions = [(int(predictions[i]), int(predictions[i+1])) for i in range(0, len(predictions), 2)][:num_items]
        else:
            positions = []
            for i in range(num_items):
                x = np.random.randint(0, max(1, int(room_width - items[i]["width"])))
                y = np.random.randint(0, max(1, int(room_height - items[i]["height"])))
                positions.append((x, y))

        # Adjust positions
        adjusted_positions = adjust_positions(positions, items, room_width, room_height)

        # Validate constraints
        has_overlap = check_overlaps(adjusted_positions, items)
        bed_near_wall = check_bed_near_wall(adjusted_positions, items, room_width, room_height)
        if not has_overlap and (bed_near_wall or "Bed" not in [item["name"].lower() for item in items]):
            status = "Layout generated successfully."
            title = f"Layout for {room_width}x{room_height}"
        else:
            status = "Layout invalid: Overlaps detected or Bed not near wall."
            title = f"Invalid Layout for {room_width}x{room_height}"

        # Generate visualization
        room_size = (room_width, room_height)
        image = plot_layout(adjusted_positions, items, room_size, title=title)

        # Prepare position output
        positions_str = "\n".join([f"{items[i]['name']}: ({x}, {y})" for i, (x, y) in enumerate(adjusted_positions)])

        return status, positions_str, image

    except Exception as e:
        return f"Error: {str(e)}", None, None

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Furniture Layout Generator")
    gr.Markdown("Enter room dimensions and furniture details to generate a layout. Format for furniture: `name,width,height` (one per line).")
    
    with gr.Row():
        room_width = gr.Textbox(label="Room Width", value="10")
        room_height = gr.Textbox(label="Room Height", value="8")
    
    furniture_inputs = gr.Textbox(
        label="Furniture (name,width,height per line)",
        value="Bed,2,3\nTable,2,2\nSofa,3,2\nChair,1,1\nWardrobe,2,2",
        lines=6
    )
    
    submit_button = gr.Button("Generate Layout")
    
    status_output = gr.Textbox(label="Status")
    positions_output = gr.Textbox(label="Positions")
    layout_image = gr.Image(label="Furniture Layout")

    submit_button.click(
        fn=generate_furniture_layout,
        inputs=[room_width, room_height, furniture_inputs],
        outputs=[status_output, positions_output, layout_image]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()