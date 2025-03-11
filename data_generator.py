import numpy as np
import rectpack
import pandas as pd
import random
import os

# Define a larger pool of furniture items
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

# Generate optimized dataset with variable room sizes
def generate_optimal_dataset(num_samples, min_room_size=8, max_room_size=12):
    data = []
    attempts = 0
    max_attempts = num_samples * 20  # Prevents infinite loops

    print(f"Starting dataset generation for {num_samples} samples...")
    
    while len(data) < num_samples and attempts < max_attempts:
        room_width = random.randint(min_room_size, max_room_size)
        room_height = random.randint(min_room_size, max_room_size)
        num_items = random.randint(3, 6)  # Choose 3-6 furniture items
        selected_items = random.sample(furniture_pool, num_items)
        items = [(f["width"], f["height"]) for f in selected_items]

        # Randomly rotate some items
        if random.random() > 0.5:
            items = [(h, w) if random.random() > 0.5 else (w, h) for w, h in items]

        try:
            packer = rectpack.newPacker(rotation=True)
            for i, (w, h) in enumerate(items):
                packer.add_rect(w, h, i)
            packer.add_bin(room_width, room_height)
            packer.pack()

            if len(packer[0]) == num_items:  # Ensure all items fit
                positions = [(rect.x, rect.y) for rect in packer[0]]
                rotated_items = [(rect.width, rect.height) for rect in packer[0]]

                # Constraint: Ensure Bed is near a wall if present
                bed_index = next((i for i, item in enumerate(selected_items) if item["name"] == "Bed"), None)
                is_near_wall = True
                if bed_index is not None:
                    bed_pos = positions[bed_index]
                    bed_width, bed_height = rotated_items[bed_index]
                    is_near_wall = (bed_pos[0] == 0 or bed_pos[0] + bed_width == room_width or
                                   bed_pos[1] == 0 or bed_pos[1] + bed_height == room_height)

                # Check for overlaps
                has_overlap = False
                for i in range(len(positions)):
                    x1, y1 = positions[i]
                    w1, h1 = rotated_items[i]
                    for j in range(i + 1, len(positions)):
                        x2, y2 = positions[j]
                        w2, h2 = rotated_items[j]
                        if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
                            has_overlap = True
                            break
                    if has_overlap:
                        break
                
                if is_near_wall and not has_overlap:
                    data.append({
                        "room_size": (room_width, room_height),
                        "items": rotated_items,
                        "positions": positions,
                        "selected_items": [item["name"] for item in selected_items]
                    })
                    print(f"✅ Added sample {len(data)} / {num_samples} | Room: {room_width}x{room_height}")
            else:
                print("⚠ Packing failed: Not all items fit")
        except Exception as e:
            print(f"⚠ Error in packing: {e}")

        attempts += 1
    
    print(f"✅ Finished: Generated {len(data)} valid samples after {attempts} attempts.")
    return data

# Prepare data for CSV storage
def prepare_data_for_csv(dataset):
    X, y = [], []
    
    max_items = max(len(entry["items"]) for entry in dataset)  # Find max items per room
    
    for entry in dataset:
        room_size = entry["room_size"]
        items = entry["items"]
        positions = entry["positions"]
        
        # Input: [room_width, room_height, item1_width, item1_height, ..., padded 0s]
        input_data = [room_size[0], room_size[1]]
        for item in items:
            input_data.extend(item)  # width, height
        
        # Pad with zeros to match max_items
        while len(input_data) < 2 + (max_items * 2):  
            input_data.append(0)

        # Output: [item1_x, item1_y, ..., padded 0s]
        output_data = []
        for pos in positions:
            output_data.extend(pos)
        
        while len(output_data) < (max_items * 2):  
            output_data.append(0)

        X.append(input_data)
        y.append(output_data)
    
    return np.array(X), np.array(y)



if __name__ == "__main__":
    # Generate dataset
    dataset = generate_optimal_dataset(500)  # Generate 500 samples
    if not dataset:
        print("❌ Failed to generate dataset. Check `rectpack` installation or increase `max_attempts`.")
        exit(1)

    print(f"✅ Successfully generated {len(dataset)} samples.")

    # Prepare and save data to CSV
    try:
        print("📂 Preparing data for CSV storage...")
        X, y = prepare_data_for_csv(dataset)
        df = pd.DataFrame({
            "input": [x.tolist() for x in X],
            "output": [y.tolist() for y in y],
            "selected_items": [entry["selected_items"] for entry in dataset]
        })

        csv_path = "furniture_dataset.csv"
        print(f"💾 Saving dataset to `{csv_path}`...")
        df.to_csv(csv_path, index=False)

        if os.path.exists(csv_path):
            print(f"✅ Dataset saved successfully: {csv_path}")
        else:
            print(f"❌ Error: CSV file `{csv_path}` was not created.")
    except Exception as e:
        print(f"❌ Error while writing CSV: {e}")
        exit(1)
