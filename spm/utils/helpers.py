import os
import cv2
import matplotlib.pyplot as plt

def render_env(num_agents, round_number, bids, agent_list, output_folder):

    plt.figure(figsize=(10, 6))

    for i in range(num_agents):
        agent_bid = bids[i]
        plt.plot([i + 1], [agent_bid], marker='o', label=f"{agent_list[i]} Bid")
    
    plt.xlabel("Agents")
    plt.ylabel("Bid Value")
    plt.ylim(0, 100)  # Set y-axis boundaries
    plt.title(f"Bids of Each Agent in Round {round_number}")
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, f"round_{round_number}_bids.png")
    plt.savefig(output_path)
    plt.close()

    return

def update_bid (action, bid):

    if action == 0:  # Same bid
        bid += 0
    elif 1 <= action <= 4:  # Higher bid
            bid *= 1.1 + (action - 1) * 0.1
    elif 5 <= action <= 8:  # Lower bid
        bid *= 0.9 - (action - 5) * 0.1

    return bid

def calculate_reward (current_rank, previous_rank, avg_min, bid, initial_bid, round, max_rounds):

    reward = 0
    
    # Reward for improving rank
    if current_rank < previous_rank:
        reward += 1
    # Negative reward for worsening rank
    elif current_rank > previous_rank:
        reward -= 1
    
    # Negative reward for bid outside range
    if not (avg_min < bid < initial_bid):
        reward -= 10
    else:
        reward += 2
        if round == max_rounds:
            if current_rank == 1: 
                reward += 10
            else: 
                reward -= 10        

    return reward

def create_video_from_pngs(images_folder):

    # Output video file name
    output_video = "outputs/output_video.mp4"

    # Get a list of all PNG files in the folder and their creation times
    image_files_with_dates = [(os.path.join(images_folder, file), os.stat(os.path.join(images_folder, file)).st_mtime) for file in os.listdir(images_folder) if file.endswith(".png")]

    # Sort the files based on creation date
    image_files_sorted = sorted(image_files_with_dates, key=lambda x: x[1])

    # Extract file paths from the sorted list
    image_files = [file[0] for file in image_files_sorted]

    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    fps = 0.8  # Decrease the frame rate to increase the delay between frames
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through each image and write it to the video
    for image_file in image_files:
        image = cv2.imread(image_file)
        video.write(image)

    # Release the video object
    video.release()

    print(f"Video created successfully: {output_video}")