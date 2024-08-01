import os
import cv2
import matplotlib.pyplot as plt

def render_env(num_agents, round_number, bids, agent_list, output_folder):

    plt.figure(figsize=(10, 6))

    # Plot bids
    [plt.plot(i + 1, bids[i], marker='o', label=f"{agent_list[i]} Bid") for i in range(num_agents)]
    
    # Plot settings
    plt.xlabel("Agents")
    plt.ylabel("Bid Value")
    plt.ylim(0, 100)
    plt.title(f"Bids of Each Agent in Round {round_number}")
    plt.legend()
    plt.grid(True)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(output_folder, f"round_{round_number}_bids.png"))
    plt.close()


def create_video_from_pngs(images_folder):

    output_video = "outputs/output_video.mp4"

    # Get sorted list of PNG file paths in the folder by creation date
    image_files = sorted(
        (os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(".png")),
        key=lambda f: os.path.getmtime(f)
    )

    if not image_files:
        print("No PNG images found in the specified folder.")
        return

    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), 0.8, (width, height))

    # Write each image to the video
    for image_file in image_files:
        video.write(cv2.imread(image_file))

    video.release()

    print(f"Video created successfully: {output_video}")