import os
import cv2
import matplotlib.pyplot as plt
import json

def render_env(num_agents, round_number, bids, agent_list, output_folder):

    plt.figure(figsize=(10, 6))
    lowest_bid = min(bids)
    lowest_bid_agent = agent_list[bids.index(lowest_bid)]
    print(f"Round Number: {round_number}")
    store_round_winner(round_number, lowest_bid, lowest_bid_agent)
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

####### WRITE THE WINNER OF EACH ROUND INSIDE THE winner_agents.json #######
def store_round_winner(round_number, bid, agent):

    winner_agent = {"round": round_number, "agent": f"{agent}", "bid": bid}
    winner_file = "outputs/winner_agents.json"
    
    if round_number == 1:
        with open(winner_file, 'w') as file:
        # Write the new data directly as a list with one dictionary
            json.dump([winner_agent], file, indent=4)
        print(winner_agent)
    else:
        try:
            with open(winner_file, 'r') as file:
                winners_data = json.load(file)
                
        except FileNotFoundError:
        # If the file doesn't exist, start with an empty list
            winners_data = []

        winners_data.append(winner_agent)
        with open(winner_file, 'w') as file:
            json.dump(winners_data, file, indent=4)
            print(winners_data)
