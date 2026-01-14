from src.clasher.gym_env import ClashRoyaleGymEnv
import gymnasium
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

env = ClashRoyaleGymEnv()
obs, info = env.reset()
logging.info("Environment reset. Starting the game loop.")

# Set up live matplotlib figure
plt.ion()
fig, ax = plt.subplots()

numSteps = 0
while True:
    # Sample a random action
    action = env.action_space.sample() #testing number

    # Take a step
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        logging.info("Game terminated. Exiting the loop.")
        break
    
    #show observation
    image = obs['p1-view']
    logging.info(f"Observation shape: {image.shape}, dtype: {image.dtype}")
    logging.info("Info keys: " + ", ".join(info.keys()))

    # Print basic info for tower entities by sampling the obs at their pixel locations
    logging.info(f"Type map: {env._type_to_id}")
    for eid, ent in env.battle.entities.items():
        name = getattr(getattr(ent, "card_stats", None), "name", ent.__class__.__name__)
        # Filter to towers (Tower / KingTower)
        if "tower" in name.lower():
            px = int((ent.position.x / env.battle.arena.width) * env.obs_shape[1])
            py = int((ent.position.y / env.battle.arena.height) * env.obs_shape[0])
            px = max(0, min(env.obs_shape[1] - 1, px))
            py = max(0, min(env.obs_shape[0] - 1, py))
            pixel = obs['p1-view'][py, px]
            logging.info(f"Entity {eid} {name} player {ent.player_id} pos ({ent.position.x:.1f},{ent.position.y:.1f}) pixel ({px},{py}) obs {pixel}")

    if numSteps % 1000 == 0:
        logging.info(f"Step: {numSteps}, Reward: {reward}, Info: {info}")
        #print observation info - map troop id to name, position, hp_frac
        for entity_meta in info["entities"]:
            type_id = entity_meta["type_id"]
            type_name = entity_meta["type_name"]
            position = entity_meta["position"]
            hp_frac = entity_meta["hp_frac"]
            logging.info(f"Entity Type ID: {type_id}, Name: {type_name}, Position: {position}, HP Fraction: {hp_frac:.2f}\n")
        ax.clear()
        ax.imshow(image)
        ax.axis('off')
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        #print player info - elixir, hand, crowns
        for player in info["players"]:
            elixir = player["elixir"]
            hand = player["hand"]
            crowns = player["crowns"]
            logging.info(f"Player {player['player_id']} - Elixir: {elixir}, Hand: {hand}, Crowns: {crowns}")
        input("Press Enter to continue to the next step...")  # Wait for user input to proceed

    numSteps += 1

env.close()
logging.info("Environment closed.")
