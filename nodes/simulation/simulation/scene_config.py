from enum import Enum

class Scene(str, Enum):
    grid = "grid"
    hospital_staircase = "hospital_staircase"
    rail_blocks = "rail_blocks"
    stone_stairs = "stone_stairs"
    generated_rail = "generated_rail"
    generated_pyramid = "generated_pyramid"
    excavator = "excavator"
    obstacle_park = "obstacle_park"