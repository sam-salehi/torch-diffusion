import time

T = 1000


def get_save_path(epochs: int, num_images, model_path):
    str_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    time_path = f"{model_path}/{num_images}@{str_time}-for-{epochs}.pth"
    return time_path
