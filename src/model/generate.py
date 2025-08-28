from generate_audio import generate as gen_audio
from generate_image import generate as gen_image


def generate():
    gen_image()
    gen_audio()


if __name__ == "__main__":
    generate()
