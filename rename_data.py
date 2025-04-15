import shutil
import glob


def main():
    for file in glob.glob("*.json"):
        new_filename = file.replace(" ", "_")
        shutil.move(file, new_filename)


if __name__ == "__main__":
    main()
