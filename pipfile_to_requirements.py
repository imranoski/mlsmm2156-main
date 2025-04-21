import configparser


def main():
    parser = configparser.ConfigParser()
    parser.read("recommender_systems/mlsmm2156-main/Pipfile")

    packages = "packages"
    with open("recommender_systems/mlsmm2156-main/requirements.txt", "w") as f:
        for key in parser[packages]:
            value = parser[packages][key]
            f.write(key + value.replace("\"", "") + "\n")


main()