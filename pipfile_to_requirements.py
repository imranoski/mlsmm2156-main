import configparser


def main():
    parser = configparser.ConfigParser()
    parser.read("Pipfile")

    packages = "packages"
    with open("requirements.txt", "w") as f:
        for key in parser[packages]:
            value = parser[packages][key]
            f.write(key + "\n")
            if '"*"' in value:
                value.replace('"*"', "")

main()