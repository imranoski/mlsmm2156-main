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

# _Originally posted by @BeyondEvil in [#3493](https://github.com/pypa/pipenv/issues/3493#issuecomment-581400046)_