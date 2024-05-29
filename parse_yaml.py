import yaml
import sys


def parse_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)


def main(file_path):
    config = parse_yaml(file_path)

    # Print the values for the Makefile to capture
    for key, value in config.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"{key.upper()}_{subkey.upper()}={subvalue}")
        else:
            print(f"{key.upper()}={value}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_yaml.py <path_to_yaml>")
        sys.exit(1)
    file_path = sys.argv[1]
    main(file_path)
