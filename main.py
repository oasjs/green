from src.app import App
from src.cli import CommandLineInterface


def main():

    app = App()
    CommandLineInterface(app)


if __name__ == "__main__":
    main()
