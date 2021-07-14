import argparse
import openamundsen as oa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config_file', help='configuration file')
    args = parser.parse_args()

    config = oa.read_config(args.config_file)
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
