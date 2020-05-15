import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to cluster info in simulation folders.")

    args = parser.parse_args()

    return args.paths


def main(data_files):
    rows = []
    for file in data_files:
        with open(file) as f:
            content = f.readlines()
            mean_be = content[2].split(":")[1].strip()
            fname = file.split("/")[-3]
            rows.append("{};{}".format(fname, mean_be))
    file_out_cont = "\n".join(rows)
    with open("mean_ie_report.txt", "w") as o:
        o.write(file_out_cont)


if __name__ == "__main__":
    paths = parse_args()
    main(paths)
