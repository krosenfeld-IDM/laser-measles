import subprocess


def test_main():
    assert subprocess.check_output(["measles", "foo", "foobar", "bazar"], text=True) == "foobar\n"


if __name__ == "__main__":
    test_main()
