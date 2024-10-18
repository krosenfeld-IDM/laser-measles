import subprocess


def test_main():
    assert subprocess.check_output(["laser-measles", "foo", "foobar"], text=True) == "foobar\n"
