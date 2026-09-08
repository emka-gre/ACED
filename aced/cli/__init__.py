"""The `aced` command-line interface.

A thin shell around `aced.analysis.analyze`: parse arguments, choose a bundle, serialise the
report, map failures onto exit codes. No analysis logic lives here, because the API worker
has to run exactly the same thing without going through argparse.
"""
