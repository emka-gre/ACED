"""ACED — automated detection of arm-technique errors in bouldering video.

The pipeline is a chain of small, independently testable stages:

    video ─► PoseExtractor ─► FeatureExtractor ─► Windower ─► InferencePipeline ─► AnalysisReport

Each stage consumes frames one at a time rather than a whole video, so batch analysis is a
loop over the same code a future real-time path would use.
"""

__version__ = "0.1.0"
