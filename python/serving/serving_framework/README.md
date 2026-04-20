# Serving framework

This directory contains a Python library that simplifies the creation of custom
prediction containers for Vertex AI. It provides the framework for building HTTP
servers that meet the platform's
[requirements](https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements).

To implement a model-specific HTTP server, frame the custom data handling and
orchestration logic within this framework.
