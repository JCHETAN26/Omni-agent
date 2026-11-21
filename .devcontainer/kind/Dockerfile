FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm
# The base image's Yarn apt source has an expired signing key, which makes
# `apt-get update` (and with it the docker-in-docker feature install) fail.
RUN rm -f /etc/apt/sources.list.d/yarn.list
# Same non-root user the base image declares for dev sessions.
USER vscode
