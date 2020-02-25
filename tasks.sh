#!/bin/bash


state=${1:-build}


if [[ $state == "build" ]]; then
    tox -e build
elif [[ $state == "ci" ]]; then
    tox -e ci
elif [[ $state == "dockerbuild" ]]; then
    tox -e dockerbuild
elif [[ $state == "view" ]]; then
    tox -e view
elif [[ $state == "dockerview" ]]; then
    tox -e dockerview
fi
