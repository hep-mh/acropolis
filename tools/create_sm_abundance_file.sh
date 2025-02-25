#! /usr/bin/env bash

if [ ! -d "alteralterbbn" ]; then
    echo -e "\x1B[38;5;69mCloning AlterAlterBBN...\x1B[0m"

    git clone https://github.com/hep-mh/alteralterbbn.git

    echo
fi


if [ ! -f "alteralterbbn/bin/alteralterbbn" ]; then
    echo -e "\x1B[38;5;69mBuilding AlterAlterBBN...\x1B[0m"

    cd alteralterbbn
    ./build.sh
    cd ..

    echo
fi


echo -e "\x1B[38;5;69mRunning AlterAlterBBN...\x1B[0m"

./alteralterbbn/bin/alteralterbbn data/sm
