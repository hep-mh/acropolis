#! /usr/bin/env bash

# Define a function to replace NE_pd and NT_pd
function replace {
    cp data/params.py.rpl acropolis/params.py
    sed -i "s/__NE_PD__/${1}/" acropolis/params.py
    sed -i "s/__NT_PD__/${2}/" acropolis/params.py
}

# Change the directory
cd ..

# Back up the old param file
cp acropolis/params.py tools/scan_pd/params.py~

# Scan the different values for NE_pd
for NE_PD in $(cat data/NE_pd.list); do
    echo $NE_PD

    replace $NE_PD 50
    echo $NE_PD $($@) >> scan_pd/NE_pd.dat
done

# Scan the different values for NT_pd
for NT_PD in $(cat data/NT_pd.list); do
    echo $NT_PD

    replace 150 $NT_PD
    echo $NT_PD $($@) >> scan_pd/NT_pd.dat
done

# Restore the old param file
mv tools/scan_pd/params.py~ acropolis/params.py

# Go back to the original directory
cd tools/
