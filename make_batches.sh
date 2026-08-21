#!/bin/bash

#setting source and destination directories. Change NC or CC or muon neutrino folder
SOURCE_DIR="/data/user/akatil/electron_neutrino/for_real/dataset_complete/reconstruction/muon_neutrino_time"

DESTINATION_DIR="/data/user/akatil/electron_neutrino/for_real/dataset_complete/batches/muon_neutrino"

#initialize counter
folder_count=0
file_count=0

for file in "$SOURCE_DIR"/*; do
    
    #creating a new folder for every 100 files
    if ((file_count % 100 == 0)); then
        folder_name="$DESTINATION_DIR/batch_$(printf $folder_count)"
        mkdir -p "$folder_name"
        ((folder_count++))
    fi

    mv "$file" "$folder_name/"
    ((file_count++))
done

echo "total files processed is $file_count"
echo "Files are organized"
