#!/bin/bash

dir=$1

for zipfile in "$dir"/*.zip; do
  if [ -f "$zipfile" ]; then
    echo "Unzipping $zipfile..."
    ./7zip/7zz e "$zipfile" -o"$dir"
  fi
done

echo "Unzipping complete."
