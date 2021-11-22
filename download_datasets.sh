#!/usr/bin/env bash

DB_DATA_PATH=data/dbpedia
mkdir -p $DB_DATA_PATH

download_infobox_data() {
    if [ -f $DB_DATA_PATH/infobox-properties_lang=$1.ttl ]; then
        echo "DBpedia info box $1 dataset downloaded"
    else
        echo "DBpedia info box $1 dataset is not found, downloading..."
        cd data/dbpedia
        wget https://databus.dbpedia.org/dbpedia/generic/infobox-properties/2021.09.01/infobox-properties_lang=$1.ttl.bz2
        bzip2 -d infobox-properties_lang=$1.ttl.bz2
        cd ..
    fi
}

# download dbpedia in all relevant languages
for lg in en zh ky es; do
  download_infobox_data $lg
done
