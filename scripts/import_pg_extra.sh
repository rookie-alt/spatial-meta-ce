#!/bin/sh
# Import the two remaining target databases for doc 110 L3 (imdb, genome).
# The dataset scripts do `drop database if exists` + `create database` + \copy
# with client-relative CSV paths, so psql must run from the dataset directory.
set -e
PSQL=/opt/pgtest/bin/psql
SOCK=/var/run/pgsock
PORT=5456
for db in genome imdb; do
  echo "=== import $db ==="
  su postgres -c "cd /var/lib/ds/datasets/$db && $PSQL -q -h $SOCK -p $PORT -U postgres -d postgres -f postgres_create_$db.sql" 2>&1 | grep -v "could not change directory" || true
  echo "=== analyze $db ==="
  su postgres -c "cd /tmp && $PSQL -q -h $SOCK -p $PORT -U postgres -d $db -c 'analyze;'" 2>&1 | grep -v "could not change directory" || true
  su postgres -c "cd /tmp && $PSQL -q -h $SOCK -p $PORT -U postgres -d postgres -c \"alter database $db set session_preload_libraries='pg_hint_plan';\"" 2>&1 | grep -v "could not change directory" || true
  echo "=== done $db ==="
done
echo "=== IMPORT DONE ==="
su postgres -c "cd /tmp && $PSQL -h $SOCK -p $PORT -U postgres -d postgres -Atc \"select datname, pg_database_size(datname)/1048576 from pg_database order by 1\"" 2>&1 | grep -v "could not change directory"
