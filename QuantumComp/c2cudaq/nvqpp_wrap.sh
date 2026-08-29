#!/usr/bin/env bash
# Wrapper around nvq++ that strips CMake dependency-file flags (-MD/-MT/-MF)
# and creates empty .d files so Make's dependency tracking doesn't break.
real=$(command -v nvq++)
filtered=()
depfile=""
i=0
args=("$@")
while [ $i -lt ${#args[@]} ]; do
    arg="${args[$i]}"
    case "$arg" in
        -MD) ;;
        -MT) (( i++ )) ;;
        -MF) (( i++ )); depfile="${args[$i]}" ;;
        *)   filtered+=("$arg") ;;
    esac
    (( i++ ))
done
if [ -n "$depfile" ]; then
    mkdir -p "$(dirname "$depfile")"
    touch "$depfile"
fi
exec "$real" --target nvidia "${filtered[@]}"
