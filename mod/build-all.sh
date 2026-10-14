#!/usr/bin/env bash
# Build the PyMC Playtime Recorder for every Minecraft version in versions.json.
#
#   ./build-all.sh                 # all versions -> dist/pymcplaytime-<mod>+<mc>.jar
#   ./build-all.sh 1.21.1 26.3     # only these
#
# Needs JDK 25 on PATH (or JAVA_HOME). Gradle and Minecraft are downloaded automatically.
# A version that fails does not stop the others; the summary at the end lists it.
set -uo pipefail
cd "$(dirname "$0")"

if [ "$#" -gt 0 ]; then
	versions=("$@")
else
	# Plain grep/sed so the script needs nothing but bash.
	mapfile -t versions < <(grep -o '"minecraft": *"[^"]*"' versions.json | sed 's/.*"\([^"]*\)"$/\1/')
fi

chmod +x gradlew
mkdir -p dist
built=()
failed=()
for mc in "${versions[@]}"; do
	echo
	echo "==> Minecraft ${mc}"
	if ./gradlew build -Pmc="${mc}" --console=plain; then
		jar=$(ls build/libs/pymcplaytime-*+"${mc}".jar 2>/dev/null | head -n 1)
		if [ -n "${jar}" ]; then
			cp "${jar}" dist/
			built+=("$(basename "${jar}")")
		else
			failed+=("${mc} (no jar produced)")
		fi
	else
		failed+=("${mc}")
	fi
done

echo
echo "Built ${#built[@]} jar(s) into $(pwd)/dist:"
for jar in "${built[@]}"; do echo "  ${jar}"; done
if [ "${#failed[@]}" -gt 0 ]; then
	echo "Failed: ${failed[*]}"
	exit 1
fi
