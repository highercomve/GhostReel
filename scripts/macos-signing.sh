#!/usr/bin/env bash
# Reusable self-signed macOS identity, following Oriel's signing workflow.
# create <directory>: any host with OpenSSL; stores the .p12 and password (0600).
# import: macOS CI; reads MACOS_CERTIFICATE (base64) and MACOS_CERT_PASSWORD.
# cleanup: removes only this job's temporary keychain and certificate files.
set -euo pipefail
umask 077

case "${1:-}" in
  create)
    out="${2:?Usage: scripts/macos-signing.sh create <directory>}"
    mkdir -p "$out"
    for name in ghostreel-codesign.p12 ghostreel-codesign.password; do
      [ ! -e "$out/$name" ] || { echo "Refusing to replace $out/$name" >&2; exit 1; }
    done
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    cat > "$tmp/openssl.cnf" <<'CONF'
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = GhostReel Code Signing
[ext]
basicConstraints = critical, CA:false
keyUsage = critical, digitalSignature
extendedKeyUsage = critical, codeSigning
subjectKeyIdentifier = hash
CONF
    openssl req -x509 -newkey rsa:3072 -nodes -sha256 -days 3650 \
      -config "$tmp/openssl.cnf" -keyout "$tmp/key.pem" -out "$tmp/cert.pem" 2>/dev/null
    openssl rand -hex 32 | tr -d '\n' > "$tmp/password"
    # macOS security import supports this PKCS#12 encryption on all supported versions.
    openssl pkcs12 -export -inkey "$tmp/key.pem" -in "$tmp/cert.pem" \
      -name 'GhostReel Code Signing' -keypbe PBE-SHA1-3DES -certpbe PBE-SHA1-3DES \
      -macalg sha1 -passout "file:$tmp/password" -out "$tmp/certificate.p12"
    cp "$tmp/certificate.p12" "$out/ghostreel-codesign.p12"
    cp "$tmp/password" "$out/ghostreel-codesign.password"
    openssl x509 -in "$tmp/cert.pem" -noout -fingerprint -sha1
    echo "Created certificate and password in $out. Keep and reuse them for every release."
    ;;
  import)
    [ "$(uname -s)" = Darwin ] || { echo 'Import requires macOS.' >&2; exit 1; }
    : "${RUNNER_TEMP:?Import is intended for CI; RUNNER_TEMP is required}"
    : "${GITHUB_ENV:?GITHUB_ENV is required}"
    : "${MACOS_CERTIFICATE:?Set MACOS_CERTIFICATE to the base64-encoded .p12}"
    : "${MACOS_CERT_PASSWORD:?Set MACOS_CERT_PASSWORD to the .p12 password}"
    tmp="$RUNNER_TEMP/ghostreel-signing"
    mkdir -p "$tmp"
    keychain="$tmp/build.keychain-db"
    [ ! -e "$keychain" ] || { echo "Refusing to replace $keychain" >&2; exit 1; }
    printf '%s' "$MACOS_CERTIFICATE" | openssl base64 -d -A > "$tmp/certificate.p12"
    # Extract only the public certificate to identify and trust this self-signed identity.
    openssl pkcs12 -in "$tmp/certificate.p12" -nokeys -clcerts \
      -passin env:MACOS_CERT_PASSWORD -out "$tmp/cert.pem"
    identity=$(openssl x509 -in "$tmp/cert.pem" -noout -fingerprint -sha1 | cut -d= -f2 | tr -d ':')
    keychain_password=$(openssl rand -hex 32)
    security list-keychains -d user > "$tmp/keychains.txt"
    security create-keychain -p "$keychain_password" "$keychain"
    security unlock-keychain -p "$keychain_password" "$keychain"
    security set-keychain-settings -lut 21600 "$keychain"
    security import "$tmp/certificate.p12" -k "$keychain" -P "$MACOS_CERT_PASSWORD" -T /usr/bin/codesign
    security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$keychain_password" "$keychain" >/dev/null
    # Preserve the runner's existing search list so system identities still resolve.
    keychains=()
    while IFS= read -r line; do
      line="${line#*\"}"; line="${line%\"*}"
      [ -z "$line" ] || keychains+=("$line")
    done < "$tmp/keychains.txt"
    security list-keychains -d user -s "$keychain" "${keychains[@]}"
    sudo security add-trusted-cert -d -r trustRoot -p codeSign -k "$keychain" "$tmp/cert.pem"
    security find-identity -v -p codesigning "$keychain" | grep -F "$identity" >/dev/null
    printf 'APPLE_SIGNING_IDENTITY=%s\n' "$identity" >> "$GITHUB_ENV"
    ;;
  cleanup)
    : "${RUNNER_TEMP:?RUNNER_TEMP is required}"
    tmp="$RUNNER_TEMP/ghostreel-signing"
    if [ -f "$tmp/cert.pem" ]; then
      sudo security remove-trusted-cert -d "$tmp/cert.pem" || true
    fi
    if [ -f "$tmp/keychains.txt" ]; then
      keychains=()
      while IFS= read -r line; do
        line="${line#*\"}"; line="${line%\"*}"
        [ -z "$line" ] || keychains+=("$line")
      done < "$tmp/keychains.txt"
      security list-keychains -d user -s "${keychains[@]}"
    fi
    [ ! -e "$tmp/build.keychain-db" ] || security delete-keychain "$tmp/build.keychain-db"
    rm -rf "$tmp"
    ;;
  *)
    echo 'Usage: scripts/macos-signing.sh create <directory> | import | cleanup' >&2
    exit 1
    ;;
esac
