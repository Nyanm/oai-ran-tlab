#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

first_executable() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

LEGACY_ASN1C=${OAI_ASN1C_LEGACY_EXEC:-}
if [[ -z "$LEGACY_ASN1C" ]]; then
  LEGACY_ASN1C=$(first_executable \
    "$SCRIPT_DIR/asn1c-v1.2.0/bin/asn1c" \
    /opt/asn1c-v1.2.0/bin/asn1c \
    /opt/asn1c-v120/bin/asn1c \
    /tmp/asn1c-v120-eAUxmb/src/asn1c/asn1c \
    || true)
fi

LTE_ASN1C=${OAI_ASN1C_LTE_EXEC:-$LEGACY_ASN1C}
NR_ASN1C=${OAI_ASN1C_NR_EXEC:-$LEGACY_ASN1C}
S1AP_ASN1C=${OAI_ASN1C_S1AP_EXEC:-$LEGACY_ASN1C}
NGAP_ASN1C=${OAI_ASN1C_NGAP_EXEC:-$LEGACY_ASN1C}
F1AP_ASN1C=${OAI_ASN1C_F1AP_EXEC:-$LEGACY_ASN1C}
DEFAULT_ASN1C=${OAI_ASN1C_DEFAULT_EXEC:-/opt/asn1c/bin/asn1c}

require_executable() {
  local compiler=$1
  local role=$2

  if [[ -z "$compiler" || ! -x "$compiler" ]]; then
    printf 'error: no executable asn1c found for %s\n' "$role" >&2
    printf 'set OAI_ASN1C_%s_EXEC or OAI_ASN1C_LEGACY_EXEC\n' "$role" >&2
    return 1
  fi
}

asn1c_output_dir() {
  local want_next=0

  for arg in "$@"; do
    if (( want_next )); then
      printf '%s\n' "$arg"
      return 0
    fi

    case "$arg" in
      -D)
        want_next=1
        ;;
      -D*)
        printf '%s\n' "${arg#-D}"
        return 0
        ;;
    esac
  done
}

run_postprocessed_asn1c() {
  local compiler=$1
  local role=$2
  shift
  shift

  require_executable "$compiler" "$role"

  "$compiler" "$@"

  local out_dir
  out_dir=$(asn1c_output_dir "$@" || true)
  if [[ -n "$out_dir" ]]; then
    python3 "$SCRIPT_DIR/oai_asn1c_postprocess.py" "$out_dir"
  fi
}

for arg in "$@"; do
  case "$arg" in
    */openair2/RRC/LTE/MESSAGES/ASN.1/lte-rrc-15.6.0.asn1|openair2/RRC/LTE/MESSAGES/ASN.1/lte-rrc-15.6.0.asn1)
      run_postprocessed_asn1c "$LTE_ASN1C" LTE "$@"
      exit 0
      ;;
    */openair2/RRC/NR/MESSAGES/ASN.1/nr-rrc-17.3.0.asn1|openair2/RRC/NR/MESSAGES/ASN.1/nr-rrc-17.3.0.asn1)
      run_postprocessed_asn1c "$NR_ASN1C" NR "$@"
      exit 0
      ;;
    */openair3/S1AP/MESSAGES/ASN1/R15/s1ap-15.6.0.asn1|openair3/S1AP/MESSAGES/ASN1/R15/s1ap-15.6.0.asn1)
      run_postprocessed_asn1c "$S1AP_ASN1C" S1AP "$@"
      exit 0
      ;;
    */openair3/NGAP/MESSAGES/ASN1/ngap-15.8.0.asn1|openair3/NGAP/MESSAGES/ASN1/ngap-15.8.0.asn1)
      run_postprocessed_asn1c "$NGAP_ASN1C" NGAP "$@"
      exit 0
      ;;
    */openair2/F1AP/MESSAGES/ASN1/R16.3.1/38473-g31.asn|openair2/F1AP/MESSAGES/ASN1/R16.3.1/38473-g31.asn)
      run_postprocessed_asn1c "$F1AP_ASN1C" F1AP "$@"
      exit 0
      ;;
  esac
done

require_executable "$DEFAULT_ASN1C" DEFAULT
exec "$DEFAULT_ASN1C" -no-gen-CBOR "$@"
