#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

LTE_ASN1C=${OAI_ASN1C_LTE_EXEC:-/tmp/asn1c-v120-eAUxmb/src/asn1c/asn1c}
NR_ASN1C=${OAI_ASN1C_NR_EXEC:-/tmp/asn1c-v120-eAUxmb/src/asn1c/asn1c}
S1AP_ASN1C=${OAI_ASN1C_S1AP_EXEC:-/tmp/asn1c-v120-eAUxmb/src/asn1c/asn1c}
NGAP_ASN1C=${OAI_ASN1C_NGAP_EXEC:-/tmp/asn1c-v120-eAUxmb/src/asn1c/asn1c}
F1AP_ASN1C=${OAI_ASN1C_F1AP_EXEC:-/tmp/asn1c-v120-eAUxmb/src/asn1c/asn1c}
DEFAULT_ASN1C=${OAI_ASN1C_DEFAULT_EXEC:-/opt/asn1c/bin/asn1c}

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
  shift

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
      run_postprocessed_asn1c "$LTE_ASN1C" "$@"
      exit 0
      ;;
    */openair2/RRC/NR/MESSAGES/ASN.1/nr-rrc-17.3.0.asn1|openair2/RRC/NR/MESSAGES/ASN.1/nr-rrc-17.3.0.asn1)
      run_postprocessed_asn1c "$NR_ASN1C" "$@"
      exit 0
      ;;
    */openair3/S1AP/MESSAGES/ASN1/R15/s1ap-15.6.0.asn1|openair3/S1AP/MESSAGES/ASN1/R15/s1ap-15.6.0.asn1)
      run_postprocessed_asn1c "$S1AP_ASN1C" "$@"
      exit 0
      ;;
    */openair3/NGAP/MESSAGES/ASN1/ngap-15.8.0.asn1|openair3/NGAP/MESSAGES/ASN1/ngap-15.8.0.asn1)
      run_postprocessed_asn1c "$NGAP_ASN1C" "$@"
      exit 0
      ;;
    */openair2/F1AP/MESSAGES/ASN1/R16.3.1/38473-g31.asn|openair2/F1AP/MESSAGES/ASN1/R16.3.1/38473-g31.asn)
      run_postprocessed_asn1c "$F1AP_ASN1C" "$@"
      exit 0
      ;;
  esac
done

exec "$DEFAULT_ASN1C" -no-gen-CBOR "$@"
