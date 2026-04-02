from __future__ import annotations

from dataclasses import replace
import hashlib
import hmac
from typing import Mapping

from .model import Claim
from .ops import ClaimRetracted
from .utils import canonical_json_dumps


def _signature_payload(claim: Claim) -> dict[str, object]:
    payload = claim.to_dict()
    provenance = dict(payload["provenance"])
    provenance.pop("signature", None)
    provenance.pop("signing_key_id", None)
    payload["provenance"] = provenance
    return payload


def _signature_digest(payload: dict[str, object], secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        canonical_json_dumps(payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def sign_claim(claim: Claim, *, secret: str, key_id: str) -> Claim:
    provenance = dict(claim.provenance)
    provenance["signing_key_id"] = key_id
    unsigned = replace(claim, provenance=provenance)
    provenance["signature"] = _signature_digest(_signature_payload(unsigned), secret)
    return replace(claim, provenance=provenance)


def claim_signature_status(claim: Claim, *, key_secrets: Mapping[str, str]) -> str:
    key_id = claim.provenance.get("signing_key_id")
    signature = claim.provenance.get("signature")
    if not isinstance(key_id, str) or not key_id:
        return "missing"
    if not isinstance(signature, str) or not signature:
        return "missing"
    secret = key_secrets.get(key_id)
    if secret is None:
        return "unknown_key"
    expected = _signature_digest(_signature_payload(claim), secret)
    if hmac.compare_digest(signature, expected):
        return "verified"
    return "invalid"


def verify_claim_signature(claim: Claim, *, key_secrets: Mapping[str, str]) -> bool:
    return claim_signature_status(claim, key_secrets=key_secrets) == "verified"


def _retraction_signature_payload(retraction: ClaimRetracted) -> dict[str, object]:
    payload = retraction.to_dict()
    payload.pop("signature", None)
    payload.pop("signing_key_id", None)
    return payload


def sign_retraction(retraction: ClaimRetracted, *, secret: str, key_id: str) -> ClaimRetracted:
    unsigned = replace(retraction, signing_key_id=key_id, signature="")
    signature = _signature_digest(_retraction_signature_payload(unsigned), secret)
    return replace(unsigned, signature=signature)


def retraction_signature_status(retraction: ClaimRetracted, *, key_secrets: Mapping[str, str]) -> str:
    key_id = retraction.signing_key_id
    signature = retraction.signature
    if not isinstance(key_id, str) or not key_id:
        return "missing"
    if not isinstance(signature, str) or not signature:
        return "missing"
    secret = key_secrets.get(key_id)
    if secret is None:
        return "unknown_key"
    expected = _signature_digest(_retraction_signature_payload(retraction), secret)
    if hmac.compare_digest(signature, expected):
        return "verified"
    return "invalid"


def verify_retraction_signature(retraction: ClaimRetracted, *, key_secrets: Mapping[str, str]) -> bool:
    return retraction_signature_status(retraction, key_secrets=key_secrets) == "verified"
