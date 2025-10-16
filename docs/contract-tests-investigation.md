# Contract Test Failure Reproduction

## Context
- Command: `pytest tests/contract/test_openapi_contract.py::test_contract_attach_device`
- Failure: contract schema validation rejects extra fields returned by `POST /v1/devices/attach`.

## Observed Error
```
E   AssertionError: Unexpected properties in response: ['id', 'name', 'secret']
```

## Resolution
- Updated `POST /v1/devices/attach` to return only the contract-defined fields.
- Increased aiohttp `client_max_size` to let the upload handler emit structured 413 responses.

## Verification
- `pytest tests/contract -q`
