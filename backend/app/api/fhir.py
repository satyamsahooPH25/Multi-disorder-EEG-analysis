"""FHIR R4 DiagnosticReport generation for inference results.

Produces a minimal but spec-compliant `DiagnosticReport` resource bundling
`Observation` entries for each predicted disorder probability so it can be
ingested by an EHR via a standard FHIR server.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from ..ml.inference_engine import InferenceResult


SNOMED_CODES = {
    "Healthy": ("17621005", "Normal"),
    "Alzheimer's": ("26929004", "Alzheimer's disease"),
    "Parkinson's": ("49049000", "Parkinson's disease"),
    "FTD": ("230270009", "Frontotemporal dementia"),
    "Schizophrenia": ("58214004", "Schizophrenia"),
}


def build_diagnostic_report(result: InferenceResult,
                            patient_id: str = "anonymous",
                            report_id: str | None = None) -> dict[str, Any]:
    report_id = report_id or str(uuid.uuid4())
    issued = datetime.now(timezone.utc).isoformat()

    observations = []
    for cls_name, prob in result.probabilities.items():
        code, display = SNOMED_CODES.get(cls_name, (cls_name, cls_name))
        obs_id = str(uuid.uuid4())
        observations.append({
            "fullUrl": f"urn:uuid:{obs_id}",
            "resource": {
                "resourceType": "Observation",
                "id": obs_id,
                "status": "final",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "exam",
                        "display": "Exam",
                    }]
                }],
                "code": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": code,
                        "display": display,
                    }],
                    "text": f"AI risk score: {display}",
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "effectiveDateTime": issued,
                "valueQuantity": {
                    "value": round(prob * 100, 2),
                    "unit": "%",
                    "system": "http://unitsofmeasure.org",
                    "code": "%",
                },
                "interpretation": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": "H" if prob > 0.5 else "N",
                        "display": "High" if prob > 0.5 else "Normal",
                    }]
                }],
            },
        })

    diagnostic_id = str(uuid.uuid4())
    diagnostic = {
        "fullUrl": f"urn:uuid:{diagnostic_id}",
        "resource": {
            "resourceType": "DiagnosticReport",
            "id": diagnostic_id,
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "NEU",
                    "display": "Neurology",
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "24708-3",
                    "display": "EEG study",
                }],
                "text": "AI-assisted multi-disorder EEG screen",
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": issued,
            "issued": issued,
            "performer": [{"display": "Temple CognitiveScreen (HAMD-Net)"}],
            "result": [{"reference": f"urn:uuid:{o['resource']['id']}"}
                        for o in observations],
            "conclusion": (
                f"Predicted: {result.predicted_class} "
                f"(confidence {result.confidence:.1%}). "
                f"AI-generated decision support; not a clinical diagnosis."
            ),
            "extension": [{
                "url": "https://temple.ai/fhir/StructureDefinition/eeg-metadata",
                "extension": [
                    {"url": "channels", "valueInteger": len(result.channels)},
                    {"url": "sampleRateHz", "valueDecimal": result.sfreq},
                    {"url": "durationSeconds", "valueDecimal": result.duration_s},
                    {"url": "windowsAnalyzed", "valueInteger": result.n_windows},
                    {"url": "inferenceMs", "valueDecimal": result.inference_ms},
                ],
            }],
        },
    }

    return {
        "resourceType": "Bundle",
        "id": report_id,
        "type": "collection",
        "timestamp": issued,
        "entry": [diagnostic, *observations],
    }
