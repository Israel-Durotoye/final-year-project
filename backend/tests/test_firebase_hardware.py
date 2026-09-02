from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from backend.ml import firebase_hardware


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body.read()


class FirebaseHardwareTests(unittest.TestCase):
    def test_push_id_provides_wall_clock_timestamp(self) -> None:
        self.assertEqual(
            firebase_hardware.firebase_push_timestamp("-P0WR_Zl8PkYTF9bskHK"),
            "2026-09-02T09:27:59.025000Z",
        )

    @patch("backend.ml.firebase_hardware.urlopen")
    def test_fetch_normalizes_the_ino_schema(self, urlopen_mock: object) -> None:
        urlopen_mock.return_value = _Response({  # type: ignore[attr-defined]
            "-P0WR_Zl8PkYTF9bskHK": {
                "node_id": "NODE_02",
                "nitrogen": 274,
                "phosphorus": 679,
                "potassium": 677,
                "moisture": 21,
                "temp": 27.5,
                "humidity": 74,
                "ph": 6.4,
                "latitude": "9.532082",
                "longitude": "6.451457",
                "timestamp": 1297,
            }
        })

        rows = firebase_hardware.fetch_hardware_rows("NODE_02", limit=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Node_ID"], "NODE_02")
        self.assertEqual(rows[0]["Temperature_C"], 27.5)
        self.assertEqual(rows[0]["Soil_pH"], 6.4)
        self.assertEqual(rows[0]["Latitude"], 9.532082)
        self.assertEqual(rows[0]["Device_Uptime_Seconds"], 1297)


if __name__ == "__main__":
    unittest.main()
