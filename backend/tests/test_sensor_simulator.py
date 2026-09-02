from __future__ import annotations

import math
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx

import sensor_simulator


class SensorSimulatorTests(unittest.TestCase):
    def test_nodes_form_a_regular_hexagon(self) -> None:
        nodes = sensor_simulator.build_hexagon_nodes(
            center_latitude=8.48225,
            center_longitude=4.54225,
            radius_meters=140,
        )

        self.assertEqual(list(nodes), [f"NODE_{index:02d}" for index in range(1, 7)])
        distances = []
        for node in nodes.values():
            north_meters = (node["lat"] - 8.48225) * 111_320
            east_meters = (
                (node["lng"] - 4.54225)
                * 111_320
                * math.cos(math.radians(8.48225))
            )
            distances.append(math.hypot(north_meters, east_meters))

        for distance in distances:
            self.assertAlmostEqual(distance, 140, delta=0.03)

    def test_batch_only_uses_simulator_owned_nodes(self) -> None:
        timestamp = datetime(2026, 8, 31, 12, 30, 15)
        batch = sensor_simulator.build_telemetry_batch(timestamp)

        self.assertEqual(len(batch), 4)
        self.assertEqual({row["Timestamp"] for row in batch}, {"2026-08-31 12:30:15"})
        self.assertEqual(
            {row["Node_ID"] for row in batch},
            set(sensor_simulator.SIMULATED_NODE_IDS),
        )
        self.assertTrue(
            {"NODE_01", "NODE_02"}.isdisjoint(row["Node_ID"] for row in batch)
        )
        for row in batch:
            node = sensor_simulator.NODES[row["Node_ID"]]
            self.assertEqual(row["Latitude"], node["lat"])
            self.assertEqual(row["Longitude"], node["lng"])

    def test_nodes_four_to_six_are_inside_fut_minna(self) -> None:
        coordinates = sensor_simulator.build_fut_minna_node_coordinates()

        self.assertEqual(set(coordinates), {"NODE_04", "NODE_05", "NODE_06"})
        for node_id, coordinate in coordinates.items():
            self.assertGreaterEqual(
                coordinate["lat"], sensor_simulator.FUT_MINNA_LATITUDE_BOUNDS[0]
            )
            self.assertLessEqual(
                coordinate["lat"], sensor_simulator.FUT_MINNA_LATITUDE_BOUNDS[1]
            )
            self.assertGreaterEqual(
                coordinate["lng"], sensor_simulator.FUT_MINNA_LONGITUDE_BOUNDS[0]
            )
            self.assertLessEqual(
                coordinate["lng"], sensor_simulator.FUT_MINNA_LONGITUDE_BOUNDS[1]
            )
            self.assertEqual(sensor_simulator.NODES[node_id]["lat"], coordinate["lat"])
            self.assertEqual(sensor_simulator.NODES[node_id]["lng"], coordinate["lng"])

    @patch("sensor_simulator.telemetry_batch_exists", return_value=False)
    def test_transport_error_is_retried_with_backoff(self, _exists: MagicMock) -> None:
        client = MagicMock()
        execute = client.table.return_value.insert.return_value.execute
        execute.side_effect = [
            httpx.ReadError(
                "temporary TLS read failure",
                request=httpx.Request("POST", "https://example.test/rest/v1/data"),
            ),
            MagicMock(),
        ]
        sleep = MagicMock()

        sensor_simulator.insert_telemetry_batch(
            client,
            [{"Timestamp": "2026-08-31 12:30:15", "Node_ID": "NODE_01"}],
            max_attempts=3,
            base_delay_seconds=0.25,
            sleep=sleep,
        )

        self.assertEqual(execute.call_count, 2)
        sleep.assert_called_once_with(0.25)

    def test_database_errors_are_not_hidden_by_transport_retry(self) -> None:
        client = MagicMock()
        client.table.return_value.insert.return_value.execute.side_effect = ValueError(
            "invalid payload"
        )

        with self.assertRaisesRegex(ValueError, "invalid payload"):
            sensor_simulator.insert_telemetry_batch(
                client,
                [{"Timestamp": "2026-08-31 12:30:15", "Node_ID": "NODE_01"}],
                max_attempts=3,
                sleep=MagicMock(),
            )


if __name__ == "__main__":
    unittest.main()
