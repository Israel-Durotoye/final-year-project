from __future__ import annotations

import math
import unittest
import random
from datetime import datetime, timedelta
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

        self.assertEqual(list(nodes), [f"NODE_{index:02d}" for index in range(1, 8)])
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

    def test_simulator_nodes_are_inside_fut_minna(self) -> None:
        coordinates = sensor_simulator.build_fut_minna_node_coordinates()

        self.assertEqual(set(coordinates), set(sensor_simulator.SIMULATED_NODE_IDS))
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

    def _simulate(self, minutes: int, seed: int = 7) -> dict[str, list[dict]]:
        sensor_simulator.reset_node_states()
        self.addCleanup(sensor_simulator.reset_node_states)
        rng = random.Random(seed)
        start = datetime(2026, 9, 1, 6, 0, 0)
        series: dict[str, list[dict]] = {node: [] for node in sensor_simulator.SIMULATED_NODE_IDS}
        for minute in range(minutes):
            for row in sensor_simulator.build_telemetry_batch(start + timedelta(minutes=minute), rng):
                series[row["Node_ID"]].append(row)
        return series

    def test_consecutive_readings_continue_from_each_other(self) -> None:
        series = self._simulate(240)
        for node_id, rows in series.items():
            temperatures = [row["Temperature_C"] for row in rows]
            steps = [abs(b - a) for a, b in zip(temperatures, temperatures[1:])]
            # Independent random draws would differ by several degrees a minute.
            self.assertLess(max(steps), 1.5, node_id)
            nitrogen = [row["Nitrogen_mg_k"] for row in rows]
            self.assertLess(max(abs(b - a) for a, b in zip(nitrogen, nitrogen[1:])), 6.0, node_id)

    def test_readings_stay_near_the_crop_reference_range(self) -> None:
        series = self._simulate(24 * 60)
        for node_id, rows in series.items():
            profile = sensor_simulator.CROP_PROFILES[sensor_simulator.NODES[node_id]["crop"]]
            for column, key, slack in (
                ("Moisture_%", "moisture", 3.0),
                ("Temperature_C", "temp", 0.0),
                ("Humidity_%", "humidity", 0.0),
                ("Nitrogen_mg_k", "n", 0.0),
                ("Phosphorus_m", "p", 0.0),
                ("Potassium_mg_", "k", 0.0),
            ):
                low, high = profile[key]
                values = [row[column] for row in rows]
                self.assertGreaterEqual(min(values), low - slack - 0.05, (node_id, column))
                self.assertLessEqual(max(values), high + 0.05, (node_id, column))

    def test_moisture_dries_and_is_replenished(self) -> None:
        series = self._simulate(24 * 60)
        for node_id, rows in series.items():
            moisture = [row["Moisture_%"] for row in rows]
            changes = [b - a for a, b in zip(moisture, moisture[1:])]
            # Drying is the normal direction; wetting arrives as occasional jumps.
            self.assertGreater(sum(change < 0 for change in changes), 2 * sum(change > 0 for change in changes), node_id)
            self.assertGreaterEqual(max(changes), 5.0, node_id)

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
