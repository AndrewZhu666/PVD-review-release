"""Fixed-vector parity checks for the released scientific core.

Expected values in this file are archival literals.  They are intentionally
not computed by reimplementing the production formulas in test helpers.
"""

from __future__ import annotations

import pytest
import torch

from path_opd import ActionContract, FlowSchedule, PathOPD


def test_action_contract_matches_archived_executable_crop() -> None:
    model_tensor = torch.tensor(
        [
            [
                [0.0, 1.0, 100.0, 101.0],
                [2.0, 3.0, 102.0, 103.0],
                [200.0, 201.0, 202.0, 203.0],
            ]
        ]
    )

    actual = ActionContract(chunk_length=2, action_dims=2).crop(model_tensor)

    expected = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("steps", "expected_times", "expected_weights"),
    [
        (4, (1.0, 0.75, 0.5, 0.25), (0.25, 0.25, 0.25, 0.25)),
        (
            8,
            (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125),
            (0.125,) * 8,
        ),
    ],
)
def test_uniform_schedule_matches_archived_time_grid(
    steps: int,
    expected_times: tuple[float, ...],
    expected_weights: tuple[float, ...],
) -> None:
    schedule = FlowSchedule.uniform(steps)

    assert schedule.times == expected_times
    assert schedule.weights == expected_weights


def test_weighted_path_loss_matches_archived_fixed_vector() -> None:
    # The third horizon position and last two action dimensions are padding.
    # Large disagreements there prove that the executable-domain crop happens
    # before the reduction.
    student = torch.tensor(
        [
            [
                [[1.0, 2.0, 900.0, 901.0], [3.0, 4.0, 902.0, 903.0], [904.0] * 4],
                [[2.0, 0.0, 910.0, 911.0], [-2.0, 4.0, 912.0, 913.0], [914.0] * 4],
            ]
        ],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            [
                [[0.0, 0.0, -900.0, -901.0], [0.0, 0.0, -902.0, -903.0], [-904.0] * 4],
                [[1.0, 1.0, -910.0, -911.0], [-1.0, 1.0, -912.0, -913.0], [-914.0] * 4],
            ]
        ],
        requires_grad=True,
    )
    objective = PathOPD(
        ActionContract(chunk_length=2, action_dims=2),
        FlowSchedule(times=(1.0, 0.5), weights=(0.25, 0.75)),
    )

    loss, per_time = objective.loss_from_velocities(student, teacher)

    # Archived oracle: per-time MSEs are 7.5 and 3.0, hence
    # 0.25 * 7.5 + 0.75 * 3.0 = 4.125.
    torch.testing.assert_close(per_time, torch.tensor([7.5, 3.0]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(loss, torch.tensor(4.125), rtol=0.0, atol=0.0)

    loss.backward()
    expected_student_gradient = torch.tensor(
        [
            [
                [[0.125, 0.25, 0.0, 0.0], [0.375, 0.5, 0.0, 0.0], [0.0] * 4],
                [
                    [0.375, -0.375, 0.0, 0.0],
                    [-0.375, 1.125, 0.0, 0.0],
                    [0.0] * 4,
                ],
            ]
        ]
    )
    assert student.grad is not None
    torch.testing.assert_close(student.grad, expected_student_gradient, rtol=0.0, atol=0.0)
    assert teacher.grad is None
