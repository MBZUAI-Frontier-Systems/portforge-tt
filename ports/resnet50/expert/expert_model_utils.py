# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_py_test_utils_common.utility_functions import is_blackhole  # models.common re-exports this module; the sandbox hides models/ except models/common


def is_blackhole_p100(device):
    is_p100 = (
        is_blackhole() and device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y != 130
    )
    return is_p100
