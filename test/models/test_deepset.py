# Copyright 2022 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
import jax.numpy as jnp
import jax

import netket as nk
import netket.nn as nknn


def test_deepset_model_output():
    # make sure that the output of a model is flattened
    ma = nk.models.DeepSetMLP(features_phi=(16,), features_rho=(16,))
    x = np.zeros((2, 1024, 16, 3))

    pars = ma.init(nk.jax.PRNGKey(), x)
    out = ma.apply(pars, x)
    assert out.shape == (2, 1024)

    ma = nk.models.DeepSetMLP(features_phi=16, features_rho=16)
    x = np.zeros((2, 1024, 16, 3))

    pars = ma.init(nk.jax.PRNGKey(), x)
    out = ma.apply(pars, x)
    assert out.shape == (2, 1024)

    ma = nk.models.DeepSetMLP(features_phi=None, features_rho=None)
    x = np.zeros((2, 1024, 16, 3))

    pars = ma.init(nk.jax.PRNGKey(), x)
    out = ma.apply(pars, x)
    assert out.shape == (2, 1024)


@pytest.mark.parametrize(
    "cusp_exponent", [pytest.param(None, id="cusp=None"), pytest.param(5, id="cusp=5")]
)
@pytest.mark.parametrize(
    "L",
    [
        pytest.param(1.0, id="1D"),
        pytest.param((1.0, 1.0), id="2D-Square"),
        pytest.param((1.0, 0.5), id="2D-Rectangle"),
    ],
)
def test_rel_dist_deepsets(cusp_exponent, L):
    d = len(L) if hasattr(L, "__len__") else 1
    hilb = nk.experimental.hilbert.Particle(
        N=2, geometry=nk.experimental.geometry.Cell(d=d, L=L, pbc=True)
    )
    sdim = len(hilb.domain)
    x = jnp.hstack([jnp.ones(4), -jnp.ones(4)]).reshape(1, -1)
    xp = jnp.roll(x, sdim)
    ds = nk.models.DeepSetRelDistance(
        hilbert=hilb,
        cusp_exponent=cusp_exponent,
        layers_phi=2,
        layers_rho=2,
        features_phi=(10, 10),
        features_rho=(10, 1),
    )
    p = ds.init(jax.random.PRNGKey(42), x)

    np.testing.assert_allclose(ds.apply(p, x), ds.apply(p, xp))


def test_rel_dist_deepsets_error():
    hilb = nk.experimental.hilbert.Particle(
        N=2, geometry=nk.experimental.geometry.Cell(d=1, L=1.0, pbc=True)
    )
    sdim = len(hilb.domain)

    x = jnp.hstack([jnp.ones(4), -jnp.ones(4)]).reshape(1, -1)
    jnp.roll(x, sdim)
    ds = nk.models.DeepSetRelDistance(
        hilbert=hilb,
        layers_phi=3,
        layers_rho=3,
        features_phi=(10, 10),
        features_rho=(10, 1),
        output_activation=nknn.gelu,
    )
    with pytest.raises(ValueError):
        ds.init(jax.random.PRNGKey(42), x)

    with pytest.raises(AssertionError):
        ds = nk.models.DeepSetRelDistance(
            hilbert=hilb,
            layers_phi=2,
            layers_rho=2,
            features_phi=(10, 10),
            features_rho=(10, 2),
        )
        ds.init(jax.random.PRNGKey(42), x)

    with pytest.raises(ValueError):
        ds = nk.models.DeepSetRelDistance(
            hilbert=nk.experimental.hilbert.Particle(
                N=2,
                geometry=nk.experimental.geometry.Cell(d=1, L=1.0, pbc=False),
            ),
            layers_phi=2,
            layers_rho=2,
            features_phi=(10, 10),
            features_rho=(10, 2),
        )
        ds.init(jax.random.PRNGKey(42), x)
